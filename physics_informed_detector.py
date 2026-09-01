"""
Physics-Informed Detector: Independent/High-Amplitude vs. Correlated/Low-Amplitude
=======================================================================================
Identical computation to the previous version. Renamed to match the
manuscript's primary terminology (Section 3.4, Section 4.3):
  - "Independent/high-amplitude" deviation (formerly "simple attack")
  - "Correlated/low-amplitude" deviation (formerly "stealthy attack")

Implements the physics-informed detector (V2 design, Section 3.4.2): the
physics residual r = P_flow - k*(theta_from - theta_to) is computed directly
on raw input using a per-line learnable susceptance vector k, calibrated only
on normal-labeled samples, then fed as an engineered feature into the
classifier alongside the raw measurements. Evaluated against the Random
Forest baseline across the same 8 seeds, on both IEEE 14-bus and 118-bus
systems.

Requires: pandapower, numpy, pandas, torch, scikit-learn
"""

import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as pn
import torch
import torch.nn as nn
import warnings
import logging
import os
import json

warnings.filterwarnings("ignore")
logging.getLogger("pandapower").setLevel(logging.ERROR)
torch.manual_seed(0)


# =======================================================================
# STEP 1: Topology extraction (from/to bus indices per line, fixed per system)
# =======================================================================
def get_line_topology(case_fn):
    """
    Returns from_bus_idx, to_bus_idx (arrays of length n_line), and n_bus.
    Topology is fixed for a given standard test case regardless of load
    variation, so this only needs to be computed once per system.
    """
    net = case_fn()
    pp.runpp(net, numba=False)
    from_idx = net.line["from_bus"].values.astype(int)
    to_idx = net.line["to_bus"].values.astype(int)
    n_bus = net.bus.shape[0]
    n_line = net.line.shape[0]
    return from_idx, to_idx, n_bus, n_line


# =======================================================================
# STEP 2: Data generation (identical seeded logic to the other scripts)
# =======================================================================
def get_measurements(net):
    v_mag = net.res_bus.vm_pu.values
    v_ang = net.res_bus.va_degree.values
    p_flow = net.res_line.p_from_mw.values
    q_flow = net.res_line.q_from_mvar.values
    return np.concatenate([v_mag, v_ang, p_flow, q_flow])


def add_sensor_noise(measurements, noise_std, rng):
    return measurements + rng.normal(0, noise_std, size=measurements.shape)


def inject_independent_deviation(measurements, rng, noise_std=0.01, attack_ratio=0.08,
                                  snr_factor=4.0):
    """Independent/high-amplitude deviation (engineering label: simple attack)."""
    n_affected = max(1, int(len(measurements) * attack_ratio))
    idx = rng.choice(len(measurements), n_affected, replace=False)
    deviated = measurements.copy()
    deviated[idx] += rng.normal(0, noise_std * snr_factor, size=n_affected)
    return deviated


def inject_correlated_deviation(measurements, n_bus, rng, noise_std=0.01, snr_factor=2.0):
    """Correlated/low-amplitude deviation (engineering label: stealthy attack)."""
    c = rng.uniform(-1, 1, size=n_bus)
    proj = rng.normal(size=(len(measurements), n_bus)) / np.sqrt(n_bus)
    deviation_vector = proj @ c
    deviation_vector = deviation_vector / (np.std(deviation_vector) + 1e-8) * (noise_std * snr_factor)
    return measurements + deviation_vector


def generate_dataset(case_fn, seed, n_normal=3000, n_independent=1000,
                      n_correlated=1000, noise_std=0.01, load_variation=0.15,
                      independent_attack_ratio=0.08, independent_snr=4.0,
                      correlated_snr=2.0):
    """Labels: 0 = normal, 1 = independent/high-amplitude, 2 = correlated/low-amplitude."""
    rng = np.random.default_rng(seed)
    samples, labels = [], []
    for _ in range(n_normal):
        net = case_fn()
        net.load["p_mw"] *= rng.uniform(1 - load_variation, 1 + load_variation, len(net.load))
        try:
            pp.runpp(net, numba=False)
        except Exception:
            continue
        m = add_sensor_noise(get_measurements(net), noise_std, rng)
        samples.append(m); labels.append(0)
    for _ in range(n_independent):
        net = case_fn()
        net.load["p_mw"] *= rng.uniform(1 - load_variation, 1 + load_variation, len(net.load))
        try:
            pp.runpp(net, numba=False)
        except Exception:
            continue
        m = get_measurements(net)
        m = inject_independent_deviation(m, rng, noise_std, independent_attack_ratio, independent_snr)
        m = add_sensor_noise(m, noise_std, rng)
        samples.append(m); labels.append(1)
    for _ in range(n_correlated):
        net = case_fn()
        net.load["p_mw"] *= rng.uniform(1 - load_variation, 1 + load_variation, len(net.load))
        try:
            pp.runpp(net, numba=False)
        except Exception:
            continue
        m = get_measurements(net)
        m = inject_correlated_deviation(m, len(net.bus), rng, noise_std, correlated_snr)
        m = add_sensor_noise(m, noise_std, rng)
        samples.append(m); labels.append(2)
    max_len = max(len(s) for s in samples)
    samples = [np.pad(s, (0, max_len - len(s)), constant_values=0) for s in samples]
    return np.array(samples, dtype=np.float32), np.array(labels)


# =======================================================================
# STEP 3: Physics-informed detector (V2 design — see Section 3.4.2)
# =======================================================================
class PhysicsInformedDetectorV2(nn.Module):
    """
    V2 design: no reconstruction/decoder. Physics residual is computed
    directly on the raw input (not a lossy reconstruction) using a
    per-line learnable susceptance vector, then fed as an engineered
    feature into the classifier alongside the original measurements.
    """
    def __init__(self, input_dim, n_bus, n_line, n_classes=3):
        super().__init__()
        self.n_bus = n_bus
        self.n_line = n_line

        # Per-line learnable susceptance (physically more correct than a
        # single global scalar, since every line has a different reactance).
        self.k_vec = nn.Parameter(torch.full((n_line,), 5.0))

        combined_dim = input_dim + n_line  # raw features + physics residual
        h1 = max(128, combined_dim // 2)
        h2 = max(64, combined_dim // 4)

        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, h1), nn.ReLU(),
            nn.Linear(h1, h2), nn.ReLU(),
            nn.Linear(h2, n_classes),
        )

    def compute_residual(self, x_raw, from_idx, to_idx):
        """
        x_raw must be in PHYSICAL units (not standardized), since the
        DC power-flow relationship only holds in physical units.
        Returns raw residual (batch, n_line) — NOT yet normalized.
        """
        n_bus = self.n_bus
        v_ang = x_raw[:, n_bus:2 * n_bus] * (np.pi / 180.0)
        p_flow = x_raw[:, 2 * n_bus:2 * n_bus + self.n_line]

        theta_from = v_ang[:, from_idx]
        theta_to = v_ang[:, to_idx]
        p_flow_pred = self.k_vec.unsqueeze(0) * (theta_from - theta_to)

        return p_flow - p_flow_pred

    def forward(self, x_scaled, x_raw, from_idx, to_idx, res_scale):
        residual = self.compute_residual(x_raw, from_idx, to_idx)
        residual_norm = residual / res_scale  # normalize so it's on a comparable scale to x_scaled

        combined = torch.cat([x_scaled, residual_norm], dim=1)
        logits = self.classifier(combined)
        return logits, residual


# =======================================================================
# STEP 4: Training and evaluation
# =======================================================================
def train_and_evaluate(X, y, from_idx, to_idx, n_bus, n_line, seed,
                        lambda_phy=0.3,
                        epochs=120, batch_size=128, lr=1e-3):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score, recall_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    # Scaled version for the classifier's general input
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Keep RAW (unscaled) versions too — physics residual must be computed
    # in physical units (degrees, MW), not standardized units.
    X_train_raw_t = torch.tensor(X_train, dtype=torch.float32)
    X_test_raw_t = torch.tensor(X_test, dtype=torch.float32)
    X_train_s_t = torch.tensor(X_train_s, dtype=torch.float32)
    X_test_s_t = torch.tensor(X_test_s, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)

    from_idx_t = torch.tensor(from_idx, dtype=torch.long)
    to_idx_t = torch.tensor(to_idx, dtype=torch.long)

    model = PhysicsInformedDetectorV2(X.shape[1], n_bus, n_line)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ce_loss = nn.CrossEntropyLoss()

    # Fixed residual normalization scale, estimated once from an initial
    # forward pass on normal-labeled training samples (detached, not learned).
    with torch.no_grad():
        normal_mask_init = (y_train_t == 0)
        res_init = model.compute_residual(X_train_raw_t[normal_mask_init], from_idx_t, to_idx_t)
        res_scale = torch.std(res_init) + 1e-6

    n_samples = X_train_s_t.shape[0]
    for epoch in range(epochs):
        perm = torch.randperm(n_samples)
        for i in range(0, n_samples, batch_size):
            idx = perm[i:i + batch_size]
            xb_s, xb_raw, yb = X_train_s_t[idx], X_train_raw_t[idx], y_train_t[idx]

            logits, residual = model(xb_s, xb_raw, from_idx_t, to_idx_t, res_scale)
            data_loss = ce_loss(logits, yb)

            # Physics-consistency loss: calibrate k_vec using ONLY samples
            # labeled normal in this batch. Deviated samples are expected to
            # violate physics — that violation is the detection signal,
            # not something to be minimized away.
            normal_mask = (yb == 0)
            if normal_mask.sum() > 0:
                phy_loss = torch.mean(residual[normal_mask] ** 2) / (res_scale ** 2)
            else:
                phy_loss = torch.tensor(0.0)

            loss = data_loss + lambda_phy * phy_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        logits_test, _ = model(X_test_s_t, X_test_raw_t, from_idx_t, to_idx_t, res_scale)
        probs = torch.softmax(logits_test, dim=1).numpy()
        y_pred = np.argmax(probs, axis=1)

    f1_normal = f1_score(y_test, y_pred, labels=[0], average=None)[0]
    f1_independent = f1_score(y_test, y_pred, labels=[1], average=None)[0]
    f1_correlated = f1_score(y_test, y_pred, labels=[2], average=None)[0]
    recall_independent = recall_score(y_test, y_pred, labels=[1], average=None)[0]
    recall_correlated = recall_score(y_test, y_pred, labels=[2], average=None)[0]

    try:
        auc_independent = roc_auc_score((y_test == 1).astype(int), probs[:, 1])
        auc_correlated = roc_auc_score((y_test == 2).astype(int), probs[:, 2])
    except ValueError:
        auc_independent, auc_correlated = np.nan, np.nan

    return {
        "seed": seed,
        "pi_f1_normal": f1_normal,
        "pi_f1_independent": f1_independent,
        "pi_f1_correlated": f1_correlated,
        "pi_recall_independent": recall_independent,
        "pi_recall_correlated": recall_correlated,
        "pi_auc_independent": auc_independent,
        "pi_auc_correlated": auc_correlated,
        "pi_correlated_minus_independent_f1": f1_correlated - f1_independent,
    }


# =======================================================================
# MAIN
# =======================================================================
if __name__ == "__main__":
    SEEDS = [0, 1, 2, 3, 4, 42, 123, 999]
    os.makedirs("fdia_datasets/physics_informed", exist_ok=True)

    all_results = {}

    for system_name, case_fn in [("14-bus", pn.case14), ("118-bus", pn.case118)]:
        print("=" * 70)
        print(f"Physics-Informed Detector: IEEE {system_name}")
        print("=" * 70)

        from_idx, to_idx, n_bus, n_line = get_line_topology(case_fn)
        print(f"  Topology: n_bus={n_bus}, n_line={n_line}")

        results = []
        for seed in SEEDS:
            print(f"  Seed {seed}...", end=" ", flush=True)
            X, y = generate_dataset(case_fn, seed=seed)
            r = train_and_evaluate(X, y, from_idx, to_idx, n_bus, n_line, seed)
            results.append(r)
            print(f"F1(independent)={r['pi_f1_independent']:.3f}  "
                  f"F1(correlated)={r['pi_f1_correlated']:.3f}  "
                  f"diff={r['pi_correlated_minus_independent_f1']:+.3f}")

        df = pd.DataFrame(results)
        df.to_csv(f"fdia_datasets/physics_informed/{system_name}_pi_results.csv", index=False)
        all_results[system_name] = df

        print(f"\n  --- {system_name} Physics-Informed Aggregate (mean ± std) ---")
        for m in ["pi_f1_normal", "pi_f1_independent", "pi_f1_correlated",
                  "pi_correlated_minus_independent_f1"]:
            print(f"  {m:35s}: {df[m].mean():.3f} ± {df[m].std():.3f}  "
                  f"[{df[m].min():.3f}, {df[m].max():.3f}]")
        print()

    # ---- Save combined summary ----
    summary = {}
    for system_name, df in all_results.items():
        summary[system_name] = {
            col: {"mean": float(df[col].mean()), "std": float(df[col].std())}
            for col in ["pi_f1_normal", "pi_f1_independent", "pi_f1_correlated",
                        "pi_correlated_minus_independent_f1"]
        }
    with open("fdia_datasets/physics_informed/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 70)
    print("Done. Compare these numbers directly against the Random Forest")
    print("baseline results (multiseed_replication.py output) for Table 3.")
    print("=" * 70)
