"""
Multi-Seed Replication of Detectability Results (Random Forest Baseline)
============================================================================
Identical computation to the previous version. Renamed to match the
manuscript's primary terminology (Section 3.2.2, Section 4.2):
  - "Independent/high-amplitude" deviation (formerly "simple attack")
  - "Correlated/low-amplitude" deviation (formerly "stealthy attack")

Reruns the calibrated dataset-generation + baseline-classifier separability
check across multiple random seeds, to establish whether the
"correlated/low-amplitude more detectable than independent/high-amplitude"
pattern (H1) is robust across seeds.

Requires: pandapower, numpy, pandas, scikit-learn
"""

import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as pn
import warnings
import logging
import os
import json

warnings.filterwarnings("ignore")
logging.getLogger("pandapower").setLevel(logging.ERROR)


# ---------------------------------------------------------------------
# Core functions (identical logic to generate_fdia_dataset.py)
# ---------------------------------------------------------------------
def get_measurements(net):
    v_mag = net.res_bus.vm_pu.values
    v_ang = net.res_bus.va_degree.values
    p_flow = net.res_line.p_from_mw.values
    q_flow = net.res_line.q_from_mvar.values
    return np.concatenate([v_mag, v_ang, p_flow, q_flow])


def add_sensor_noise(measurements, noise_std, rng):
    noise = rng.normal(0, noise_std, size=measurements.shape)
    return measurements + noise


def inject_independent_deviation(measurements, rng, noise_std=0.01, attack_ratio=0.08,
                                  snr_factor=4.0):
    """Independent/high-amplitude deviation (engineering label: simple attack)."""
    n_affected = max(1, int(len(measurements) * attack_ratio))
    affected_idx = rng.choice(len(measurements), n_affected, replace=False)
    deviated = measurements.copy()
    perturb = rng.normal(0, noise_std * snr_factor, size=n_affected)
    deviated[affected_idx] += perturb
    return deviated


def inject_correlated_deviation(measurements, n_bus, rng, noise_std=0.01, snr_factor=2.0):
    """Correlated/low-amplitude deviation (engineering label: stealthy attack)."""
    c = rng.uniform(-1, 1, size=n_bus)
    proj = rng.normal(size=(len(measurements), n_bus)) / np.sqrt(n_bus)
    deviation_vector = proj @ c
    deviation_vector = deviation_vector / (np.std(deviation_vector) + 1e-8) * (noise_std * snr_factor)
    return measurements + deviation_vector


def generate_dataset(case_fn, seed, n_normal=3000, n_independent=1000,
                      n_correlated=1000, noise_std=0.01,
                      load_variation=0.15, independent_attack_ratio=0.08,
                      independent_snr=4.0, correlated_snr=2.0):
    """
    Fully seeded dataset generation — every stochastic element tied to `seed`.
    Labels: 0 = normal, 1 = independent/high-amplitude, 2 = correlated/low-amplitude.
    """
    rng = np.random.default_rng(seed)
    samples, labels = [], []
    n_failed = 0

    for _ in range(n_normal):
        net = case_fn()
        net.load["p_mw"] *= rng.uniform(1 - load_variation, 1 + load_variation, len(net.load))
        try:
            pp.runpp(net, numba=False)
        except Exception:
            n_failed += 1
            continue
        m = get_measurements(net)
        m = add_sensor_noise(m, noise_std, rng)
        samples.append(m)
        labels.append(0)

    for _ in range(n_independent):
        net = case_fn()
        net.load["p_mw"] *= rng.uniform(1 - load_variation, 1 + load_variation, len(net.load))
        try:
            pp.runpp(net, numba=False)
        except Exception:
            n_failed += 1
            continue
        m = get_measurements(net)
        m = inject_independent_deviation(m, rng, noise_std=noise_std,
                                          attack_ratio=independent_attack_ratio,
                                          snr_factor=independent_snr)
        m = add_sensor_noise(m, noise_std, rng)
        samples.append(m)
        labels.append(1)

    for _ in range(n_correlated):
        net = case_fn()
        net.load["p_mw"] *= rng.uniform(1 - load_variation, 1 + load_variation, len(net.load))
        try:
            pp.runpp(net, numba=False)
        except Exception:
            n_failed += 1
            continue
        m = get_measurements(net)
        m = inject_correlated_deviation(m, n_bus=len(net.bus), rng=rng,
                                         noise_std=noise_std, snr_factor=correlated_snr)
        m = add_sensor_noise(m, noise_std, rng)
        samples.append(m)
        labels.append(2)

    max_len = max(len(s) for s in samples)
    samples = [np.pad(s, (0, max_len - len(s)), constant_values=0) for s in samples]
    return np.array(samples), np.array(labels), n_failed


def evaluate_seed(X, y_multi, seed):
    """Train baseline RF classifier, return per-class F1 for both binary and multiclass."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score, recall_score

    y_binary = (y_multi > 0).astype(int)

    # Binary
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.2, random_state=seed, stratify=y_binary
    )
    clf = RandomForestClassifier(n_estimators=100, random_state=seed)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    binary_f1_normal = f1_score(y_test, y_pred, pos_label=0)
    binary_f1_deviation = f1_score(y_test, y_pred, pos_label=1)

    # Multiclass
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_multi, test_size=0.2, random_state=seed, stratify=y_multi
    )
    clf = RandomForestClassifier(n_estimators=100, random_state=seed)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    f1_normal = f1_score(y_test, y_pred, labels=[0], average=None)[0]
    f1_independent = f1_score(y_test, y_pred, labels=[1], average=None)[0]
    f1_correlated = f1_score(y_test, y_pred, labels=[2], average=None)[0]
    recall_independent = recall_score(y_test, y_pred, labels=[1], average=None)[0]
    recall_correlated = recall_score(y_test, y_pred, labels=[2], average=None)[0]

    return {
        "seed": seed,
        "binary_f1_normal": binary_f1_normal,
        "binary_f1_deviation": binary_f1_deviation,
        "multiclass_f1_normal": f1_normal,
        "multiclass_f1_independent": f1_independent,
        "multiclass_f1_correlated": f1_correlated,
        "multiclass_recall_independent": recall_independent,
        "multiclass_recall_correlated": recall_correlated,
        "correlated_minus_independent_f1": f1_correlated - f1_independent,
    }


# ---------------------------------------------------------------------
# MAIN: run across multiple seeds for both systems
# ---------------------------------------------------------------------
if __name__ == "__main__":

    SEEDS = [0, 1, 2, 3, 4, 42, 123, 999]  # 8 seeds for reasonable variance estimate
    os.makedirs("fdia_datasets/multiseed", exist_ok=True)

    all_results = {"14-bus": [], "118-bus": []}

    for system_name, case_fn in [("14-bus", pn.case14), ("118-bus", pn.case118)]:
        print("=" * 70)
        print(f"Multi-seed replication: IEEE {system_name}")
        print("=" * 70)

        for seed in SEEDS:
            print(f"  Seed {seed}...", end=" ", flush=True)
            X, y, n_failed = generate_dataset(case_fn, seed=seed)
            result = evaluate_seed(X, y, seed)
            all_results[system_name].append(result)
            print(f"F1(independent)={result['multiclass_f1_independent']:.3f}  "
                  f"F1(correlated)={result['multiclass_f1_correlated']:.3f}  "
                  f"diff={result['correlated_minus_independent_f1']:+.3f}"
                  + (f"  [failed_pf={n_failed}]" if n_failed else ""))

    # ---------------------------------------------------------------
    # Aggregate statistics
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS (mean ± std across seeds)")
    print("=" * 70)

    summary = {}
    for system_name in ["14-bus", "118-bus"]:
        df = pd.DataFrame(all_results[system_name])
        print(f"\n--- IEEE {system_name} (n={len(df)} seeds) ---")

        metrics = ["multiclass_f1_normal", "multiclass_f1_independent",
                   "multiclass_f1_correlated", "correlated_minus_independent_f1"]
        sys_summary = {}
        for m in metrics:
            mean_val = df[m].mean()
            std_val = df[m].std()
            sys_summary[m] = {"mean": mean_val, "std": std_val,
                               "min": df[m].min(), "max": df[m].max()}
            print(f"  {m:35s}: {mean_val:.3f} ± {std_val:.3f}  "
                  f"[{df[m].min():.3f}, {df[m].max():.3f}]")

        n_correlated_wins = (df["correlated_minus_independent_f1"] > 0).sum()
        print(f"  Correlated > Independent F1 in {n_correlated_wins}/{len(df)} seeds")
        sys_summary["correlated_wins_count"] = int(n_correlated_wins)
        sys_summary["n_seeds"] = len(df)

        summary[system_name] = sys_summary

        # Save per-seed results
        df.to_csv(f"fdia_datasets/multiseed/{system_name}_perseed_results.csv", index=False)

    # Save aggregate summary as JSON for easy inclusion in paper
    with open("fdia_datasets/multiseed/summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)

    print("\n" + "=" * 70)
    print("Per-seed CSVs and summary.json saved to fdia_datasets/multiseed/")
    print("=" * 70)
