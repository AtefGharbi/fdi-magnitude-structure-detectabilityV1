"""
FDI Synthetic Dataset Generator for IEEE 14-bus and 118-bus Systems (v4 — terminology-aligned)
==================================================================================================
Identical computation to the previous (v3) version. Renamed to match the
manuscript's primary terminology (Section 3.2.2):
  - "Independent/high-amplitude deviation" (formerly "simple/sparse-random attack")
  - "Correlated/low-amplitude deviation" (formerly "stealthy/structured attack")
The engineering labels are retained in comments/docstrings for continuity,
exactly as the paper does parenthetically.

No numeric behavior changed: same seeds, same calibration (attack_ratio=0.08,
independent_snr=4.0, correlated_snr=2.0), same class labels (0/1/2), same
CSV structure. Only identifiers, docstrings, and printed text are renamed.

Requires: pandapower, numpy, pandas, scikit-learn
"""

import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as pn
import warnings
import logging
import os

warnings.filterwarnings("ignore")
logging.getLogger("pandapower").setLevel(logging.ERROR)
np.random.seed(42)


# ---------------------------------------------------------------------
# STEP 1: Run power flow and extract measurement vector
# ---------------------------------------------------------------------
def get_measurements(net):
    v_mag = net.res_bus.vm_pu.values
    v_ang = net.res_bus.va_degree.values
    p_flow = net.res_line.p_from_mw.values
    q_flow = net.res_line.q_from_mvar.values
    return np.concatenate([v_mag, v_ang, p_flow, q_flow])


# ---------------------------------------------------------------------
# STEP 2: Sensor noise (benign, non-malicious)
# ---------------------------------------------------------------------
def add_sensor_noise(measurements, noise_std=0.01):
    noise = np.random.normal(0, noise_std, size=measurements.shape)
    return measurements + noise


# ---------------------------------------------------------------------
# STEP 3: Independent/high-amplitude deviation
# (engineering label: simple/sparse-random attack)
# ---------------------------------------------------------------------
def inject_independent_deviation(measurements, noise_std=0.01, attack_ratio=0.08,
                                  snr_factor=4.0, seed=None):
    """
    Independent/high-amplitude deviation (Section 3.2.2): perturbs a randomly
    re-selected subset of measurements with additive noise at snr_factor
    multiples of the sensor-noise floor. Higher snr_factor = larger per-channel
    magnitude = easier to detect in isolation.
    """
    rng = np.random.default_rng(seed)
    n_affected = max(1, int(len(measurements) * attack_ratio))
    affected_idx = rng.choice(len(measurements), n_affected, replace=False)

    deviated = measurements.copy()
    perturb = rng.normal(0, noise_std * snr_factor, size=n_affected)
    deviated[affected_idx] += perturb

    mask = np.zeros(len(measurements))
    mask[affected_idx] = 1
    return deviated, mask


# ---------------------------------------------------------------------
# STEP 4: Correlated/low-amplitude deviation
# (engineering label: stealthy/structured attack)
# ---------------------------------------------------------------------
def inject_correlated_deviation(measurements, n_bus, noise_std=0.01,
                                 snr_factor=2.0, seed=None):
    """
    Correlated/low-amplitude deviation (Section 3.2.2): a structured, correlated
    perturbation (low-rank projection from a random state-shift vector) whose
    per-element magnitude is calibrated to snr_factor * noise_std — i.e., only
    marginally above the sensor-noise floor. What makes this condition
    "correlated" is that the deviation is spread across many measurements
    simultaneously via a shared low-rank structure, rather than concentrated
    in a few independently perturbed channels, and its per-channel magnitude
    is deliberately close to normal sensor variation.
    """
    rng = np.random.default_rng(seed)
    c = rng.uniform(-1, 1, size=n_bus)

    proj = rng.normal(size=(len(measurements), n_bus)) / np.sqrt(n_bus)
    deviation_vector = proj @ c

    # Calibrate so std of the deviation vector equals snr_factor * noise_std
    deviation_vector = deviation_vector / (np.std(deviation_vector) + 1e-8) * (noise_std * snr_factor)

    deviated = measurements + deviation_vector
    mask = (np.abs(deviation_vector) > noise_std).astype(int)
    return deviated, mask


# ---------------------------------------------------------------------
# STEP 5: Dataset generator
# ---------------------------------------------------------------------
def generate_dataset(case_fn, n_normal=3000, n_independent=1000,
                      n_correlated=1000, noise_std=0.01,
                      load_variation=0.15,
                      independent_attack_ratio=0.08, independent_snr=4.0,
                      correlated_snr=2.0):
    """
    Labels: 0 = normal, 1 = independent/high-amplitude, 2 = correlated/low-amplitude
    (unchanged numeric convention from the previous version — only names differ).
    """
    samples, labels = [], []
    n_failed = 0
    seed_counter = 0

    # --- Normal samples ---
    for _ in range(n_normal):
        net = case_fn()
        net.load["p_mw"] *= np.random.uniform(1 - load_variation, 1 + load_variation,
                                               len(net.load))
        try:
            pp.runpp(net, numba=False)
        except Exception:
            n_failed += 1
            continue
        m = get_measurements(net)
        m = add_sensor_noise(m, noise_std)
        samples.append(m)
        labels.append(0)

    # --- Independent/high-amplitude samples ---
    for _ in range(n_independent):
        net = case_fn()
        net.load["p_mw"] *= np.random.uniform(1 - load_variation, 1 + load_variation,
                                               len(net.load))
        try:
            pp.runpp(net, numba=False)
        except Exception:
            n_failed += 1
            continue
        m = get_measurements(net)
        m, _ = inject_independent_deviation(m, noise_std=noise_std,
                                             attack_ratio=independent_attack_ratio,
                                             snr_factor=independent_snr, seed=seed_counter)
        seed_counter += 1
        m = add_sensor_noise(m, noise_std)
        samples.append(m)
        labels.append(1)

    # --- Correlated/low-amplitude samples ---
    for _ in range(n_correlated):
        net = case_fn()
        net.load["p_mw"] *= np.random.uniform(1 - load_variation, 1 + load_variation,
                                               len(net.load))
        try:
            pp.runpp(net, numba=False)
        except Exception:
            n_failed += 1
            continue
        m = get_measurements(net)
        m, _ = inject_correlated_deviation(m, n_bus=len(net.bus), noise_std=noise_std,
                                            snr_factor=correlated_snr, seed=seed_counter)
        seed_counter += 1
        m = add_sensor_noise(m, noise_std)
        samples.append(m)
        labels.append(2)

    if n_failed > 0:
        print(f"  Note: {n_failed} power-flow runs failed to converge and were skipped.")

    max_len = max(len(s) for s in samples)
    samples = [np.pad(s, (0, max_len - len(s)), constant_values=0) for s in samples]

    X = np.array(samples)
    y = np.array(labels)
    return X, y


def to_binary_labels(y):
    return (y > 0).astype(int)


def save_dataset(X, y, filename_prefix, output_dir="fdia_datasets"):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(X)
    df["label_multiclass"] = y
    df["label_binary"] = to_binary_labels(y)
    filepath = os.path.join(output_dir, f"{filename_prefix}.csv")
    df.to_csv(filepath, index=False)
    print(f"Saved {len(df)} samples to {filepath}")
    print(f"  Normal: {(y == 0).sum()} | Independent/high-amplitude: {(y == 1).sum()} "
          f"| Correlated/low-amplitude: {(y == 2).sum()}")
    return filepath


def make_train_test_split(X, y, test_size=0.2, seed=42):
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


# ---------------------------------------------------------------------
# STEP 6: Built-in separability check (runs automatically after generation)
# ---------------------------------------------------------------------
def run_separability_check(X, y_multi, name):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report

    y_binary = to_binary_labels(y_multi)

    print(f"\n{'='*50}\n{name} — Binary (Normal vs Any Deviation)\n{'='*50}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
    )
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=["Normal", "Deviation"]))

    print(f"{name} — Multiclass (Normal vs Independent vs Correlated)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_multi, test_size=0.2, random_state=42, stratify=y_multi
    )
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred,
                                 target_names=["Normal", "Independent/High-Amp",
                                               "Correlated/Low-Amp"]))


# ---------------------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------------------
if __name__ == "__main__":

    # Calibration parameters — unchanged values from the prior version,
    # only the names are updated to match the manuscript's terminology.
    INDEPENDENT_SNR = 4.0   # independent/high-amplitude: 4x noise floor
    CORRELATED_SNR = 2.0    # correlated/low-amplitude: 2x noise floor
    NOISE_STD = 0.01

    print("=" * 70)
    print("Generating IEEE 14-bus FDI dataset...")
    print("=" * 70)
    X14, y14 = generate_dataset(
        pn.case14, n_normal=3000, n_independent=1000, n_correlated=1000,
        noise_std=NOISE_STD, independent_snr=INDEPENDENT_SNR, correlated_snr=CORRELATED_SNR
    )
    save_dataset(X14, y14, "ieee14_fdia_dataset")
    run_separability_check(X14, y14, "14-bus")

    print("\n" + "=" * 70)
    print("Generating IEEE 118-bus FDI dataset...")
    print("=" * 70)
    X118, y118 = generate_dataset(
        pn.case118, n_normal=3000, n_independent=1000, n_correlated=1000,
        noise_std=NOISE_STD, independent_snr=INDEPENDENT_SNR, correlated_snr=CORRELATED_SNR
    )
    save_dataset(X118, y118, "ieee118_fdia_dataset")
    run_separability_check(X118, y118, "118-bus")

    # --- Train/test splits for downstream model use ---
    X14_train, X14_test, y14_train, y14_test = make_train_test_split(X14, to_binary_labels(y14))
    X118_train, X118_test, y118_train, y118_test = make_train_test_split(X118, to_binary_labels(y118))

    np.savez("fdia_datasets/ieee14_split.npz",
             X_train=X14_train, X_test=X14_test, y_train=y14_train, y_test=y14_test)
    np.savez("fdia_datasets/ieee118_split.npz",
             X_train=X118_train, X_test=X118_test, y_train=y118_train, y_test=y118_test)

    print("\nAll datasets generated, calibrated, and validated.")
    print("Files are in the 'fdia_datasets/' directory.")
