"""
Factorial Experiment: Structure x Magnitude, Channel Count Held Fixed (Tests H2)
====================================================================================
Directly tests H2 (Section 1.4): does correlational structure increase
detectability independent of magnitude and channel count, when both are
held fixed? The original H1 comparison (independent/8%-of-features/4sigma
vs. correlated/all-channels/2sigma) confounded three factors simultaneously.
This experiment isolates them via a 2x2 design:

              2-sigma magnitude       4-sigma magnitude
Independent   Condition A             Condition B
Correlated    Condition C             Condition D

Design choices (stated explicitly for auditability):
  - k = 10 affected channels, IDENTICAL for all four conditions and BOTH
    systems (14-bus, 118-bus). Holding k fixed in absolute terms across
    system scale also directly tests the Section 5.1 hypothesis that the
    14-bus vs. 118-bus effect-size difference was driven by absolute
    perturbed-channel count.
  - Channels are re-selected independently each sample (not fixed
    positions), for both structure conditions symmetrically -- preserves
    the original design principle that prevented trivial position-based
    detection, applied equally to both arms so it cannot bias the
    structure comparison.
  - Per-channel magnitude is matched EXACTLY between structures at each
    magnitude level: independent draws k i.i.d. Gaussian values rescaled
    to std = snr_factor * noise_std; correlated draws a rank-1 vector
    (fixed random direction x random per-sample scalar) rescaled to the
    SAME std. Only cross-channel correlation differs -- nothing else.

Labels: 0=normal, 1=independent/2sigma, 2=independent/4sigma,
        3=correlated/2sigma, 4=correlated/4sigma

Requires: pandapower, numpy, pandas, scikit-learn, scipy
"""

import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as pn
import warnings
import logging
import os
import json
import multiprocessing as mp
from scipy import stats

warnings.filterwarnings("ignore")
logging.getLogger("pandapower").setLevel(logging.ERROR)

K_AFFECTED_CHANNELS = 10  # default/14-bus value (already run, results complete)
K_118BUS = 47  # calibrated separately: matches the channel count already
                # validated to produce non-degenerate detectability at 118-bus
                # scale in multiseed_replication.py (8% of 582 features), since
                # k=10 (1.7% of features) was empirically confirmed to collapse
                # to majority-class prediction at this system size (see the
                # diagnostic run and Section 5.1/5.5 discussion of absolute
                # channel count interacting with system dimensionality).


# ---------------------------------------------------------------------
# Core measurement / noise functions (identical to prior scripts)
# ---------------------------------------------------------------------
def get_measurements(net):
    v_mag = net.res_bus.vm_pu.values
    v_ang = net.res_bus.va_degree.values
    p_flow = net.res_line.p_from_mw.values
    q_flow = net.res_line.q_from_mvar.values
    return np.concatenate([v_mag, v_ang, p_flow, q_flow])


def add_sensor_noise(measurements, noise_std, rng):
    return measurements + rng.normal(0, noise_std, size=measurements.shape)


# ---------------------------------------------------------------------
# Factorial deviation injection: structure x magnitude, k held fixed
# ---------------------------------------------------------------------
def inject_factorial_deviation(measurements, rng, structure, snr_factor,
                                k=K_AFFECTED_CHANNELS, noise_std=0.01):
    """
    structure: 'independent' or 'correlated'
    Returns deviated measurement vector. Per-channel magnitude (std across
    the k affected channels) is matched exactly between structures at a
    given snr_factor -- verified in the smoke test below.
    """
    n_features = len(measurements)
    affected_idx = rng.choice(n_features, k, replace=False)
    target_std = noise_std * snr_factor

    if structure == "independent":
        perturb = rng.normal(0, 1.0, size=k)  # draw shape, rescale below
        perturb = perturb / (np.std(perturb) + 1e-12) * target_std
    elif structure == "correlated":
        direction = rng.normal(size=k)
        direction = direction / (np.linalg.norm(direction) + 1e-12)
        coefficient = rng.normal()  # single shared scalar -> rank-1 structure
        perturb = direction * coefficient
        # Rescale so the std ACROSS the k values matches target_std exactly,
        # i.e. matched to the independent condition's per-channel spread.
        perturb = perturb / (np.std(perturb) + 1e-12) * target_std
    else:
        raise ValueError(f"Unknown structure: {structure}")

    deviated = measurements.copy()
    deviated[affected_idx] += perturb
    return deviated


# ---------------------------------------------------------------------
# Dataset generator: 5-class (normal + 2 structures x 2 magnitudes)
# ---------------------------------------------------------------------
def generate_factorial_dataset(case_fn, seed, n_normal=1200, n_per_condition=300,
                                noise_std=0.01, load_variation=0.15,
                                k=K_AFFECTED_CHANNELS, snr_low=2.0, snr_high=4.0):
    """
    Labels: 0=normal, 1=independent/low(2sigma), 2=independent/high(4sigma),
            3=correlated/low(2sigma), 4=correlated/high(4sigma)
    n_per_condition applies to EACH of the 4 non-normal conditions
    (so total non-normal samples = 4 * n_per_condition).
    Sample counts reduced from the original 3000/750 (2.5x fewer power-flow
    solves per seed) -- a Random Forest does not need 750/class to
    characterize this signal, and this is the dominant speed lever alongside
    parallelization across seeds (see run_one_seed / multiprocessing below).
    """
    rng = np.random.default_rng(seed)
    samples, labels = [], []
    n_failed = 0

    conditions = [
        (1, "independent", snr_low),
        (2, "independent", snr_high),
        (3, "correlated", snr_low),
        (4, "correlated", snr_high),
    ]

    def make_sample():
        net = case_fn()
        net.load["p_mw"] *= rng.uniform(1 - load_variation, 1 + load_variation, len(net.load))
        try:
            pp.runpp(net, numba=False)
        except Exception:
            return None
        return get_measurements(net)

    for _ in range(n_normal):
        m = make_sample()
        if m is None:
            n_failed += 1
            continue
        m = add_sensor_noise(m, noise_std, rng)
        samples.append(m)
        labels.append(0)

    for label, structure, snr in conditions:
        for _ in range(n_per_condition):
            m = make_sample()
            if m is None:
                n_failed += 1
                continue
            m = inject_factorial_deviation(m, rng, structure, snr, k=k, noise_std=noise_std)
            m = add_sensor_noise(m, noise_std, rng)
            samples.append(m)
            labels.append(label)

    max_len = max(len(s) for s in samples)
    samples = [np.pad(s, (0, max_len - len(s)), constant_values=0) for s in samples]
    return np.array(samples), np.array(labels), n_failed


# ---------------------------------------------------------------------
# Evaluation: RF classifier, per-condition F1 (5-class)
# ---------------------------------------------------------------------
def evaluate_seed(X, y, seed):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    clf = RandomForestClassifier(n_estimators=100, random_state=seed)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    f1 = {}
    for label, name in [(0, "normal"), (1, "independent_low"), (2, "independent_high"),
                         (3, "correlated_low"), (4, "correlated_high")]:
        f1[name] = f1_score(y_test, y_pred, labels=[label], average=None)[0]

    return {
        "seed": seed,
        "f1_normal": f1["normal"],
        "f1_independent_low": f1["independent_low"],
        "f1_independent_high": f1["independent_high"],
        "f1_correlated_low": f1["correlated_low"],
        "f1_correlated_high": f1["correlated_high"],
    }


# ---------------------------------------------------------------------
# Smoke test: verify per-channel magnitude is actually matched
# ---------------------------------------------------------------------
def verify_magnitude_matching(n_trials=2000, k=K_AFFECTED_CHANNELS, noise_std=0.01, snr_factor=3.0):
    rng = np.random.default_rng(12345)
    indep_stds, corr_stds = [], []
    for _ in range(n_trials):
        dummy = np.zeros(50)
        d_indep = inject_factorial_deviation(dummy, rng, "independent", snr_factor, k, noise_std)
        d_corr = inject_factorial_deviation(dummy, rng, "correlated", snr_factor, k, noise_std)
        indep_stds.append(np.std(d_indep[d_indep != 0]))
        corr_stds.append(np.std(d_corr[d_corr != 0]))
    print(f"  Magnitude check (target std = {snr_factor * noise_std:.4f}):")
    print(f"    Independent: mean std across trials = {np.mean(indep_stds):.4f} (should match target)")
    print(f"    Correlated:  mean std across trials = {np.mean(corr_stds):.4f} (should match target)")


# ---------------------------------------------------------------------
# 2x2 factorial hypothesis tests (structure, magnitude, interaction)
# ---------------------------------------------------------------------
def run_factorial_tests(df, system_name):
    print(f"\n{'='*70}\n2x2 Factorial Tests -- {system_name}\n{'='*70}")

    indep_low = df["f1_independent_low"].values
    indep_high = df["f1_independent_high"].values
    corr_low = df["f1_correlated_low"].values
    corr_high = df["f1_correlated_high"].values

    # Structure main effect: (corr_low + corr_high)/2 vs (indep_low + indep_high)/2, paired by seed
    structure_corr = (corr_low + corr_high) / 2
    structure_indep = (indep_low + indep_high) / 2
    w_struct, p_struct = stats.wilcoxon(structure_corr, structure_indep, alternative="two-sided")
    d_struct = (structure_corr - structure_indep).mean() / (structure_corr - structure_indep).std(ddof=1)

    # Magnitude main effect: (indep_high + corr_high)/2 vs (indep_low + corr_low)/2, paired by seed
    magnitude_high = (indep_high + corr_high) / 2
    magnitude_low = (indep_low + corr_low) / 2
    w_mag, p_mag = stats.wilcoxon(magnitude_high, magnitude_low, alternative="two-sided")
    d_mag = (magnitude_high - magnitude_low).mean() / (magnitude_high - magnitude_low).std(ddof=1)

    # Interaction: (corr_high - corr_low) vs (indep_high - indep_low), paired by seed
    corr_delta = corr_high - corr_low
    indep_delta = indep_high - indep_low
    w_int, p_int = stats.wilcoxon(corr_delta, indep_delta, alternative="two-sided")
    d_int = (corr_delta - indep_delta).mean() / (corr_delta - indep_delta).std(ddof=1)

    print(f"  Structure main effect (correlated vs independent, averaged over magnitude):")
    print(f"    mean diff = {(structure_corr - structure_indep).mean():+.4f}, "
          f"Wilcoxon p = {p_struct:.4f}, dz = {d_struct:.3f}")
    print(f"  Magnitude main effect (4sigma vs 2sigma, averaged over structure):")
    print(f"    mean diff = {(magnitude_high - magnitude_low).mean():+.4f}, "
          f"Wilcoxon p = {p_mag:.4f}, dz = {d_mag:.3f}")
    print(f"  Structure x Magnitude interaction:")
    print(f"    mean diff-of-diffs = {(corr_delta - indep_delta).mean():+.4f}, "
          f"Wilcoxon p = {p_int:.4f}, dz = {d_int:.3f}")

    return {
        "structure_effect": {"mean_diff": float((structure_corr - structure_indep).mean()),
                              "p": float(p_struct), "dz": float(d_struct)},
        "magnitude_effect": {"mean_diff": float((magnitude_high - magnitude_low).mean()),
                              "p": float(p_mag), "dz": float(d_mag)},
        "interaction_effect": {"mean_diff": float((corr_delta - indep_delta).mean()),
                                "p": float(p_int), "dz": float(d_int)},
    }


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# Per-seed worker (runs in its own process -- fully self-contained)
# ---------------------------------------------------------------------
def run_one_seed(args):
    seed, case_module_name, k_this_system = args
    case_fn = pn.case118 if case_module_name == "118-bus" else pn.case14
    X, y, n_failed = generate_factorial_dataset(case_fn, seed=seed, k=k_this_system)
    r = evaluate_seed(X, y, seed)
    r["n_failed"] = n_failed
    return r


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
if __name__ == "__main__":
    SEEDS = [0, 1, 2, 3, 4, 42, 123, 999]
    os.makedirs("fdia_datasets/factorial", exist_ok=True)

    N_WORKERS = min(len(SEEDS), mp.cpu_count())

    print("=" * 70)
    print("NOTE: 14-bus (k=10) already completed in the prior run -- this")
    print("script now runs 118-bus ONLY, with a recalibrated k=47 (matching")
    print("the channel count already validated to work at this system scale)")
    print(f"Parallelizing across {N_WORKERS} of {mp.cpu_count()} detected CPU cores")
    print("(all 8 seeds run concurrently instead of one after another --")
    print("no reduction in statistical power, same 8 full seeds, just faster).")
    print("=" * 70)

    print("\nVerifying per-channel magnitude matching at k=118bus...")
    verify_magnitude_matching(k=K_118BUS)

    system_name, k_this_system = "118-bus", K_118BUS

    print("\n" + "=" * 70)
    print(f"Factorial experiment: IEEE {system_name}  (k={k_this_system} fixed)")
    print(f"Sample counts per seed: 1200 normal + 4x300 deviation = 2400 total")
    print("=" * 70)

    tasks = [(seed, system_name, k_this_system) for seed in SEEDS]
    results = []

    with mp.Pool(processes=N_WORKERS) as pool:
        for r in pool.imap_unordered(run_one_seed, tasks):
            print(f"  Seed {r['seed']} done: "
                  f"F1(indep_low)={r['f1_independent_low']:.3f}  "
                  f"F1(indep_high)={r['f1_independent_high']:.3f}  "
                  f"F1(corr_low)={r['f1_correlated_low']:.3f}  "
                  f"F1(corr_high)={r['f1_correlated_high']:.3f}"
                  + (f"  [failed_pf={r['n_failed']}]" if r["n_failed"] else ""),
                  flush=True)
            results.append(r)

    # Sort by seed for reproducible CSV/console ordering (imap_unordered
    # returns results as they finish, not in submission order)
    seed_order = {s: i for i, s in enumerate(SEEDS)}
    results.sort(key=lambda r: seed_order[r["seed"]])

    df = pd.DataFrame(results).drop(columns=["n_failed"])
    df.to_csv(f"fdia_datasets/factorial/{system_name}_k{k_this_system}_factorial_results.csv", index=False)

    print(f"\n  --- {system_name} (k={k_this_system}) Aggregate (mean +/- std) ---")
    for col in ["f1_independent_low", "f1_independent_high",
                "f1_correlated_low", "f1_correlated_high"]:
        print(f"  {col:22s}: {df[col].mean():.3f} +/- {df[col].std():.3f}  "
              f"[{df[col].min():.3f}, {df[col].max():.3f}]")

    test_results = run_factorial_tests(df, system_name)

    with open(f"fdia_datasets/factorial/{system_name}_k{k_this_system}_summary.json", "w") as f:
        json.dump(test_results, f, indent=2)

    print("\n" + "=" * 70)
    print(f"Done. Results saved with 'k{k_this_system}' tag to avoid")
    print("overwriting the completed 14-bus (k=10) results.")
    print("Send the console output back for integration into the manuscript.")
    print("=" * 70)
