"""
M1 -- Conditional k = 1 compatibility check (closure guide 4.5). A SEPARATE ARM: notation.K_APPROX_DEGREE_VALUES stays {3,5,7,9}.

Question: is a degree-1 comparison polynomial mathematically compatible with the construction (approximation interval, parity,
fit method)? If yes, run k = 1 through the same simulated sweep protocol as k in {3,5,7,9}, so the result is a real feasibility
boundary (or a sharper "minimal-complexity comparison suffices" finding). If no, document why. The compatibility analysis is in
docs/k1_compatibility.md; this script is the run.

Construction check (asserted before anything is scored)
  * fit: the SAME method as every other k -- Chebyshev interpolation of soft_step(z, gamma) on [-a, a], gamma from the frozen
    coefficient file, converted to the power basis;
  * parity: soft_step(-z) = 1 - soft_step(z), so the degree-1 interpolant is c0 + c1*z with c0 = 1/2 exactly (to float precision)
    and c1 < 0 -- the branch decision sign(z) is preserved and the polynomial is monotone decreasing;
  * the evaluator applies to degree 1 unchanged (a ct x pt multiply and an add), depth 1.

Protocol otherwise IDENTICAL to eval_ckks.py: 5 seeds x 5 labels, TEST split, frozen validation-tuned margin thresholds shared with the
plaintext arm, per-seed differencing then averaging, 2000-resample paired bootstrap with one shared stay-index draw, |dF1| statistic
(signed values reported alongside). SIMULATED circuit-faithful evaluator.

Output: results/m1_ckks_sweep_k1_simulated.json (the original sweep results/m1_ckks_sweep_simulated.json is untouched).
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import xgboost as xgb

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import ckks_circuit as cc  # noqa: E402
import eval_metrics as em  # noqa: E402
from select_gamma_anchor import fit_power_coefficients  # noqa: E402
from splits import MODEL_SEEDS, patient_level_split  # noqa: E402

COHORT_PATH = os.path.join(_HERE, "..", "data", "m1_cohort.parquet")
FEATURES_PATH = os.path.join(_HERE, "..", "data", "m1_features.parquet")
MODEL_DIR = os.path.join(_HERE, "..", "data", "models")
BASELINE_PATH = os.path.join(_HERE, "..", "results", "m1_plaintext_baseline.json")
OUT_PATH = os.path.join(_HERE, "..", "results", "m1_ckks_sweep_k1_simulated.json")

K = 1
N_BOOTSTRAP, BOOTSTRAP_SEED = 2000, 42
UTILITY_TOLERANCE = 0.03


def main() -> int:
    t_start = time.perf_counter()
    art = cc.load_circuit_frozen_artifacts()
    a = art["a_half_width"]
    gamma = art["poly_coeffs"]["gamma"]

    # ---- construction check ----
    coeffs_k1 = fit_power_coefficients(gamma, a, K)
    c0, c1 = coeffs_k1["k=1"]["coefficients_power_basis"]
    assert abs(c0 - 0.5) < 1e-9, f"parity check failed: c0={c0} (expected 1/2)"
    assert c1 < 0, f"monotonicity check failed: c1={c1} (expected < 0)"
    print(f"construction check: degree-1 interpolant = {c0:.6f} + ({c1:.6f}) z on [-{a}, {a}] "
          f"(c0 = 1/2, c1 < 0: sign(z) preserved, monotone decreasing): PASS")
    lo_val, hi_val = c0 + c1 * a, c0 - c1 * a
    print(f"  value at z=+a: {lo_val:.4f}, at z=-a: {hi_val:.4f} (outside [0,1]: the line over/undershoots the step)")

    cohort = pd.read_parquet(COHORT_PATH)
    features = pd.read_parquet(FEATURES_PATH)
    label_cols = [c for c in cohort.columns if c.startswith("label_")]
    merged = features.merge(cohort[["stay_id", "subject_id"] + label_cols], on="stay_id", how="inner")
    feature_cols = [c for c in features.columns if c != "stay_id"]
    _, _, test_ids = patient_level_split(merged)
    test_df = merged[merged["subject_id"].isin(test_ids)].reset_index(drop=True)
    X_test_df = test_df[feature_cols]
    X_test_mat = X_test_df.to_numpy(dtype=np.float64)
    y_true = test_df[label_cols].to_numpy(dtype=int)

    with open(BASELINE_PATH, "r", encoding="utf-8") as f:
        baseline = json.load(f)
    threshold_margin = {s: {} for s in MODEL_SEEDS}
    for sr in baseline["per_seed_results"]:
        for label, pl in zip(label_cols, sr["per_label"]):
            threshold_margin[sr["seed"]][label] = pl["tuned_threshold_margin"]

    margin_plain = {s: {} for s in MODEL_SEEDS}
    margin_k1 = {s: {} for s in MODEL_SEEDS}
    for seed in MODEL_SEEDS:
        for label in label_cols:
            t0 = time.perf_counter()
            clf = xgb.XGBClassifier()
            clf.load_model(os.path.join(MODEL_DIR, f"xgb_float_seed{seed}_{label}.json"))
            booster = clf.get_booster()
            booster.feature_names = feature_cols
            margin_plain[seed][label] = booster.predict(xgb.DMatrix(X_test_df, feature_names=feature_cols), output_margin=True)
            structure = cc.extract_ensemble_leaf_circuit(booster, art["continuous_features"], tree_depth=6)
            batch_levels = cc.prepare_batch_levels(structure, feature_cols, art["norm_bounds"], art["a_half_width"])
            coeffs = cc.assemble_coefficient_vectors(structure, coeffs_k1, K)
            margin_k1[seed][label] = cc.evaluate_circuit_batch(batch_levels, coeffs, structure.leaf_value, X_test_mat, a)
            print(f"  seed={seed} {label}: done in {time.perf_counter() - t0:.1f}s", flush=True)

    pp_all, pc_all, per_seed = [], [], []
    for seed in MODEL_SEEDS:
        pp = np.column_stack([em.threshold_at_margin(margin_plain[seed][l], threshold_margin[seed][l]) for l in label_cols])
        pc = np.column_stack([em.threshold_at_margin(margin_k1[seed][l], threshold_margin[seed][l]) for l in label_cols])
        pp_all.append(pp)
        pc_all.append(pc)
        f_p, f_c = em.macro_f1(y_true, pp), em.macro_f1(y_true, pc)
        per_seed.append({"seed": seed, "f1_plain": f_p, "f1_ckks": f_c, "delta_f1": em.per_seed_delta_f1(f_p, f_c),
                         "signed_delta_f1_ckks_minus_plain": f_c - f_p,
                         "agreement_mean_over_labels": float(np.mean([em.prediction_agreement(margin_plain[seed][l], margin_k1[seed][l]) for l in label_cols]))})
    boot = em.multi_seed_paired_bootstrap_delta_f1(y_true, pp_all, pc_all, n_resamples=N_BOOTSTRAP, seed=BOOTSTRAP_SEED)
    mean_delta = float(np.mean([r["delta_f1"] for r in per_seed]))
    passes = bool(mean_delta <= UTILITY_TOLERANCE)

    out = {
        "protocol": "closure guide 4.5; identical to eval_ckks.py except k=1 (separate arm; K_APPROX_DEGREE_VALUES unchanged)",
        "backend": "SIMULATED (circuit-faithful numpy, NOT real ciphertext execution)",
        "k": K, "gamma": gamma, "a_half_width": a,
        "degree1_polynomial": {"c0": c0, "c1": c1, "value_at_plus_a": lo_val, "value_at_minus_a": hi_val},
        "n_test_stays": int(len(test_df)), "n_bootstrap_resamples": N_BOOTSTRAP, "seeds": list(MODEL_SEEDS),
        "per_seed": per_seed, "delta_f1_mean_over_seeds": mean_delta,
        "signed_delta_f1_mean_over_seeds": float(np.mean([r["signed_delta_f1_ckks_minus_plain"] for r in per_seed])),
        "paired_bootstrap": boot, "utility_tolerance": UTILITY_TOLERANCE, "meets_utility_tolerance": passes,
        "verdict": ("k=1 PASSES the utility tolerance: no infeasible boundary is reached at the cheapest degree; k*=3 is the lowest degree "
                    "of the locked family {3,5,7,9}, not a feasibility boundary"
                    if passes else
                    "k=1 FAILS the utility tolerance: a real infeasible boundary exists below the locked family, so k*=3 is the lowest feasible degree"),
        "runtime_seconds": round(time.perf_counter() - t_start, 1),
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nk=1: DeltaF1_mean={mean_delta:.4f} (signed {out['signed_delta_f1_mean_over_seeds']:+.4f})  "
          f"95% CI=[{boot['ci_95_lower']:.4f}, {boot['ci_95_upper']:.4f}]  tolerance<=0.03: {passes}")
    print(out["verdict"])
    print(f"written -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
