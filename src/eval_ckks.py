#!/usr/bin/env python3
"""
The k-sweep: CKKS circuit-faithful simulation over the FULL test set, all
5 training seeds, all 4 k in {3,5,7,9}, per m1-0-implementation-lock.md
Amendment v1.1 Sec 15.5 item 1 (the predictive-metrics arm of the sampling
protocol). Real encrypted TenSEAL execution on the pre-registered n=500
subsample is a SEPARATE script (eval_ckks_encrypted.py) that validates this
simulation; nothing here is a real ciphertext measurement.

Per guide Sec 4.1: for EACH seed independently, compute F1_plain and
F1_CKKS on the SAME test set with the SAME frozen (margin-space) thresholds,
then take the difference. Report the mean over seeds. Never average
predictions or F1 across seeds first.

Output: results/m1_ckks_sweep_simulated.json
Usage: python eval_ckks.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from notation import K_APPROX_DEGREE_VALUES  # noqa: E402
from splits import patient_level_split, MODEL_SEEDS  # noqa: E402
import ckks_circuit as cc  # noqa: E402
import eval_metrics as em  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
COHORT_PATH = os.path.join(_HERE, "..", "data", "m1_cohort.parquet")
FEATURES_PATH = os.path.join(_HERE, "..", "data", "m1_features.parquet")
MODEL_DIR = os.path.join(_HERE, "..", "data", "models")
BASELINE_PATH = os.path.join(_HERE, "..", "results", "m1_plaintext_baseline.json")
# Default unchanged. The closure-guide reproduction (3.4) sets M1_EVAL_CKKS_OUT so the ORIGINAL committed sweep result is never overwritten.
OUT_PATH = os.environ.get("M1_EVAL_CKKS_OUT") or os.path.join(_HERE, "..", "results", "m1_ckks_sweep_simulated.json")

N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 42


def main() -> int:
    t_start = time.perf_counter()
    cohort = pd.read_parquet(COHORT_PATH)
    features = pd.read_parquet(FEATURES_PATH)
    label_cols = [c for c in cohort.columns if c.startswith("label_")]
    merged = features.merge(cohort[["stay_id", "subject_id"] + label_cols], on="stay_id", how="inner")
    feature_cols = [c for c in features.columns if c != "stay_id"]

    _, _, test_ids = patient_level_split(merged)
    test_df = merged[merged["subject_id"].isin(test_ids)].reset_index(drop=True)
    max_rows = os.environ.get("M1_EVAL_SMOKE_TEST_MAX_ROWS")
    if max_rows:
        test_df = test_df.iloc[: int(max_rows)].reset_index(drop=True)
        print(f"SMOKE TEST MODE: test set capped at {len(test_df)} rows via M1_EVAL_SMOKE_TEST_MAX_ROWS")
    X_test_df = test_df[feature_cols]
    X_test_mat = X_test_df.to_numpy(dtype=np.float64)
    y_true = test_df[label_cols].to_numpy(dtype=int)
    n_test = len(test_df)
    print(f"test set: {n_test} stays, {len(label_cols)} labels, {len(feature_cols)} features")

    with open(BASELINE_PATH, "r", encoding="utf-8") as f:
        baseline = json.load(f)
    art = cc.load_circuit_frozen_artifacts()

    # margin_plain[seed][label] -> (n_test,) ; margin_ckks[seed][k][label] -> (n_test,)
    margin_plain: dict[int, dict[str, np.ndarray]] = {s: {} for s in MODEL_SEEDS}
    margin_ckks: dict[int, dict[int, dict[str, np.ndarray]]] = {s: {k: {} for k in K_APPROX_DEGREE_VALUES} for s in MODEL_SEEDS}
    threshold_margin: dict[int, dict[str, float]] = {s: {} for s in MODEL_SEEDS}
    per_label_per_seed_per_k: list[dict] = []

    for seed_result in baseline["per_seed_results"]:
        seed = seed_result["seed"]
        for label, plabel in zip(label_cols, seed_result["per_label"]):
            threshold_margin[seed][label] = plabel["tuned_threshold_margin"]

    for si, seed in enumerate(MODEL_SEEDS):
        for label in label_cols:
            t0 = time.perf_counter()
            model_path = os.path.join(MODEL_DIR, f"xgb_float_seed{seed}_{label}.json")
            clf = xgb.XGBClassifier()
            clf.load_model(model_path)
            booster = clf.get_booster()
            booster.feature_names = feature_cols

            dm = xgb.DMatrix(X_test_df, feature_names=feature_cols)
            plain = booster.predict(dm, output_margin=True)
            margin_plain[seed][label] = plain

            structure = cc.extract_ensemble_leaf_circuit(booster, art["continuous_features"], tree_depth=6)
            batch_levels = cc.prepare_batch_levels(structure, feature_cols, art["norm_bounds"], art["a_half_width"])

            thr = threshold_margin[seed][label]
            row = {"seed": seed, "label": label, "n_leaves": structure.n_leaves}
            for k in K_APPROX_DEGREE_VALUES:
                coeffs = cc.assemble_coefficient_vectors(structure, art["poly_coeffs"]["targets"]["soft_step"], k)
                sim = cc.evaluate_circuit_batch(batch_levels, coeffs, structure.leaf_value, X_test_mat, art["a_half_width"])
                margin_ckks[seed][k][label] = sim

                y_col = test_df[label].to_numpy(dtype=int)
                row[f"k={k}"] = em.per_class_metrics(y_col, plain, sim, thr)
            per_label_per_seed_per_k.append(row)
            print(f"  seed={seed} label={label}: n_leaves={structure.n_leaves} done in {time.perf_counter()-t0:.1f}s")

    # Sec 4.1: per-seed macro-F1, per k, in the frozen order; then Sec 4.2's
    # multi-seed paired bootstrap for the headline Delta-F1(k).
    k_sweep_results = {}
    for k in K_APPROX_DEGREE_VALUES:
        pred_plain_per_seed = []
        pred_ckks_per_seed = []
        per_seed_f1 = []
        for seed in MODEL_SEEDS:
            pp = np.column_stack(
                [em.threshold_at_margin(margin_plain[seed][lab], threshold_margin[seed][lab]) for lab in label_cols]
            )
            pc = np.column_stack(
                [em.threshold_at_margin(margin_ckks[seed][k][lab], threshold_margin[seed][lab]) for lab in label_cols]
            )
            pred_plain_per_seed.append(pp)
            pred_ckks_per_seed.append(pc)
            per_seed_f1.append(
                {
                    "seed": seed,
                    "f1_plain": em.macro_f1(y_true, pp),
                    "f1_ckks": em.macro_f1(y_true, pc),
                }
            )
        for r in per_seed_f1:
            r["delta_f1"] = em.per_seed_delta_f1(r["f1_plain"], r["f1_ckks"])

        bootstrap = em.multi_seed_paired_bootstrap_delta_f1(
            y_true, pred_plain_per_seed, pred_ckks_per_seed, n_resamples=N_BOOTSTRAP, seed=BOOTSTRAP_SEED
        )
        k_sweep_results[f"k={k}"] = {
            "per_seed": per_seed_f1,
            "delta_f1_mean_over_seeds": float(np.mean([r["delta_f1"] for r in per_seed_f1])),
            "paired_bootstrap": bootstrap,
            "meets_utility_tolerance_0.03": bool(np.mean([r["delta_f1"] for r in per_seed_f1]) <= 0.03),
        }
        print(
            f"k={k}: DeltaF1_mean={k_sweep_results[f'k={k}']['delta_f1_mean_over_seeds']:.4f}  "
            f"bootstrap 95% CI=[{bootstrap['ci_95_lower']:.4f}, {bootstrap['ci_95_upper']:.4f}]  "
            f"tolerance<=0.03: {k_sweep_results[f'k={k}']['meets_utility_tolerance_0.03']}"
        )

    out = {
        "protocol": "guide v4.1 Sec 4 (per-seed order, Sec 4.1) + paired bootstrap (Sec 4.2, extended to multi-seed per eval_metrics.multi_seed_paired_bootstrap_delta_f1)",
        "backend": "SIMULATED (circuit-faithful numpy, NOT real ciphertext execution -- see Amendment v1.1 Sec 15.5)",
        "n_test_stays": n_test,
        "n_bootstrap_resamples": N_BOOTSTRAP,
        "seeds": list(MODEL_SEEDS),
        "k_values": list(K_APPROX_DEGREE_VALUES),
        "utility_tolerance": 0.03,
        "k_sweep": k_sweep_results,
        "per_label_detail": per_label_per_seed_per_k,
        "total_runtime_seconds": time.perf_counter() - t_start,
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nwritten -> {OUT_PATH}  (total {out['total_runtime_seconds']:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
