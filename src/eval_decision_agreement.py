#!/usr/bin/env python3
"""
Added analysis S2.1 (docs/step2_prespec_m1.md, section B): decision-level fidelity of the simulated CKKS
circuit against the plaintext model, per label, seed and degree k in {1, 3, 5, 7, 9}, on the full test split.

Same simulator path as eval_ckks.py / eval_ckks_k1.py (circuit-faithful NumPy; same frozen models, per-(seed,
label) margin thresholds and coefficients), so the margins are the ones behind the reported Delta-F1. New
output per cell: decision disagreement at the frozen threshold (with the two flip directions), probability-space
error, AUROC difference, and a patient-cluster bootstrap of the disagreement rate.

One process handles one seed (python eval_decision_agreement.py <seed>) and writes an AGGREGATE-ONLY file
results/m1_decision_agreement_seed<seed>.json. The bootstrap resamples test subjects with weights drawn from
a fixed generator, identical in every seed, so the seeds can be combined afterwards (summarize_decision_agreement.py).
Nothing patient-level is stored or printed. M1_EVAL_SMOKE_TEST_MAX_ROWS caps the rows for a smoke test and
M1_DA_OUT_DIR redirects the output.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

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
OUT_DIR = os.environ.get("M1_DA_OUT_DIR") or os.path.join(_HERE, "..", "results")
DEGREES = (1, 3, 5, 7, 9)
N_BOOT, BOOT_SEED = 1000, 42


def sigmoid(m: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-m))


def main() -> int:
    seed = int(sys.argv[1])
    assert seed in MODEL_SEEDS, seed
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
        print(f"SMOKE TEST MODE: test set capped at {len(test_df)} rows", flush=True)
    X_test_df = test_df[feature_cols]
    X_test_mat = X_test_df.to_numpy(dtype=np.float64)
    n_test = len(test_df)

    # subject index for the cluster bootstrap (kept in memory only)
    codes, uniques = pd.factorize(test_df["subject_id"])
    n_subj = len(uniques)
    stays_per_subject = np.bincount(codes, minlength=n_subj).astype(np.int64)
    rng = np.random.default_rng(BOOT_SEED)
    w = rng.multinomial(n_subj, np.full(n_subj, 1.0 / n_subj), size=N_BOOT).astype(np.int64)  # (B, n_subj)
    n_boot_stays = w @ stays_per_subject  # (B,)

    with open(BASELINE_PATH, "r", encoding="utf-8") as f:
        baseline = json.load(f)
    thr = {}
    for sr in baseline["per_seed_results"]:
        if sr["seed"] == seed:
            for label, pl in zip(label_cols, sr["per_label"]):
                thr[label] = pl["tuned_threshold_margin"]
    art = cc.load_circuit_frozen_artifacts()
    gamma, a = art["poly_coeffs"]["gamma"], art["a_half_width"]
    coeffs_k1 = fit_power_coefficients(gamma, a, 1)

    cells = []
    for label in label_cols:
        t0 = time.perf_counter()
        clf = xgb.XGBClassifier()
        clf.load_model(os.path.join(MODEL_DIR, f"xgb_float_seed{seed}_{label}.json"))
        booster = clf.get_booster()
        booster.feature_names = feature_cols
        plain = booster.predict(xgb.DMatrix(X_test_df, feature_names=feature_cols), output_margin=True)
        y = test_df[label].to_numpy(dtype=int)
        pp = em.threshold_at_margin(plain, thr[label])
        p_plain = sigmoid(plain)
        structure = cc.extract_ensemble_leaf_circuit(booster, art["continuous_features"], tree_depth=6)
        batch_levels = cc.prepare_batch_levels(structure, feature_cols, art["norm_bounds"], art["a_half_width"])
        for k in DEGREES:
            coeff_src = coeffs_k1 if k == 1 else art["poly_coeffs"]["targets"]["soft_step"]
            coeffs = cc.assemble_coefficient_vectors(structure, coeff_src, k)
            sim = cc.evaluate_circuit_batch(batch_levels, coeffs, structure.leaf_value, X_test_mat, a)
            pc = em.threshold_at_margin(sim, thr[label])
            diff = pp != pc
            per_subject = np.bincount(codes[diff], minlength=n_subj).astype(np.int64)
            boot_rates = (w @ per_subject) / n_boot_stays
            dp = np.abs(p_plain - sigmoid(sim))
            cells.append({
                "seed": seed, "label": label, "k": k,
                "n_test_stays": int(n_test),
                "n_disagree": int(diff.sum()),
                "disagree_rate": float(diff.mean()),
                "n_plain_pos_to_ckks_neg": int(((pp == 1) & (pc == 0)).sum()),
                "n_plain_neg_to_ckks_pos": int(((pp == 0) & (pc == 1)).sum()),
                "predicted_positive_rate_plain": float(pp.mean()),
                "prob_abs_error_mean": float(dp.mean()),
                "prob_abs_error_max": float(dp.max()),
                "auroc_plain": float(roc_auc_score(y, plain)),
                "auroc_ckks": float(roc_auc_score(y, sim)),
                "boot_disagree_rates": [float(r) for r in boot_rates],
            })
        print(f"  seed={seed} {label}: done in {time.perf_counter() - t0:.1f}s", flush=True)

    out = {
        "protocol": "docs/step2_prespec_m1.md section B (added analysis S2.1); same simulator path as eval_ckks.py",
        "backend": "SIMULATED (circuit-faithful NumPy)",
        "seed": seed, "degrees": list(DEGREES), "n_test_stays": int(n_test), "n_test_subjects": int(n_subj),
        "n_bootstrap_resamples": N_BOOT, "bootstrap_seed": BOOT_SEED,
        "bootstrap": "patient-cluster (test subjects resampled; identical weights in every seed)",
        "smoke_test_rows": int(max_rows) if max_rows else None,
        "cells": cells,
        "runtime_seconds": time.perf_counter() - t_start,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"m1_decision_agreement_seed{seed}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f)
    print(f"written -> {path} ({out['runtime_seconds']:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
