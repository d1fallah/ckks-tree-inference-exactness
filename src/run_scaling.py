#!/usr/bin/env python3
"""
Controlled scaling ablation, per guide v4.1 Sec 5.5: (a) tree count in
{5,10,20,30,50} at fixed depth 3; (b) depth in {3,4,5,6} at fixed tree
count 50; (c) the combined final configuration (50 trees, depth 6) is
already the mandatory-core result in results/m1_ckks_sweep_simulated.json
and is only referenced here, not recomputed.

Scope, stated up front: this scan trains and evaluates at a SINGLE seed
(42) and the mandatory-core k=5 (notation.py::K_MANDATORY_CORE_DEGREE),
not the full 5-seed x 4-k grid used for the primary result -- retraining
9 additional models x 5 seeds x 4 k would multiply the already-substantial
sweep cost by ~9x for a secondary ablation. This is a scope reduction
driven by compute cost, stated here rather than hidden, consistent with
this project's practice throughout (see Amendment v1.1 Sec 15.5 for the
same kind of documented scope decision on the encrypted-execution arm).

Output: results/m1_scaling_ablation.json
Usage: python run_scaling.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from splits import patient_level_split  # noqa: E402
import ckks_circuit as cc  # noqa: E402
import eval_metrics as em  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
COHORT_PATH = os.path.join(_HERE, "..", "data", "m1_cohort.parquet")
FEATURES_PATH = os.path.join(_HERE, "..", "data", "m1_features.parquet")
OUT_PATH = os.path.join(_HERE, "..", "results", "m1_scaling_ablation.json")

SCALING_SEED = 42
SCALING_K = 5  # notation.py::K_MANDATORY_CORE_DEGREE
TREE_COUNT_GRID = [5, 10, 20, 30, 50]
FIXED_DEPTH_FOR_TREE_SWEEP = 3
DEPTH_GRID = [3, 4, 5, 6]
FIXED_TREES_FOR_DEPTH_SWEEP = 50


def train_and_eval_one(X_train, y_train, X_val, y_val, X_test_df, X_test_mat, y_test_col, art, feature_cols, n_estimators, max_depth, seed, degree):
    thr_grid = np.arange(0.05, 0.96, 0.01)
    clf = xgb.XGBClassifier(
        n_estimators=n_estimators, max_depth=max_depth, tree_method="hist",
        subsample=0.8, colsample_bytree=0.8, base_score=0.5, random_state=seed, eval_metric="logloss",
    )
    clf.fit(X_train, y_train)
    booster = clf.get_booster()
    booster.feature_names = feature_cols

    proba_val = clf.predict_proba(X_val)[:, 1]
    best_thr, best_f1 = 0.5, -1.0
    from sklearn.metrics import f1_score
    for t in thr_grid:
        f = f1_score(y_val, (proba_val >= t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_thr = f, t
    thr_margin = float(np.log(best_thr / (1 - best_thr)))

    dm = xgb.DMatrix(X_test_df, feature_names=feature_cols)
    plain_margin = booster.predict(dm, output_margin=True)

    structure = cc.extract_ensemble_leaf_circuit(booster, art["continuous_features"], tree_depth=max_depth)
    batch_levels = cc.prepare_batch_levels(structure, feature_cols, art["norm_bounds"], art["a_half_width"])
    coeffs = cc.assemble_coefficient_vectors(structure, art["poly_coeffs"]["targets"]["soft_step"], degree)
    ckks_margin = cc.evaluate_circuit_batch(batch_levels, coeffs, structure.leaf_value, X_test_mat, art["a_half_width"])

    metrics = em.per_class_metrics(y_test_col, plain_margin, ckks_margin, thr_margin)
    metrics["n_leaves"] = structure.n_leaves
    metrics["n_estimators"] = n_estimators
    metrics["max_depth"] = max_depth
    return metrics


def main() -> int:
    cohort = pd.read_parquet(COHORT_PATH)
    features = pd.read_parquet(FEATURES_PATH)
    label_cols = [c for c in cohort.columns if c.startswith("label_")]
    label = "label_hypertension_uncomplicated"  # single representative label, same choice as Gate M1-A
    merged = features.merge(cohort[["stay_id", "subject_id"] + label_cols], on="stay_id", how="inner")
    feature_cols = [c for c in features.columns if c != "stay_id"]

    train_ids, val_ids, test_ids = patient_level_split(merged)
    tr = merged["subject_id"].isin(train_ids)
    va = merged["subject_id"].isin(val_ids)
    te = merged["subject_id"].isin(test_ids)
    X_train, y_train = merged.loc[tr, feature_cols], merged.loc[tr, label]
    X_val, y_val = merged.loc[va, feature_cols], merged.loc[va, label]
    X_test_df = merged.loc[te, feature_cols].reset_index(drop=True)
    y_test_col = merged.loc[te, label].to_numpy(dtype=int)
    X_test_mat = X_test_df.to_numpy(dtype=np.float64)

    art = cc.load_circuit_frozen_artifacts()

    results = {
        "scope": (
            f"single seed ({SCALING_SEED}), single label ({label}), fixed k={SCALING_K} -- "
            "see module docstring for the compute-cost rationale"
        ),
        "tree_count_sweep_depth3": [],
        "depth_sweep_trees50": [],
    }

    for n_trees in TREE_COUNT_GRID:
        m = train_and_eval_one(
            X_train, y_train, X_val, y_val, X_test_df, X_test_mat, y_test_col, art, feature_cols,
            n_trees, FIXED_DEPTH_FOR_TREE_SWEEP, SCALING_SEED, SCALING_K,
        )
        results["tree_count_sweep_depth3"].append(m)
        print(f"trees={n_trees:2d} depth={FIXED_DEPTH_FOR_TREE_SWEEP}: delta_f1={m['delta_f1']:.4f} agreement={m['agreement']:.4f} n_leaves={m['n_leaves']}")

    for depth in DEPTH_GRID:
        m = train_and_eval_one(
            X_train, y_train, X_val, y_val, X_test_df, X_test_mat, y_test_col, art, feature_cols,
            FIXED_TREES_FOR_DEPTH_SWEEP, depth, SCALING_SEED, SCALING_K,
        )
        results["depth_sweep_trees50"].append(m)
        print(f"trees={FIXED_TREES_FOR_DEPTH_SWEEP} depth={depth}: delta_f1={m['delta_f1']:.4f} agreement={m['agreement']:.4f} n_leaves={m['n_leaves']}")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nwritten -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
