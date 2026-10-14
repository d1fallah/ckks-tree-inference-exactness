#!/usr/bin/env python3
"""
Gate M1-A: staged circuit feasibility check, per guide v4.1 Sec 3.10 and
m1-0-implementation-lock.md Sec 11. MUST run and pass (or be reported as
failing, honestly) BEFORE the 50-tree/depth-6 circuit is attempted.

Configuration (locked, Sec 11): 5 trees, depth <= 3, fixed k=5 (the
mandatory-core degree), synthetic data + a small clinical subset. Uses the
SAME frozen circuit artifacts (approximation interval, poly_coeffs_v1.json,
normalization bounds) as the full sweep will -- M1-A validates the frozen
circuit specification at small scale, it does not re-derive a new one.

Pass criteria (Sec 11.1, pre-registered before this script was ever run):
    prediction-agreement (sign of margin) >= 0.95   [gating]
    normalized MAE = mean|s_ckks - s_plain| / range(s_plain) <= 0.10  [gating]
    normalized RMSE <= 0.15                          [gating]
    circuit executes end-to-end                      [gating]

Output: results/m1_gate_a_report.json
Usage: python run_gate_m1a.py
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

_HERE = os.path.dirname(os.path.abspath(__file__))
COHORT_PATH = os.path.join(_HERE, "..", "data", "m1_cohort.parquet")
FEATURES_PATH = os.path.join(_HERE, "..", "data", "m1_features.parquet")
OUT_PATH = os.path.join(_HERE, "..", "results", "m1_gate_a_report.json")

# Locked (Sec 11): M1-A never touches the mandatory-core 50-tree/depth-6
# configuration.
M1A_N_ESTIMATORS = 5
M1A_MAX_DEPTH = 3
M1A_K = 5
M1A_SEED = 42
M1A_LABEL = "label_hypertension_uncomplicated"
N_SYNTHETIC = 300
N_CLINICAL_SUBSET = 300

# Pre-registered gating thresholds (Sec 11.1) -- written before this script
# produced any result.
THRESH_AGREEMENT_MIN = 0.95
THRESH_NORM_MAE_MAX = 0.10
THRESH_NORM_RMSE_MAX = 0.15


def make_synthetic_arm(feature_cols: list[str], continuous_features: set, n: int, seed: int) -> pd.DataFrame:
    """Random feature vectors respecting each column's realistic marginal
    range -- binary columns Bernoulli(0.5), continuous columns uniform over
    the observed training range. Distinct from the real clinical subset arm;
    exercises circuit mechanics without any real-world label structure."""
    rng = np.random.default_rng(seed)
    art = cc.load_circuit_frozen_artifacts()
    data = {}
    for col in feature_cols:
        if col in continuous_features:
            lo, hi = art["norm_bounds"][col]["min"], art["norm_bounds"][col]["max"]
            data[col] = rng.uniform(lo, hi, n)
        else:
            data[col] = rng.integers(0, 2, n).astype(float)
    return pd.DataFrame(data)


def run_arm(name: str, X: pd.DataFrame, structure: cc.TreeCircuitStructure, art: dict, degree: int) -> dict:
    coeffs = cc.assemble_coefficient_vectors(structure, art["poly_coeffs"]["targets"]["soft_step"], degree)
    plain = np.array([cc.evaluate_plaintext_exact(structure, X.iloc[i]) for i in range(len(X))])
    ckks_sim = np.array(
        [
            cc.evaluate_circuit_numpy(structure, coeffs, X.iloc[i], art["norm_bounds"], art["a_half_width"])
            for i in range(len(X))
        ]
    )
    agreement = float(np.mean(np.sign(plain) == np.sign(ckks_sim)))
    rng_span = float(plain.max() - plain.min()) if plain.max() > plain.min() else 1.0
    err = np.abs(ckks_sim - plain)
    norm_mae = float(err.mean() / rng_span)
    norm_rmse = float(np.sqrt((err**2).mean()) / rng_span)
    return {
        "arm": name,
        "n": len(X),
        "agreement_sign": agreement,
        "normalized_mae": norm_mae,
        "normalized_rmse": norm_rmse,
        "score_mae_raw": float(err.mean()),
        "score_rmse_raw": float(np.sqrt((err**2).mean())),
        "plain_margin_range": [float(plain.min()), float(plain.max())],
        "gate_pass": bool(
            agreement >= THRESH_AGREEMENT_MIN
            and norm_mae <= THRESH_NORM_MAE_MAX
            and norm_rmse <= THRESH_NORM_RMSE_MAX
        ),
    }


def main() -> int:
    cohort = pd.read_parquet(COHORT_PATH)
    features = pd.read_parquet(FEATURES_PATH)
    df = features.merge(cohort[["stay_id", "subject_id", M1A_LABEL]], on="stay_id")
    train_ids, val_ids, test_ids = patient_level_split(df.rename(columns={"subject_id": "subject_id"}))

    train_df = df[df["subject_id"].isin(train_ids)]
    test_df = df[df["subject_id"].isin(test_ids)]

    feature_cols = [c for c in features.columns if c != "stay_id"]

    clf = xgb.XGBClassifier(
        n_estimators=M1A_N_ESTIMATORS,
        max_depth=M1A_MAX_DEPTH,
        tree_method="hist",
        subsample=0.8,
        colsample_bytree=0.8,
        base_score=0.5,
        random_state=M1A_SEED,
        eval_metric="logloss",
    )
    clf.fit(train_df[feature_cols], train_df[M1A_LABEL])
    booster = clf.get_booster()
    booster.feature_names = feature_cols

    art = cc.load_circuit_frozen_artifacts()
    structure = cc.extract_ensemble_leaf_circuit(booster, art["continuous_features"], tree_depth=M1A_MAX_DEPTH)

    # Sanity: the extracted structure must reproduce the exact booster margin
    # BEFORE any approximation is introduced. This is the same unit test
    # used during development (numpy circuit == booster.predict exactly).
    probe = test_df[feature_cols].iloc[:50].reset_index(drop=True)
    dm = xgb.DMatrix(probe, feature_names=feature_cols)
    booster_margin = booster.predict(dm, output_margin=True)
    exact_circuit = np.array([cc.evaluate_plaintext_exact(structure, probe.iloc[i]) for i in range(len(probe))])
    extraction_max_err = float(np.max(np.abs(exact_circuit - booster_margin)))

    rng = np.random.default_rng(M1A_SEED)
    clinical_idx = rng.choice(len(test_df), size=min(N_CLINICAL_SUBSET, len(test_df)), replace=False)
    clinical_X = test_df[feature_cols].iloc[clinical_idx].reset_index(drop=True)
    synthetic_X = make_synthetic_arm(feature_cols, art["continuous_features"], N_SYNTHETIC, M1A_SEED)

    results = {
        "config": {
            "n_estimators": M1A_N_ESTIMATORS,
            "max_depth": M1A_MAX_DEPTH,
            "k": M1A_K,
            "seed": M1A_SEED,
            "label": M1A_LABEL,
        },
        "structure_extraction_sanity_check": {
            "description": "exact leaf-circuit vs booster.predict(output_margin=True), zero approximation",
            "max_abs_error": extraction_max_err,
            "pass": bool(extraction_max_err < 1e-4),
        },
        "arms": [
            run_arm("synthetic", synthetic_X, structure, art, M1A_K),
            run_arm("small_clinical_subset", clinical_X, structure, art, M1A_K),
        ],
        "pre_registered_thresholds": {
            "agreement_sign_min": THRESH_AGREEMENT_MIN,
            "normalized_mae_max": THRESH_NORM_MAE_MAX,
            "normalized_rmse_max": THRESH_NORM_RMSE_MAX,
            "source": "m1-0-implementation-lock.md Sec 11.1, written before this experiment ran",
        },
    }
    results["overall_gate_pass"] = bool(
        results["structure_extraction_sanity_check"]["pass"] and all(a["gate_pass"] for a in results["arms"])
    )

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print(f"\nGate M1-A: {'PASS' if results['overall_gate_pass'] else 'FAIL'}")
    print(f"written -> {OUT_PATH}")
    return 0 if results["overall_gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
