#!/usr/bin/env python3
"""
Real CKKS ciphertext execution -- the encrypted-execution arm of Amendment
v1.1 Sec 15.5 item 2. Validates the numpy simulation (eval_ckks.py) against
ACTUAL TenSEAL encrypt -> evaluate -> decrypt, measuring
Agreement(XGB_CKKS_encrypted, XGB_CKKS_simulated), E_score, and real
per-patient latency.

k=9 IS NOT RUN HERE. Real-circuit depth accounting (this session,
2026-09-18) found the branch polynomial's ACTUAL depth cost at k=9 is 5,
not the 4 tabled in m1-0-implementation-lock.md Sec 5 -- the table counted
the balanced power ladder's depth but not the extra level consumed by
multiplying each power by its (per-slot) coefficient before summing.
5 (poly) + 3 (balanced path product) = 8, one level beyond the locked
[60,40x7,60] chain's empirically confirmed 7-level ceiling (verified by a
standalone sequential-multiply probe: depth 7 succeeds, depth 8 raises
"scale out of bounds"). k=3 (depth up to 3), k=5 and k=7 (depth up to 5)
all fit and are numerically validated below; k=9 does not fit under this
implementation's evaluation strategy and is reported as a real, measured
parameter-budget limitation, not worked around by silently loosening the
chain (which would require re-deriving and re-confirming the security
level -- out of scope for this validation pass).

Scope: n=REAL_N samples, ONE representative model (seed=42,
label_hypertension_uncomplicated), same representative choice used
throughout (Gate M1-A, the TFHE arm). A full n=500 x 5-seed x 5-label real
run at ~1-1.4s/sample would take roughly 10 hours; this is a measured-cost
scope reduction in the same spirit as eval_tfhe.py's.

Output: results/m1_encrypted_subsample.json
Usage: python eval_ckks_encrypted.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import tenseal as ts
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from splits import patient_level_split  # noqa: E402
import ckks_circuit as cc  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
COHORT_PATH = os.path.join(_HERE, "..", "data", "m1_cohort.parquet")
FEATURES_PATH = os.path.join(_HERE, "..", "data", "m1_features.parquet")
MODEL_DIR = os.path.join(_HERE, "..", "data", "models")
OUT_PATH = os.path.join(_HERE, "..", "results", "m1_encrypted_subsample.json")

REPRESENTATIVE_SEED = 42
REPRESENTATIVE_LABEL = "label_hypertension_uncomplicated"
REAL_N = 30
K_VALUES_REAL_EXECUTABLE = (3, 5, 7)  # k=9 excluded -- see module docstring
K_VALUE_EXCLUDED = 9
N_VALUE = 16384  # notation.py::N_POLY_MODULUS_DEGREE; not named bare `N` per the notation gate
COEFF_MOD_BIT_SIZES = [60, 40, 40, 40, 40, 40, 40, 40, 60]


def main() -> int:
    cohort = pd.read_parquet(COHORT_PATH)
    features = pd.read_parquet(FEATURES_PATH)
    label_cols = [c for c in cohort.columns if c.startswith("label_")]
    merged = features.merge(cohort[["stay_id", "subject_id"] + label_cols], on="stay_id", how="inner")
    feature_cols = [c for c in features.columns if c != "stay_id"]
    _, _, test_ids = patient_level_split(merged)
    test_df = merged[merged["subject_id"].isin(test_ids)].reset_index(drop=True)

    rng = np.random.default_rng(REPRESENTATIVE_SEED)
    idx = rng.choice(len(test_df), size=REAL_N, replace=False)
    X = test_df[feature_cols].iloc[idx].reset_index(drop=True)

    clf = xgb.XGBClassifier()
    clf.load_model(os.path.join(MODEL_DIR, f"xgb_float_seed{REPRESENTATIVE_SEED}_{REPRESENTATIVE_LABEL}.json"))
    booster = clf.get_booster()
    booster.feature_names = feature_cols
    structure = cc.extract_ensemble_leaf_circuit(booster, cc.load_circuit_frozen_artifacts()["continuous_features"], tree_depth=6)
    art = cc.load_circuit_frozen_artifacts()

    ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=N_VALUE, coeff_mod_bit_sizes=COEFF_MOD_BIT_SIZES)
    ctx.global_scale = 2**40
    ctx.generate_galois_keys()

    by_k = {}
    for k in K_VALUES_REAL_EXECUTABLE:
        coeffs = cc.assemble_coefficient_vectors(structure, art["poly_coeffs"]["targets"]["soft_step"], k)
        sims, encs, phase_totals = [], [], {"encrypt": [], "poly_eval": [], "path_product": [], "sum_decrypt": []}
        for i in range(len(X)):
            sim = cc.evaluate_circuit_numpy(structure, coeffs, X.iloc[i], art["norm_bounds"], art["a_half_width"])
            enc, timing = cc.evaluate_circuit_tenseal(ctx, structure, coeffs, X.iloc[i], art["norm_bounds"], art["a_half_width"])
            sims.append(sim)
            encs.append(enc)
            for phase in phase_totals:
                phase_totals[phase].append(timing[phase])
        sims = np.array(sims)
        encs = np.array(encs)
        err = np.abs(encs - sims)
        rng_span = float(sims.max() - sims.min()) or 1.0
        total_per_sample = [sum(phase_totals[p][i] for p in phase_totals) for i in range(len(X))]
        by_k[f"k={k}"] = {
            "n": len(X),
            "agreement_encrypted_vs_simulated": float(np.mean(np.sign(sims) == np.sign(encs))),
            "e_score_mae_raw": float(err.mean()),
            "e_score_mae_normalized": float(err.mean() / rng_span),
            "e_score_max_abs": float(err.max()),
            "online_latency_seconds_per_patient_mean": float(np.mean(total_per_sample)),
            "online_latency_seconds_per_patient_std": float(np.std(total_per_sample)),
            "phase_breakdown_mean_seconds": {p: float(np.mean(v)) for p, v in phase_totals.items()},
        }
        print(f"k={k}: agreement={by_k[f'k={k}']['agreement_encrypted_vs_simulated']:.4f}  "
              f"E_score(norm MAE)={by_k[f'k={k}']['e_score_mae_normalized']:.6f}  "
              f"latency={by_k[f'k={k}']['online_latency_seconds_per_patient_mean']:.2f}s/patient")

    out = {
        "representative_model": {"seed": REPRESENTATIVE_SEED, "label": REPRESENTATIVE_LABEL},
        "n_real_executions_per_k": REAL_N,
        "k_values_executed": list(K_VALUES_REAL_EXECUTABLE),
        "k_value_excluded": K_VALUE_EXCLUDED,
        "k_excluded_reason": (
            "Real depth accounting: poly_eval depth 5 (not the tabled 4 -- the table omitted the "
            "coefficient-multiply level) + balanced path-product depth 3 = 8, exceeding the locked "
            "N=16384/[60,40x7,60] chain's empirically confirmed 7-level ceiling."
        ),
        "N_poly_modulus_degree": N_VALUE,
        "coeff_modulus_bit_sizes": COEFF_MOD_BIT_SIZES,
        "by_k": by_k,
        "scope_reduction_reason": (
            f"n={REAL_N} on one representative model, not the full 5-seed x 5-label x 500-sample "
            f"grid -- at the measured ~1-1.4s/sample real-execution cost, the full grid would take "
            f"on the order of 10 hours. Consistent with the same measured-cost scoping in eval_tfhe.py."
        ),
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nwritten -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
