"""
M1 revision experiment E1 -- FEASIBILITY ONLY (before the pre-specification is signed off).

Question: with the coefficient-folding evaluator (ckks_circuit.eval_poly_tenseal_folded), does the
full circuit fit the fixed chain (N=16384, [60,40x7,60], scale 2^40) at each k in {3,5,7,9}?
Nothing about the parameters, the polynomials, the model, or the k-selection rule changes.

Data: the same single VALIDATION row and model (seed 42, hypertension) as probe_multiplicative_depth.py;
no test row is read, no label is used, no utility is computed. This answers "is k=9 executable", which
the pre-specification needs before it can propose any measurement.
Output: results/m1_folded_evaluator_feasibility.json
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import tenseal as ts  # noqa: F401  (imported for the context helpers re-used below)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import ckks_circuit as cc  # noqa: E402
import probe_multiplicative_depth as pmd  # noqa: E402
from notation import K_APPROX_DEGREE_VALUES  # noqa: E402

OUT_PATH = os.path.join(_HERE, "..", "results", "m1_folded_evaluator_feasibility.json")


def poly_levels(ctx, evaluator, degree: int, n_slots: int = 8) -> int:
    rng = np.random.default_rng(0)
    z = ts.ckks_vector(ctx, list(rng.uniform(-0.5, 0.5, n_slots)))
    coeff_mat = rng.uniform(-1.0, 1.0, size=(degree + 1, n_slots))
    out = evaluator(coeff_mat, z)
    return pmd.chain_index(ctx, z) - pmd.chain_index(ctx, out)


def poly_value_check(ctx, degree: int, art, n_slots: int = 64) -> dict:
    """Folded vs original vs numpy on the real Chebyshev coefficients, synthetic z in [-a, a]."""
    rng = np.random.default_rng(1)
    a = art["a_half_width"]
    z = rng.uniform(-a, a, n_slots)
    coeffs = np.asarray(art["poly_coeffs"]["targets"]["soft_step"][f"k={degree}"]["coefficients_power_basis"])
    coeff_mat = np.repeat(coeffs[:, None], n_slots, axis=1)
    ref = cc.eval_poly_numpy(coeff_mat, z)
    folded = np.asarray(cc.eval_poly_tenseal_folded(coeff_mat, ts.ckks_vector(ctx, z.tolist())).decrypt())
    return {"max_abs_error_folded_vs_numpy": float(np.max(np.abs(folded - ref))), "n_slots": n_slots}


def full_circuit(ctx, k, structure, art, row) -> dict:
    coeffs = cc.assemble_coefficient_vectors(structure, art["poly_coeffs"]["targets"]["soft_step"], k)
    t0 = time.perf_counter()
    try:
        enc, timing = cc.evaluate_circuit_tenseal(ctx, structure, coeffs, row, art["norm_bounds"],
                                                  art["a_half_width"], poly_evaluator=cc.eval_poly_tenseal_folded)
        sim = cc.evaluate_circuit_numpy(structure, coeffs, row, art["norm_bounds"], art["a_half_width"])
        return {"completed": True, "abs_error_vs_simulation": float(abs(enc - sim)),
                "same_sign_as_simulation": bool(np.sign(enc) == np.sign(sim)),
                "seconds": round(time.perf_counter() - t0, 3)}
    except Exception as exc:
        return {"completed": False, "exception_type": type(exc).__name__, "exception_message": str(exc),
                "seconds": round(time.perf_counter() - t0, 3)}


def main() -> int:
    import pandas as pd
    import xgboost as xgb
    from splits import patient_level_split

    ctx = pmd.make_context()
    available = len(pmd.COEFF_MOD_BIT_SIZES) - 2
    cohort = pd.read_parquet(pmd.COHORT_PATH)
    features = pd.read_parquet(pmd.FEATURES_PATH)
    label_cols = [c for c in cohort.columns if c.startswith("label_")]
    merged = features.merge(cohort[["stay_id", "subject_id"] + label_cols], on="stay_id", how="inner")
    feature_cols = [c for c in features.columns if c != "stay_id"]
    _, val_ids, _ = patient_level_split(merged)
    row = merged[merged["subject_id"].isin(val_ids)][feature_cols].iloc[0]
    art = cc.load_circuit_frozen_artifacts()
    clf = xgb.XGBClassifier()
    clf.load_model(os.path.join(pmd.MODEL_DIR, f"xgb_float_seed{pmd.PROBE_SEED}_{pmd.PROBE_LABEL}.json"))
    booster = clf.get_booster()
    booster.feature_names = feature_cols
    structure = cc.extract_ensemble_leaf_circuit(booster, art["continuous_features"], tree_depth=6)

    per_k = {}
    for k in K_APPROX_DEGREE_VALUES:
        rec = {
            "poly_levels_original_evaluator": poly_levels(ctx, cc.eval_poly_tenseal, k),
            "poly_levels_folded_evaluator": poly_levels(ctx, cc.eval_poly_tenseal_folded, k),
            "levels_available": available,
            "poly_value_check": poly_value_check(ctx, k, art),
            "full_circuit_folded_one_validation_row": full_circuit(ctx, k, structure, art, row),
        }
        per_k[f"k={k}"] = rec
        print(f"k={k}: poly levels original={rec['poly_levels_original_evaluator']} "
              f"folded={rec['poly_levels_folded_evaluator']}  full circuit: "
              f"{rec['full_circuit_folded_one_validation_row']}")
    out = {"purpose": "E1 feasibility before pre-specification sign-off; no test data, no utility",
           "backend": "real TenSEAL/SEAL ciphertexts", "fixed_parameters": {
               "N": pmd.N_VALUE, "coeff_mod_bit_sizes": pmd.COEFF_MOD_BIT_SIZES, "scale_bits": pmd.SCALE_BITS},
           "probe_patient": "one row of the VALIDATION split (same as probe_multiplicative_depth.py)",
           "probe_model": {"seed": pmd.PROBE_SEED, "label": pmd.PROBE_LABEL}, "per_degree": per_k}
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"written -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
