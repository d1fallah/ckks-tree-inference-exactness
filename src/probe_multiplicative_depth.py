"""
M1 -- Empirical multiplicative-depth probe on the LOCKED CKKS parameters (closure guide 3.3).

The claim in Amendment v1.2 sec 17 -- "7 sequential ciphertext multiplications succeed, the 8th fails with
'scale out of bounds'" -- was cited three times in prose but no probe was ever committed. This is that probe,
run against REAL TenSEAL ciphertexts on the locked chain (N=16384, [60,40x7,60], scale 2^40).
Nothing here changes any parameter.

What it measures (real ciphertexts, not simulation):
  1. Sequential squaring until failure, reading the true SEAL chain index after every multiply.
  2. The level cost of ciphertext x plaintext.
  3. For each k in {3,5,7,9}: the level actually consumed by the committed polynomial evaluator
     (ckks_circuit.eval_poly_tenseal), compared with the code-derived count from src/depth_accounting.py.
  4. For each k: whether the committed full circuit (ckks_circuit.evaluate_circuit_tenseal) runs to
     completion on one patient, or raises. k=9 is EXPECTED to raise; that is the measured limitation.

Split provenance: step 4 uses ONE row of the VALIDATION split (no test row is touched) and needs no label.
Output: results/m1_depth_probe.json
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import tenseal as ts
import xgboost as xgb

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import ckks_circuit as cc  # noqa: E402
import depth_accounting as da  # noqa: E402
from notation import K_APPROX_DEGREE_VALUES  # noqa: E402
from splits import patient_level_split  # noqa: E402

COHORT_PATH = os.path.join(_HERE, "..", "data", "m1_cohort.parquet")
FEATURES_PATH = os.path.join(_HERE, "..", "data", "m1_features.parquet")
MODEL_DIR = os.path.join(_HERE, "..", "data", "models")
OUT_PATH = os.path.join(_HERE, "..", "results", "m1_depth_probe.json")

N_VALUE = 16384
COEFF_MOD_BIT_SIZES = [60, 40, 40, 40, 40, 40, 40, 40, 60]  # locked; identical to eval_ckks_encrypted.py
SCALE_BITS = 40
TC128_MAX_BITS_AT_N16384 = 438  # HE Security Standard / SEAL CoeffModulus.MaxBitCount(16384, TC128)
PROBE_SEED, PROBE_LABEL = 42, "label_hypertension_uncomplicated"


def make_context():
    ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=N_VALUE, coeff_mod_bit_sizes=COEFF_MOD_BIT_SIZES)
    ctx.global_scale = 2**SCALE_BITS
    ctx.generate_galois_keys()
    return ctx


def chain_index(ctx, ct) -> int:
    return ctx.seal_context().data.get_context_data(ct.ciphertext()[0].parms_id()).chain_index()


def sequential_probe(ctx) -> dict:
    v = ts.ckks_vector(ctx, [0.5, -0.25])
    fresh = chain_index(ctx, v)
    w, trace, failed_at, failure = v, [], None, None
    for i in range(1, fresh + 3):
        try:
            w = w * w
            trace.append({"multiplication": i, "chain_index_after": chain_index(ctx, w)})
        except Exception as exc:  # the expected failure past the ceiling
            failed_at, failure = i, str(exc)
            break
    u = v * [0.3, 0.3]
    return {
        "fresh_chain_index": fresh,
        "sequential_ct_x_ct_trace": trace,
        "max_successful_sequential_multiplications": len(trace),
        "first_failure_at_multiplication": failed_at,
        "first_failure_message": failure,
        "levels_consumed_by_one_ct_x_pt_multiply": fresh - chain_index(ctx, u),
    }


def measured_poly_levels(ctx, degree: int, n_slots: int = 8) -> int:
    """Levels consumed by the committed polynomial evaluator on a real ciphertext."""
    rng = np.random.default_rng(0)
    z = ts.ckks_vector(ctx, list(rng.uniform(-0.5, 0.5, n_slots)))
    coeff_mat = rng.uniform(-1.0, 1.0, size=(degree + 1, n_slots))
    out = cc.eval_poly_tenseal(coeff_mat, z)
    return chain_index(ctx, z) - chain_index(ctx, out)


def full_circuit_real(ctx, degree_k: int, structure, art, row) -> dict:
    coeffs = cc.assemble_coefficient_vectors(structure, art["poly_coeffs"]["targets"]["soft_step"], degree_k)
    t0 = time.perf_counter()
    try:
        enc, _timing = cc.evaluate_circuit_tenseal(ctx, structure, coeffs, row, art["norm_bounds"], art["a_half_width"])
        sim = cc.evaluate_circuit_numpy(structure, coeffs, row, art["norm_bounds"], art["a_half_width"])
        return {"completed": True, "abs_error_vs_simulation": float(abs(enc - sim)),
                "seconds": round(time.perf_counter() - t0, 3)}
    except Exception as exc:
        return {"completed": False, "exception_type": type(exc).__name__, "exception_message": str(exc),
                "seconds": round(time.perf_counter() - t0, 3)}


def main() -> int:
    da.assert_product_recursion_matches_committed_code()
    ctx = make_context()
    available = da.chain_levels_available(COEFF_MOD_BIT_SIZES)
    seq = sequential_probe(ctx)
    assert seq["fresh_chain_index"] == available, "chain-index accounting disagrees with the prime count"

    # one VALIDATION row, no label: the probe measures depth, not utility
    cohort = pd.read_parquet(COHORT_PATH)
    features = pd.read_parquet(FEATURES_PATH)
    label_cols = [c for c in cohort.columns if c.startswith("label_")]
    merged = features.merge(cohort[["stay_id", "subject_id"] + label_cols], on="stay_id", how="inner")
    feature_cols = [c for c in features.columns if c != "stay_id"]
    _, val_ids, _ = patient_level_split(merged)
    row = merged[merged["subject_id"].isin(val_ids)][feature_cols].iloc[0]

    art = cc.load_circuit_frozen_artifacts()
    clf = xgb.XGBClassifier()
    clf.load_model(os.path.join(MODEL_DIR, f"xgb_float_seed{PROBE_SEED}_{PROBE_LABEL}.json"))
    booster = clf.get_booster()
    booster.feature_names = feature_cols
    structure = cc.extract_ensemble_leaf_circuit(booster, art["continuous_features"], tree_depth=6)

    per_k = {}
    for k in K_APPROX_DEGREE_VALUES:
        derived = da.walk_circuit(k)
        closed = da.closed_form_depth(k)
        measured_poly = measured_poly_levels(ctx, k)
        assert measured_poly == derived["polynomial_depth"], (
            f"k={k}: real evaluator consumed {measured_poly} levels but the code-derived count is "
            f"{derived['polynomial_depth']}: the accounting is wrong")
        total = derived["total_multiplicative_depth"]
        coeff_bits = COEFF_MOD_BIT_SIZES[0] + 40 * total + COEFF_MOD_BIT_SIZES[-1]
        per_k[f"k={k}"] = {
            "code_derived": {"polynomial_depth": derived["polynomial_depth"],
                             "path_product_depth": derived["path_product_depth"],
                             "total_multiplicative_depth": total},
            "closed_form_previously_tabled": closed,
            "closed_form_undercount_levels": total - closed["total_multiplicative_depth"],
            "measured_polynomial_levels_real_ciphertext": measured_poly,
            "levels_available_on_locked_chain": available,
            "fits_locked_chain": bool(total <= available),
            "slack_levels": available - total,
            "coeff_modulus_bits_needed_for_this_depth": coeff_bits,
            "tc128_max_bits_at_N16384": TC128_MAX_BITS_AT_N16384,
            "would_fit_N16384_at_128_bit_security": bool(coeff_bits <= TC128_MAX_BITS_AT_N16384),
            "full_circuit_real_execution_one_patient": full_circuit_real(ctx, k, structure, art, row),
            "polynomial_operation_graph": derived["polynomial_operation_graph"],
            "path_product_operation_graph": derived["path_product_operation_graph"],
        }
        # the measured behaviour must agree with the derived fit/no-fit verdict
        completed = per_k[f"k={k}"]["full_circuit_real_execution_one_patient"]["completed"]
        assert completed == per_k[f"k={k}"]["fits_locked_chain"], (
            f"k={k}: derived fits={per_k[f'k={k}']['fits_locked_chain']} but real execution completed={completed}")
        print(f"k={k}: derived total={total} (closed form {closed['total_multiplicative_depth']})  "
              f"slack={available - total:+d}  real poly levels={measured_poly}  "
              f"full circuit completed={completed}")

    out = {
        "protocol": "closure guide 3.3; probe of the LOCKED parameters, none changed",
        "locked_parameters": {"N_poly_modulus_degree": N_VALUE, "coeff_modulus_bit_sizes": COEFF_MOD_BIT_SIZES,
                              "scale_bits": SCALE_BITS},
        "backend": "REAL TenSEAL/SEAL ciphertexts",
        "evaluator_note": ("The committed evaluator is a balanced power ladder (ckks_circuit._ct_power) with a "
                           "plaintext coefficient multiply per power. It is NOT Paterson-Stockmeyer, which is not "
                           "implemented (a baby-step/giant-step scheme could fold the coefficient multiply in and "
                           "recover a level; noted as possible future work only)."),
        "sequential_multiplication_probe": seq,
        "per_degree": per_k,
        "probe_patient": "one row of the VALIDATION split; no label used",
        "probe_model": {"seed": PROBE_SEED, "label": PROBE_LABEL},
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"sequential probe: {seq['max_successful_sequential_multiplications']} multiplications succeed, "
          f"#{seq['first_failure_at_multiplication']} fails ({seq['first_failure_message']!r}); "
          f"ct x pt costs {seq['levels_consumed_by_one_ct_x_pt_multiply']} level")
    print(f"written -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
