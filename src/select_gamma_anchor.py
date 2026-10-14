"""
M1 -- Gamma-anchor re-selection on the VALIDATION split (closure guide 3.1).

Executes docs/anchor_reselection_preregistration.md, committed as 6fc0ecb BEFORE this file
existed. Read it first: the candidates, the scoring split, the scoring degree, the metric,
the tie-break and the outcome cascade are fixed there and restated here, not tuned.

Scope and split provenance
  * Rows scored:   the VALIDATION split of splits.patient_level_split (SPLIT_SEED=42, 20% of
                   subjects). The TEST split is never read for scoring; test ids are computed
                   only to assert disjointness (T1).
  * z_ref(P):      frozen TRAIN-split quantiles in results/m1_circuit_params.json. Re-derived
                   from the training split and asserted equal before scoring (T2).
  * Degree:        k = notation.K_MANDATORY_CORE_DEGREE (5) ONLY, so that the anchor is not
                   chosen because of the k-trend it produces.
  * Metric:        mean sign-agreement between the plaintext margin and the simulated-CKKS
                   margin. No labels are used anywhere in this script.
  * Models:        all 25 frozen models in data/models/ (5 seeds x 5 labels).

Simulated, not real: the circuit-faithful numpy evaluator (ckks_circuit.evaluate_circuit_batch),
identical to the one the degree sweep uses.

Writes results/m1_gamma_anchor_selection.json. Nothing else is modified.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from numpy.polynomial import chebyshev as C

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import ckks_circuit as cc  # noqa: E402
from derive_circuit_params import derive_normalization, load_frozen_models, pooled_interval  # noqa: E402
from generate_poly_coefficients import SIGMOID_EDGE_LEVEL, soft_step  # noqa: E402
from notation import K_MANDATORY_CORE_DEGREE  # noqa: E402
from splits import MODEL_SEEDS, patient_level_split  # noqa: E402

COHORT_PATH = os.path.join(_HERE, "..", "data", "m1_cohort.parquet")
FEATURES_PATH = os.path.join(_HERE, "..", "data", "m1_features.parquet")
MODEL_DIR = os.path.join(_HERE, "..", "data", "models")
CIRCUIT_PARAMS = os.path.join(_HERE, "..", "results", "m1_circuit_params.json")
COEFF_PATH = os.path.join(_HERE, "..", "data", "poly_coeffs_v1.json")
OUT_PATH = os.path.join(_HERE, "..", "results", "m1_gamma_anchor_selection.json")

# --- fixed by docs/anchor_reselection_preregistration.md (not adjustable here) -----------------
CANDIDATE_PERCENTILES = [50, 75, 90, 95, 99.9]
SCORE_DEGREE = K_MANDATORY_CORE_DEGREE
TIE_TOLERANCE = 0.002
PREREG_COMMIT = "6fc0ecb"
# ------------------------------------------------------------------------------------------------


def fit_power_coefficients(gamma: float, a: float, degree: int) -> dict:
    """Same fit as generate_poly_coefficients.fit_and_measure, minus the error grid: Chebyshev
    interpolation of the soft step on [-a, a], converted to the power basis the evaluator uses."""
    cheb = C.Chebyshev.interpolate(lambda z: soft_step(z, gamma), deg=degree, domain=[-a, a])
    coef = [float(c) for c in cheb.convert(kind=np.polynomial.Polynomial).coef]
    return {f"k={degree}": {"coefficients_power_basis": coef}}


def load_frames():
    cohort = pd.read_parquet(COHORT_PATH)
    features = pd.read_parquet(FEATURES_PATH)
    label_cols = [c for c in cohort.columns if c.startswith("label_")]
    merged = features.merge(cohort[["stay_id", "subject_id"] + label_cols], on="stay_id", how="inner")
    feature_cols = [c for c in features.columns if c != "stay_id"]
    return merged, feature_cols, label_cols


def run_checks(merged, feature_cols, val_df, train_ids, val_ids, test_ids, frozen_params):
    """Correctness tests from the pre-registration; every one is a hard assertion."""
    # T1 -- no test leakage: the scoring frame is exactly the validation split and shares nothing
    # with the test split (or the training split).
    scoring_subjects = set(val_df["subject_id"])
    if os.environ.get("M1_SELECT_SMOKE_MAX_ROWS"):  # capped smoke frame: must be a subset of validation
        assert scoring_subjects <= set(val_ids), "T1 FAILED: smoke frame leaves the validation split"
    else:  # the real run scores exactly the validation split
        assert scoring_subjects == set(val_ids), "T1 FAILED: scoring frame is not the validation split"
    assert scoring_subjects.isdisjoint(set(test_ids)), "T1 FAILED: scoring frame overlaps the TEST split"
    assert scoring_subjects.isdisjoint(set(train_ids)), "T1 FAILED: scoring frame overlaps the TRAIN split"
    print(f"T1 scoring frame = validation split ({len(scoring_subjects)} subjects), disjoint from test and train: PASS")

    # T2 -- the quantiles are train-derived: re-derive them from the training split alone.
    X_train = merged.loc[merged["subject_id"].isin(train_ids), feature_cols]
    nunique = X_train.nunique()
    binary_features = set(nunique[nunique <= 2].index)
    norm = derive_normalization(X_train)
    models = load_frozen_models(MODEL_DIR)
    fresh = pooled_interval(models, X_train, norm, binary_features)
    frozen_q = frozen_params["approximation_interval"]["abs_diff_quantiles"]
    for q, v in fresh["abs_diff_quantiles"].items():
        assert abs(v - frozen_q[str(q)]) < 1e-6, f"T2 FAILED: p{q} re-derived {v} != frozen {frozen_q[str(q)]}"
    assert abs(fresh["a_half_width"] - frozen_params["approximation_interval"]["a_half_width"]) < 1e-6
    print(f"T2 frozen quantiles re-derived from the TRAINING split alone (n_sampled_rows={fresh['n_sampled_rows']}): PASS")

    # T3 -- reproduces the currently frozen gamma at p75.
    with open(COEFF_PATH, "r", encoding="utf-8") as f:
        frozen_coeffs = json.load(f)
    g75 = float(np.log(SIGMOID_EDGE_LEVEL / (1 - SIGMOID_EDGE_LEVEL)) / frozen_q["75"])
    assert abs(g75 - frozen_coeffs["gamma"]) < 1e-9 * abs(frozen_coeffs["gamma"]), "T3 FAILED"
    print(f"T3 gamma_75 = {g75:.12f} reproduces the frozen gamma {frozen_coeffs['gamma']:.12f}: PASS")


def main() -> int:
    t_start = time.perf_counter()
    merged, feature_cols, label_cols = load_frames()
    train_ids, val_ids, test_ids = patient_level_split(merged)

    # The scoring frame. Test rows are never selected.
    val_df = merged[merged["subject_id"].isin(val_ids)].reset_index(drop=True)
    max_rows = os.environ.get("M1_SELECT_SMOKE_MAX_ROWS")
    if max_rows:  # smoke testing only; a capped run never writes the artifact
        val_df = val_df.iloc[: int(max_rows)].reset_index(drop=True)
        print(f"SMOKE MODE: validation frame capped at {len(val_df)} rows (result will NOT be written)")

    with open(CIRCUIT_PARAMS, "r", encoding="utf-8") as f:
        frozen_params = json.load(f)
    a = frozen_params["approximation_interval"]["a_half_width"]
    quantiles = frozen_params["approximation_interval"]["abs_diff_quantiles"]

    print("Correctness checks (must all pass before any candidate is scored):")
    run_checks(merged, feature_cols, val_df, train_ids, val_ids, test_ids, frozen_params)

    art = cc.load_circuit_frozen_artifacts()
    X_val_df = val_df[feature_cols]
    X_val_mat = X_val_df.to_numpy(dtype=np.float64)

    candidates = {}
    for P in CANDIDATE_PERCENTILES:
        z_ref = quantiles[str(P)]
        gamma = float(np.log(SIGMOID_EDGE_LEVEL / (1 - SIGMOID_EDGE_LEVEL)) / z_ref)
        candidates[P] = {"percentile": P, "z_ref": z_ref, "gamma": gamma,
                         "coeffs": fit_power_coefficients(gamma, a, SCORE_DEGREE),
                         "agreement_per_model": {}, "nmae_per_model": {}}

    n_done = 0
    for seed in MODEL_SEEDS:
        for label in label_cols:
            t0 = time.perf_counter()
            clf = xgb.XGBClassifier()
            clf.load_model(os.path.join(MODEL_DIR, f"xgb_float_seed{seed}_{label}.json"))
            booster = clf.get_booster()
            booster.feature_names = feature_cols
            plain = booster.predict(xgb.DMatrix(X_val_df, feature_names=feature_cols), output_margin=True)
            structure = cc.extract_ensemble_leaf_circuit(booster, art["continuous_features"], tree_depth=6)
            batch_levels = cc.prepare_batch_levels(structure, feature_cols, art["norm_bounds"], art["a_half_width"])
            span = float(plain.max() - plain.min()) or 1.0
            key = f"seed{seed}/{label}"
            for P, c in candidates.items():
                coeffs = cc.assemble_coefficient_vectors(structure, c["coeffs"], SCORE_DEGREE)
                sim = cc.evaluate_circuit_batch(batch_levels, coeffs, structure.leaf_value, X_val_mat, a)
                c["agreement_per_model"][key] = float(np.mean(np.sign(plain) == np.sign(sim)))
                c["nmae_per_model"][key] = float(np.mean(np.abs(sim - plain)) / span)
            n_done += 1
            print(f"  [{n_done:2d}/25] {key} scored in {time.perf_counter() - t0:.1f}s", flush=True)

    # ---- pre-registered decision rule ----------------------------------------------------------
    table = []
    for P, c in candidates.items():
        table.append({
            "percentile": P, "z_ref": c["z_ref"], "gamma": c["gamma"],
            "pooled_mean_agreement": float(np.mean(list(c["agreement_per_model"].values()))),
            "pooled_mean_normalized_mae": float(np.mean(list(c["nmae_per_model"].values()))),
            "agreement_per_model": c["agreement_per_model"],
            "normalized_mae_per_model": c["nmae_per_model"],
        })
    ranked = sorted(table, key=lambda r: -r["pooled_mean_agreement"])
    top = ranked[0]
    tied = [r for r in ranked if top["pooled_mean_agreement"] - r["pooled_mean_agreement"] <= TIE_TOLERANCE]
    # tie-break within tolerance: lower normalized MAE, then larger gamma
    winner = sorted(tied, key=lambda r: (r["pooled_mean_normalized_mae"], -r["gamma"]))[0]
    tie_break_used = len(tied) > 1

    if max_rows:
        print("SMOKE MODE: not writing the artifact.")
        for r in table:
            print(f"  p{r['percentile']}: gamma={r['gamma']:.4f} agreement={r['pooled_mean_agreement']:.4f} nmae={r['pooled_mean_normalized_mae']:.4f}")
        return 0

    result = {
        "protocol": "docs/anchor_reselection_preregistration.md",
        "preregistration_commit": PREREG_COMMIT,
        "scoring_split": "VALIDATION (splits.patient_level_split, SPLIT_SEED=42); test split not read for scoring",
        "n_validation_rows": int(len(val_df)),
        "n_validation_subjects": int(val_df["subject_id"].nunique()),
        "scoring_degree_k": SCORE_DEGREE,
        "metric": "mean sign-agreement, plaintext margin vs simulated-CKKS margin (label-free)",
        "backend": "SIMULATED (circuit-faithful numpy evaluator, NOT real ciphertext execution)",
        "n_models": 25,
        "sigmoid_edge_level_preregistered": SIGMOID_EDGE_LEVEL,
        "approximation_half_width_a": a,
        "tie_tolerance": TIE_TOLERANCE,
        "candidates": table,
        "ranking_by_agreement": [r["percentile"] for r in ranked],
        "tied_within_tolerance": [r["percentile"] for r in tied],
        "tie_break_used": tie_break_used,
        "selected_percentile": winner["percentile"],
        "selected_gamma": winner["gamma"],
        "selected_z_ref": winner["z_ref"],
        "previously_frozen_percentile": 75,
        "reselection_changed_the_anchor": bool(winner["percentile"] != 75),
        "runtime_seconds": round(time.perf_counter() - t_start, 1),
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("\nCandidate table (validation split, k=%d, pooled over 25 models):" % SCORE_DEGREE)
    print(f"  {'P':>6s} {'gamma':>9s} {'agreement':>10s} {'nMAE':>8s}")
    for r in table:
        mark = "  <- selected" if r["percentile"] == winner["percentile"] else ""
        print(f"  {r['percentile']:>6} {r['gamma']:>9.4f} {r['pooled_mean_agreement']:>10.4f} {r['pooled_mean_normalized_mae']:>8.4f}{mark}")
    print(f"\nTop two within tie tolerance {TIE_TOLERANCE}: {[r['percentile'] for r in tied]} (tie-break used: {tie_break_used})")
    print(f"SELECTED anchor: p{winner['percentile']}  (previously p75; changed: {winner['percentile'] != 75})")
    print(f"Wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
