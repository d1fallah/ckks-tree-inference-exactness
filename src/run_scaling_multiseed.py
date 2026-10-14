"""
M1 -- Multi-seed scaling analysis with an encrypted-latency arm (closure guide 4.4).

Extends the single-seed pilot in run_scaling.py (results/m1_scaling_ablation.json, kept untouched) to ALL FIVE seeds and adds
REAL encrypted latency per configuration. Without the latency arm the pilot is a utility ablation, not a scaling analysis.

Grid (unchanged from the pilot): tree count {5,10,20,30,50} at depth 3; depth {3,4,5,6} at 50 trees; k = 5 (mandatory core);
one label (hypertension). The (50 trees, depth 3) cell belongs to both sweeps and is computed once. 8 distinct configurations.

Utility arm (SIMULATED circuit-faithful evaluator, same as the sweep)
  * per (seed, configuration): retrain on the train split with random_state=seed, tune the decision threshold on the
    VALIDATION split only, evaluate plaintext vs simulated-CKKS on the TEST split. Same hyperparameters as the pilot.
  * reported: mean +/- SD across the 5 seeds of |dF1| (the locked statistic), of the SIGNED difference f1_ckks - f1_plain,
    and of agreement. Per-seed differencing order is preserved (difference first, then average across seeds).
  * reproducibility gate: the seed-42 cells must reproduce the committed single-seed pilot.

Latency arm (REAL TenSEAL/SEAL ciphertexts)
  * per (seed, configuration): 2 discarded warm-up evaluations, then real encrypted evaluation of the FIRST 10 patients of that
    seed's frozen validation-independent sample (data/m1_real_ckks_samples/seed{S}.csv, test-split rows, fixed before any run).
  * reported: wall-clock per patient (mean over the 10, then mean +/- SD across seeds), poly_eval and path_product buckets,
    input ciphertexts per patient (one per tree level, so it scales with depth) and the output ciphertext, in bytes.
  * a configuration whose code-derived depth exceeds the locked chain is NOT executed and is reported as not fitting.

Output: results/m1_scaling_multiseed.json
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
import ckks_instrumented as ci  # noqa: E402
import depth_accounting as da  # noqa: E402
import eval_metrics as em  # noqa: E402
from splits import MODEL_SEEDS, patient_level_split  # noqa: E402

COHORT_PATH = os.path.join(_HERE, "..", "data", "m1_cohort.parquet")
FEATURES_PATH = os.path.join(_HERE, "..", "data", "m1_features.parquet")
SAMPLE_DIR = os.path.join(_HERE, "..", "data", "m1_real_ckks_samples")
PILOT_PATH = os.path.join(_HERE, "..", "results", "m1_scaling_ablation.json")
OUT_PATH = os.path.join(_HERE, "..", "results", "m1_scaling_multiseed.json")

LABEL = "label_hypertension_uncomplicated"
K = 5  # notation.K_MANDATORY_CORE_DEGREE
CONFIGS = [(5, 3), (10, 3), (20, 3), (30, 3), (50, 3), (50, 4), (50, 5), (50, 6)]  # (n_estimators, max_depth)
N_LATENCY, N_WARM = 10, 2
SLOTS = ci.LOCKED_N // 2
GATE_TOL = 1e-4  # seed-42 cells vs the committed pilot


def train_config(X_train, y_train, X_val, y_val, feature_cols, n_estimators, max_depth, seed):
    """Identical training and validation-only threshold tuning to run_scaling.train_and_eval_one."""
    clf = xgb.XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, tree_method="hist", subsample=0.8,
                            colsample_bytree=0.8, base_score=0.5, random_state=seed, eval_metric="logloss")
    clf.fit(X_train, y_train)
    booster = clf.get_booster()
    booster.feature_names = feature_cols
    proba_val = clf.predict_proba(X_val)[:, 1]
    from sklearn.metrics import f1_score
    best_thr, best_f1 = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.01):
        f = f1_score(y_val, (proba_val >= t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_thr = f, t
    return booster, float(np.log(best_thr / (1 - best_thr)))


def main() -> int:
    t_start = time.perf_counter()
    cohort = pd.read_parquet(COHORT_PATH)
    features = pd.read_parquet(FEATURES_PATH)
    label_cols = [c for c in cohort.columns if c.startswith("label_")]
    merged = features.merge(cohort[["stay_id", "subject_id"] + label_cols], on="stay_id", how="inner")
    feature_cols = [c for c in features.columns if c != "stay_id"]
    train_ids, val_ids, test_ids = patient_level_split(merged)
    tr, va, te = (merged["subject_id"].isin(s) for s in (train_ids, val_ids, test_ids))
    X_train, y_train = merged.loc[tr, feature_cols], merged.loc[tr, LABEL]
    X_val, y_val = merged.loc[va, feature_cols], merged.loc[va, LABEL]
    X_test_df = merged.loc[te, feature_cols].reset_index(drop=True)
    y_test = merged.loc[te, LABEL].to_numpy(dtype=int)
    X_test_mat = X_test_df.to_numpy(dtype=np.float64)
    frame = merged.set_index("stay_id")

    art = cc.load_circuit_frozen_artifacts()
    ctx = ci.make_ckks_context()
    seal = ci.make_seal_handles(ctx)

    cells = []
    for seed in MODEL_SEEDS:
        sample = pd.read_csv(os.path.join(SAMPLE_DIR, f"seed{seed}.csv"))
        lat_X = frame.loc[sample["stay_id"].to_numpy()[:N_LATENCY + N_WARM], feature_cols].reset_index(drop=True)
        for n_trees, depth in CONFIGS:
            t0 = time.perf_counter()
            booster, thr = train_config(X_train, y_train, X_val, y_val, feature_cols, n_trees, depth, seed)
            plain = booster.predict(xgb.DMatrix(X_test_df, feature_names=feature_cols), output_margin=True)
            structure = cc.extract_ensemble_leaf_circuit(booster, art["continuous_features"], tree_depth=depth)
            batch_levels = cc.prepare_batch_levels(structure, feature_cols, art["norm_bounds"], art["a_half_width"])
            coeffs = cc.assemble_coefficient_vectors(structure, art["poly_coeffs"]["targets"]["soft_step"], K)
            sim = cc.evaluate_circuit_batch(batch_levels, coeffs, structure.leaf_value, X_test_mat, art["a_half_width"])
            m = em.per_class_metrics(y_test, plain, sim, thr)
            cell = {"seed": seed, "n_estimators": n_trees, "max_depth": depth, "n_leaves": int(structure.n_leaves),
                    "f1_plain": m["f1_plain"], "f1_ckks": m["f1_ckks"], "abs_delta_f1": m["delta_f1"],
                    "signed_delta_f1_ckks_minus_plain": m["f1_ckks"] - m["f1_plain"], "agreement": m["agreement"]}

            # ---- latency arm: real ciphertexts ----
            derived = da.walk_circuit(K, n_factors=depth)
            cell["code_derived_depth"] = derived["total_multiplicative_depth"]
            fits = derived["total_multiplicative_depth"] <= da.chain_levels_available(ci.LOCKED_CHAIN) and structure.n_leaves <= SLOTS
            cell["fits_locked_chain_and_slots"] = bool(fits)
            if fits:
                coeffs_ct = coeffs
                for i in range(N_WARM):  # discarded
                    ci.evaluate_instrumented(ctx, seal, structure, coeffs_ct, lat_X.iloc[N_LATENCY + i], art["norm_bounds"], art["a_half_width"], decompose=False)
                recs = [ci.evaluate_instrumented(ctx, seal, structure, coeffs_ct, lat_X.iloc[i], art["norm_bounds"], art["a_half_width"], decompose=False)[1]
                        for i in range(N_LATENCY)]
                cell["real_latency"] = {
                    "n_patients": N_LATENCY,
                    "wall_clock_s_mean": float(np.mean([r["wall_clock_total_s"] for r in recs])),
                    "poly_eval_s_mean": float(np.mean([r["poly_eval_s"] for r in recs])),
                    "path_product_s_mean": float(np.mean([r["path_product_s"] for r in recs])),
                    "input_ciphertext_bytes_per_patient": recs[0]["input_ciphertext_bytes_total"],
                    "output_ciphertext_bytes": recs[0]["output_ciphertext_bytes"],
                    "n_input_ciphertexts": len(recs[0]["input_ciphertext_bytes_each"]),
                }
            cells.append(cell)
            lat = cell.get("real_latency", {}).get("wall_clock_s_mean")
            print(f"seed={seed:>3d} trees={n_trees:>2d} depth={depth}: dF1={cell['abs_delta_f1']:.5f} (signed {cell['signed_delta_f1_ckks_minus_plain']:+.5f}) "
                  f"agree={cell['agreement']:.4f} leaves={cell['n_leaves']:>4d} depth_total={cell['code_derived_depth']} "
                  f"latency={'n/a' if lat is None else f'{lat:.3f}s'}  [{time.perf_counter() - t0:.0f}s]", flush=True)

    df = pd.DataFrame(cells)

    # ---- reproducibility gate: seed-42 cells vs the committed single-seed pilot ----
    with open(PILOT_PATH, "r", encoding="utf-8") as f:
        pilot = json.load(f)
    pilot_cells = {(r["n_estimators"], r["max_depth"]): r for r in pilot["tree_count_sweep_depth3"] + pilot["depth_sweep_trees50"]}
    gate = []
    for (n_trees, depth), pr in pilot_cells.items():
        mine = df[(df.seed == 42) & (df.n_estimators == n_trees) & (df.max_depth == depth)].iloc[0]
        gate.append({"n_estimators": n_trees, "max_depth": depth, "pilot_delta_f1": pr["delta_f1"], "this_run_delta_f1": float(mine.abs_delta_f1),
                     "abs_difference": abs(pr["delta_f1"] - float(mine.abs_delta_f1)),
                     "within_tolerance": bool(abs(pr["delta_f1"] - float(mine.abs_delta_f1)) <= GATE_TOL)})
    print(f"\nReproducibility gate vs the committed single-seed pilot: {sum(g['within_tolerance'] for g in gate)}/{len(gate)} seed-42 cells within {GATE_TOL}")

    # ---- aggregation across seeds (difference first per seed, then average) ----
    agg = []
    for (n_trees, depth), g in df.groupby(["n_estimators", "max_depth"], sort=False):
        row = {"n_estimators": int(n_trees), "max_depth": int(depth), "n_seeds": int(len(g)), "n_leaves_mean": float(g.n_leaves.mean()),
               "code_derived_depth": int(g.code_derived_depth.iloc[0]), "fits_locked_chain_and_slots": bool(g.fits_locked_chain_and_slots.all())}
        for col in ("abs_delta_f1", "signed_delta_f1_ckks_minus_plain", "agreement"):
            row[col] = {"mean": float(g[col].mean()), "sd_across_seeds": float(g[col].std(ddof=1))}
        if row["fits_locked_chain_and_slots"]:
            lat = [c["real_latency"]["wall_clock_s_mean"] for c in cells if c["n_estimators"] == n_trees and c["max_depth"] == depth]
            row["real_latency_wall_clock_s"] = {"mean": float(np.mean(lat)), "sd_across_seeds": float(np.std(lat, ddof=1))}
            row["real_input_ciphertext_bytes_per_patient"] = [c["real_latency"]["input_ciphertext_bytes_per_patient"] for c in cells
                                                            if c["n_estimators"] == n_trees and c["max_depth"] == depth][0]
        agg.append(row)

    out = {
        "protocol": "closure guide 4.4; see src/run_scaling_multiseed.py docstring",
        "utility_backend": "SIMULATED circuit-faithful evaluator (same as the degree sweep)",
        "latency_backend": "REAL TenSEAL/SEAL ciphertext execution on the locked parameters",
        "label": LABEL, "k": K, "seeds": list(MODEL_SEEDS), "configurations": [{"n_estimators": a, "max_depth": b} for a, b in CONFIGS],
        "statistic_note": ("abs_delta_f1 is the locked statistic |f1_plain - f1_ckks|; the signed difference is reported alongside because "
                           "the absolute value hides direction"),
        "single_seed_pilot_preserved": "results/m1_scaling_ablation.json (untouched)",
        "reproducibility_gate_vs_pilot": gate,
        "per_cell": cells,
        "aggregated_across_seeds": agg,
        "runtime_seconds": round(time.perf_counter() - t_start, 1),
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nAggregated across {len(MODEL_SEEDS)} seeds (mean +/- SD):")
    for r in agg:
        lat = r.get("real_latency_wall_clock_s")
        lat_txt = "n/a (does not fit)" if lat is None else "%.3f +/- %.3fs" % (lat["mean"], lat["sd_across_seeds"])
        print(f"  trees={r['n_estimators']:>2d} depth={r['max_depth']}: dF1={r['abs_delta_f1']['mean']:.5f} +/- {r['abs_delta_f1']['sd_across_seeds']:.5f}  latency={lat_txt}")
    print(f"written -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
