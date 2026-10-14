"""
M1 revision E2 -- the full test split on REAL CKKS ciphertexts at the selected degree.
Executes docs/revision_prespec_m1.md (E2).

Configuration (fixed in the pre-specification): seed 42, k = 3, the ORIGINAL polynomial evaluator, all five labels, every test stay,
the frozen validation-tuned thresholds. One seed only; the manuscript says so.

For every (label, stay): real encrypt -> evaluate -> decrypt with TenSEAL/SEAL (ckks_circuit.evaluate_circuit_tenseal, unmodified), next to
the plaintext XGBoost margin and the circuit-faithful numpy simulation of the same stay. Work is split into units of UNIT_SIZE stays per
label. Each unit is written atomically to data/m1_real_full_split/ when finished, so an interruption loses at most one unit, and the run
resumes by skipping finished units. A stay whose real evaluation raises is recorded as NaN with the exception and is NOT retried; the
analysis reports the count.

usage (WSL, from the module root):  python src/eval_ckks_real_full_split.py --worker I --workers W
Aggregation and statistics: src/analyze_real_full_split.py  (reads only the unit files; never re-runs ciphertext evaluation).
"""
from __future__ import annotations

import argparse
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
from ckks_instrumented import make_ckks_context  # noqa: E402
from splits import patient_level_split  # noqa: E402

COHORT_PATH = os.path.join(_HERE, "..", "data", "m1_cohort.parquet")
FEATURES_PATH = os.path.join(_HERE, "..", "data", "m1_features.parquet")
MODEL_DIR = os.path.join(_HERE, "..", "data", "models")
BASELINE_PATH = os.path.join(_HERE, "..", "results", "m1_plaintext_baseline.json")
UNIT_DIR = os.path.join(_HERE, "..", "data", "m1_real_full_split")

SEED = 42
K = 3
UNIT_SIZE = 1000


def load_test():
    cohort = pd.read_parquet(COHORT_PATH)
    features = pd.read_parquet(FEATURES_PATH)
    label_cols = [c for c in cohort.columns if c.startswith("label_")]
    merged = features.merge(cohort[["stay_id", "subject_id"] + label_cols], on="stay_id", how="inner")
    feature_cols = [c for c in features.columns if c != "stay_id"]
    _, _, test_ids = patient_level_split(merged)
    test_df = merged[merged["subject_id"].isin(test_ids)].reset_index(drop=True)
    return test_df, feature_cols, label_cols


def unit_list(n_test: int, n_labels: int):
    n_chunks = -(-n_test // UNIT_SIZE)
    return [(li, ch) for ch in range(n_chunks) for li in range(n_labels)]


def unit_path(label_idx: int, chunk: int) -> str:
    return os.path.join(UNIT_DIR, f"unit_l{label_idx}_c{chunk:03d}.npz")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, required=True)
    ap.add_argument("--workers", type=int, required=True)
    args = ap.parse_args()
    os.makedirs(UNIT_DIR, exist_ok=True)

    test_df, feature_cols, label_cols = load_test()
    n_test = len(test_df)
    X_df = test_df[feature_cols]
    X_mat = X_df.to_numpy(dtype=np.float64)
    units = unit_list(n_test, len(label_cols))
    mine = [u for i, u in enumerate(units) if i % args.workers == args.worker]
    print(f"worker {args.worker}/{args.workers}: test stays={n_test}, labels={len(label_cols)}, units total={len(units)}, mine={len(mine)}", flush=True)

    art = cc.load_circuit_frozen_artifacts()
    ctx = make_ckks_context()
    cache: dict[int, dict] = {}

    def prepare(li: int) -> dict:
        if li not in cache:
            label = label_cols[li]
            clf = xgb.XGBClassifier()
            clf.load_model(os.path.join(MODEL_DIR, f"xgb_float_seed{SEED}_{label}.json"))
            booster = clf.get_booster()
            booster.feature_names = feature_cols
            structure = cc.extract_ensemble_leaf_circuit(booster, art["continuous_features"], tree_depth=6)
            coeffs = cc.assemble_coefficient_vectors(structure, art["poly_coeffs"]["targets"]["soft_step"], K)
            batch_levels = cc.prepare_batch_levels(structure, feature_cols, art["norm_bounds"], art["a_half_width"])
            cache[li] = {"structure": structure, "coeffs": coeffs, "booster": booster, "batch_levels": batch_levels}
        return cache[li]

    t_worker = time.time()
    done_here = 0
    for (li, ch) in mine:
        path = unit_path(li, ch)
        if os.path.exists(path):
            continue
        lo, hi = ch * UNIT_SIZE, min((ch + 1) * UNIT_SIZE, n_test)
        p = prepare(li)
        # plaintext and simulated margins per unit: simulating the whole split at once needs several GB per worker
        plain_u = p["booster"].predict(xgb.DMatrix(X_df.iloc[lo:hi], feature_names=feature_cols), output_margin=True)
        sim_u = cc.evaluate_circuit_batch(p["batch_levels"], p["coeffs"], p["structure"].leaf_value, X_mat[lo:hi], art["a_half_width"])
        real = np.full(hi - lo, np.nan)
        failures = []
        t0 = time.time()
        for j, i in enumerate(range(lo, hi)):
            try:
                margin, _ = cc.evaluate_circuit_tenseal(ctx, p["structure"], p["coeffs"], X_df.iloc[i], art["norm_bounds"], art["a_half_width"])
                real[j] = margin
            except Exception as exc:  # recorded, never retried
                failures.append({"row": int(i), "exception_type": type(exc).__name__, "exception_message": str(exc)})
        tmp = path + ".tmp.npz"
        np.savez(tmp, stay_id=test_df["stay_id"].to_numpy()[lo:hi], real=real, sim=sim_u, plain=plain_u,
                 y=test_df[label_cols[li]].to_numpy(dtype=int)[lo:hi], seconds=np.array([time.time() - t0]),
                 failures=np.array([json.dumps(failures)]))
        os.replace(tmp, path)
        done_here += 1
        print(f"worker {args.worker}: unit label={li} chunk={ch} rows={hi - lo} failed={len(failures)} "
              f"{time.time() - t0:.0f}s  (done here {done_here}/{len(mine)}, elapsed {(time.time() - t_worker) / 3600:.2f} h)", flush=True)
    print(f"worker {args.worker}: finished", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
