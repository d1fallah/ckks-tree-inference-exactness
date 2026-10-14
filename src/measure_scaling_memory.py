"""
M1 -- Per-configuration memory in the scaling grid (open item from the closure log, Entry 10: "memory was not measured per configuration").

Same grid as run_scaling_multiseed.py (trees {5,10,20,30,50} at depth 3; depth {3,4,5,6} at 50 trees; k = 5; hypertension), one model seed (42).
Memory is a property of ciphertext count and size, not of the training seed, so one seed is measured; this is stated as a limit, not hidden.

Protocol (identical to measure_ckks_latency.py, per configuration):
  * the parent trains each configuration (same train_config as the scaling run) and saves the booster; the parent's memory is never reported.
  * one FRESH child process per configuration: build the locked context and keys, load the cohort table and the saved model, reset the kernel's
    peak-RSS counter (VmHWM), run 2 discarded warm-up evaluations and 10 timed evaluations on the frozen seed-42 sample, read the peak.
  * reported: peak RSS during the evaluation phase, its increment over the RSS immediately before evaluation, and the one-off context+keys increment.
    Whole-process RSS from /proc (SEAL native allocations included); tracemalloc does not see native memory.
  * gates: the trained structure's leaf count equals the committed scaling run's seed-42 cell; the first patient's real margin agrees with the
    simulated evaluator within the encryption-noise band.
Latency recorded here is a cross-check on the scaling run's latency, not a new latency claim. Output: results/m1_scaling_memory.json
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

COHORT_PATH = os.path.join(_HERE, "..", "data", "m1_cohort.parquet")
FEATURES_PATH = os.path.join(_HERE, "..", "data", "m1_features.parquet")
SAMPLE_PATH = os.path.join(_HERE, "..", "data", "m1_real_ckks_samples", "seed42.csv")
SCALING_PATH = os.path.join(_HERE, "..", "results", "m1_scaling_multiseed.json")
OUT_PATH = os.path.join(_HERE, "..", "results", "m1_scaling_memory.json")
LABEL, SEED, K_DEGREE = "label_hypertension_uncomplicated", 42, 5
CONFIGS = [(5, 3), (10, 3), (20, 3), (30, 3), (50, 3), (50, 4), (50, 5), (50, 6)]  # (n_estimators, max_depth)
N_WARM, N_TIMED = 2, 10
FIDELITY_TOL = 5e-4
MARK = "@@CHILD_RESULT@@"


def child(model_path: str, depth: int) -> dict:
    import xgboost as xgb
    import ckks_circuit as cc
    import ckks_instrumented as ci

    rss0 = ci.proc_memory_kb()
    ctx = ci.make_ckks_context()
    seal = ci.make_seal_handles(ctx)
    rss_after_keygen = ci.proc_memory_kb()

    cohort = pd.read_parquet(COHORT_PATH)
    features = pd.read_parquet(FEATURES_PATH)
    feature_cols = [c for c in features.columns if c != "stay_id"]
    frame = features.merge(cohort[["stay_id", "subject_id"]], on="stay_id", how="inner").set_index("stay_id")
    sample = pd.read_csv(SAMPLE_PATH)
    X = frame.loc[sample["stay_id"].to_numpy(), feature_cols].reset_index(drop=True)

    art = cc.load_circuit_frozen_artifacts()
    booster = xgb.Booster()
    booster.load_model(model_path)
    booster.feature_names = feature_cols
    structure = cc.extract_ensemble_leaf_circuit(booster, art["continuous_features"], tree_depth=depth)
    coeffs = cc.assemble_coefficient_vectors(structure, art["poly_coeffs"]["targets"]["soft_step"], K_DEGREE)
    rss_ready = ci.proc_memory_kb()

    reset_ok = ci.reset_peak_rss()
    rss_before = ci.proc_memory_kb()
    for i in range(N_WARM):  # discarded
        ci.evaluate_instrumented(ctx, seal, structure, coeffs, X.iloc[N_TIMED + i], art["norm_bounds"], art["a_half_width"], decompose=False)
    recs, gate = [], None
    for i in range(N_TIMED):
        margin, rec = ci.evaluate_instrumented(ctx, seal, structure, coeffs, X.iloc[i], art["norm_bounds"], art["a_half_width"], decompose=False)
        recs.append(rec)
        if i == 0:
            sim = cc.evaluate_circuit_numpy(structure, coeffs, X.iloc[0], art["norm_bounds"], art["a_half_width"])
            gate = abs(margin - sim)
            assert gate <= FIDELITY_TOL, f"real margin disagrees with the simulation by {gate}"
    rss_peak = ci.proc_memory_kb()
    return {
        "n_leaves": int(structure.n_leaves),
        "n_input_ciphertexts": len(recs[0]["input_ciphertext_bytes_each"]),
        "input_ciphertext_bytes_per_patient": recs[0]["input_ciphertext_bytes_total"],
        "memory_kb": {
            "peak_rss_reset_supported": reset_ok,
            "rss_at_process_start": rss0["rss_kb"],
            "context_and_keys_increment": rss_after_keygen["rss_kb"] - rss0["rss_kb"],
            "rss_after_model_load": rss_ready["rss_kb"],
            "rss_immediately_before_evaluation": rss_before["rss_kb"],
            "peak_rss_during_evaluation_phase": rss_peak["peak_rss_kb"],
            "evaluation_phase_increment_over_pre_evaluation_rss": rss_peak["peak_rss_kb"] - rss_before["rss_kb"],
        },
        "latency_wall_clock_s_mean_crosscheck": float(np.mean([r["wall_clock_total_s"] for r in recs])),
        "fidelity_abs_diff_vs_simulation_first_patient": gate,
    }


def main() -> int:
    if len(sys.argv) == 4 and sys.argv[1] == "--child":
        print(MARK + json.dumps(child(sys.argv[2], int(sys.argv[3]))))
        return 0

    import xgboost as xgb
    from run_scaling_multiseed import train_config
    from splits import patient_level_split
    import ckks_circuit as cc

    with open(SCALING_PATH, "r", encoding="utf-8") as f:
        committed_leaves = {(c["n_estimators"], c["max_depth"]): c["n_leaves"] for c in json.load(f)["per_cell"] if c["seed"] == SEED}

    cohort = pd.read_parquet(COHORT_PATH)
    features = pd.read_parquet(FEATURES_PATH)
    label_cols = [c for c in cohort.columns if c.startswith("label_")]
    merged = features.merge(cohort[["stay_id", "subject_id"] + label_cols], on="stay_id", how="inner")
    feature_cols = [c for c in features.columns if c != "stay_id"]
    train_ids, val_ids, _ = patient_level_split(merged)
    tr, va = (merged["subject_id"].isin(s) for s in (train_ids, val_ids))
    X_train, y_train = merged.loc[tr, feature_cols], merged.loc[tr, LABEL]
    X_val, y_val = merged.loc[va, feature_cols], merged.loc[va, LABEL]
    art = cc.load_circuit_frozen_artifacts()

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for n_trees, depth in CONFIGS:
            t0 = time.perf_counter()
            booster, _thr = train_config(X_train, y_train, X_val, y_val, feature_cols, n_trees, depth, SEED)
            structure = cc.extract_ensemble_leaf_circuit(booster, art["continuous_features"], tree_depth=depth)
            assert int(structure.n_leaves) == committed_leaves[(n_trees, depth)], (
                f"({n_trees},{depth}): retrained model has {structure.n_leaves} leaves, committed scaling run had {committed_leaves[(n_trees, depth)]}")
            path = os.path.join(tmp, f"m_{n_trees}_{depth}.json")
            booster.save_model(path)
            print(f"trees={n_trees} depth={depth}: trained ({time.perf_counter() - t0:.0f}s), fresh process...", flush=True)
            proc = subprocess.run([sys.executable, "-u", os.path.abspath(__file__), "--child", path, str(depth)],
                                  capture_output=True, text=True, cwd=_HERE)
            if proc.returncode != 0:
                print(proc.stdout[-2000:], proc.stderr[-2000:])
                raise SystemExit(f"child for ({n_trees},{depth}) failed")
            res = json.loads(next(ln for ln in proc.stdout.splitlines() if ln.startswith(MARK))[len(MARK):])
            res.update({"n_estimators": n_trees, "max_depth": depth})
            rows.append(res)
            m = res["memory_kb"]
            print(f"  eval-phase +{m['evaluation_phase_increment_over_pre_evaluation_rss'] / 1024:.0f} MiB  peak {m['peak_rss_during_evaluation_phase'] / 1024:.0f} MiB  "
                  f"keys +{m['context_and_keys_increment'] / 1024:.0f} MiB  input ct {res['input_ciphertext_bytes_per_patient'] / 1e6:.2f} MB/patient  "
                  f"latency {res['latency_wall_clock_s_mean_crosscheck']:.3f}s", flush=True)

    out = {
        "protocol": "see src/measure_scaling_memory.py docstring (closure log Entry 10 open item)",
        "backend": "REAL TenSEAL/SEAL ciphertext execution",
        "k": K_DEGREE, "label": LABEL, "model_seed": SEED, "n_warmup_discarded": N_WARM, "n_timed": N_TIMED,
        "limit": "one model seed; memory depends on ciphertext count and size, not on the training seed, but this was not tested across seeds; one machine (WSL2), host load not controlled",
        "process_isolation": "one fresh process per configuration; the parent trains and never reports its own memory",
        "per_configuration": rows,
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"written -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
