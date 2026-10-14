"""
M1 -- Full CKKS latency, memory and ciphertext-size breakdown per k (closure guide 4.1).

Real TenSEAL/SEAL execution on fixed hardware with a stated protocol. Nothing about the circuit or parameters changes; the
evaluator is ckks_instrumented.evaluate_instrumented (same arithmetic as ckks_circuit.evaluate_circuit_tenseal).

Protocol (fixed here, not tuned)
  * model: seed 42, label hypertension (the representative model used throughout); k in {3, 5, 7}. k = 9 does not fit the locked chain.
  * patients: the 25 frozen seed-42 test-split patients from data/m1_real_ckks_samples/seed42.csv, same for every k.
      warm-up : patients 20-24, 5 evaluations, discarded
      timed   : patients 0-19, ONE evaluation each = 20 timed repetitions across distinct patients
  * one FRESH PROCESS per k, so memory belongs to that k alone.
  * what is timed: see ckks_instrumented.py. wall_clock_total is a perf_counter around the whole online path, not a sum of buckets.
  * reported: mean and SD over the 20 timed repetitions, per bucket. Context / key generation is timed separately (one-off, not per patient).
  * fidelity gate: the instrumented margin must agree with the committed evaluator's margin and with the simulation to within
    CKKS encryption noise (encryption is randomized, so two runs cannot be bit-identical).

Memory: resident-set-size of the whole Python process (SEAL native allocations included), read from /proc. The kernel's peak counter (VmHWM) is
reset immediately before the evaluation phase, so the reported evaluation-phase increment excludes loading the cohort table and the model; the
one-off context/key cost is reported separately. tracemalloc is NOT used: it does not see native memory.
Output: results/m1_latency_breakdown.json
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

# Revision E1: M1_LATENCY_EVALUATOR=folded measures the coefficient-folding evaluator at k in {3,5,7,9} into a separate file; the default
# (original evaluator, k in {3,5,7}, results/m1_latency_breakdown.json) is unchanged.
FOLDED = os.environ.get("M1_LATENCY_EVALUATOR") == "folded"
OUT_PATH = os.path.join(_HERE, "..", "results", "m1_latency_breakdown_folded.json" if FOLDED else "m1_latency_breakdown.json")
SAMPLE_PATH = os.path.join(_HERE, "..", "data", "m1_real_ckks_samples", "seed42.csv")
COHORT_PATH = os.path.join(_HERE, "..", "data", "m1_cohort.parquet")
FEATURES_PATH = os.path.join(_HERE, "..", "data", "m1_features.parquet")
MODEL_DIR = os.path.join(_HERE, "..", "data", "models")
LABEL, SEED = "label_hypertension_uncomplicated", 42
K_VALUES = (3, 5, 7, 9) if FOLDED else (3, 5, 7)
N_WARMUP, N_TIMED = 5, 20
FIDELITY_TOL = 5e-4  # |margin_instrumented - margin_committed|, both within the same encryption-noise band
MARK = "@@CHILD_RESULT@@"


def hardware() -> dict:
    cpu = ""
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            mem_kb = int(f.readline().split()[1])
    except OSError:
        mem_kb = None
    import tenseal as ts
    return {"cpu_model": cpu, "logical_cpus": os.cpu_count(), "total_memory_kb": mem_kb, "kernel": platform.release(),
            "platform": platform.platform(), "python": platform.python_version(), "tenseal": getattr(ts, "__version__", "unknown"),
            "env_OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"), "note": "WSL2 guest; host load not controlled"}


def child(degree_k: int) -> dict:
    import xgboost as xgb
    import ckks_circuit as cc
    import ckks_instrumented as ci

    rss0 = ci.proc_memory_kb()
    t0 = time.perf_counter()
    ctx = ci.make_ckks_context()
    t_keygen = time.perf_counter() - t0
    seal = ci.make_seal_handles(ctx)
    rss_after_keygen = ci.proc_memory_kb()
    key_sizes = ci.context_key_sizes(ctx)

    cohort = pd.read_parquet(COHORT_PATH)
    features = pd.read_parquet(FEATURES_PATH)
    feature_cols = [c for c in features.columns if c != "stay_id"]
    frame = features.merge(cohort[["stay_id", "subject_id"]], on="stay_id", how="inner").set_index("stay_id")
    sample = pd.read_csv(SAMPLE_PATH)
    X = frame.loc[sample["stay_id"].to_numpy(), feature_cols].reset_index(drop=True)

    art = cc.load_circuit_frozen_artifacts()
    clf = xgb.XGBClassifier()
    clf.load_model(os.path.join(MODEL_DIR, f"xgb_float_seed{SEED}_{LABEL}.json"))
    booster = clf.get_booster()
    booster.feature_names = feature_cols
    structure = cc.extract_ensemble_leaf_circuit(booster, art["continuous_features"], tree_depth=6)
    coeffs = cc.assemble_coefficient_vectors(structure, art["poly_coeffs"]["targets"]["soft_step"], degree_k)
    rss_ready = ci.proc_memory_kb()

    poly_evaluator = cc.eval_poly_tenseal_folded if FOLDED else None
    args = (structure, coeffs)
    reset_ok = ci.reset_peak_rss()  # from here on, VmHWM reflects only the evaluation phase
    rss_before_eval = ci.proc_memory_kb()
    warm_idx = list(range(len(X) - N_WARMUP, len(X)))
    timed_idx = list(range(N_TIMED))
    for i in warm_idx:  # discarded
        ci.evaluate_instrumented(ctx, seal, *args, X.iloc[i], art["norm_bounds"], art["a_half_width"], poly_evaluator=poly_evaluator)

    recs, gate = [], []
    for i in timed_idx:
        margin, rec = ci.evaluate_instrumented(ctx, seal, *args, X.iloc[i], art["norm_bounds"], art["a_half_width"], poly_evaluator=poly_evaluator)
        recs.append(rec)
        if i < 3:  # fidelity gate on the first three timed patients (outside the timing of the rest)
            m_committed, _ = cc.evaluate_circuit_tenseal(ctx, structure, coeffs, X.iloc[i], art["norm_bounds"], art["a_half_width"],
                                                          poly_evaluator=poly_evaluator)
            m_sim = cc.evaluate_circuit_numpy(structure, coeffs, X.iloc[i], art["norm_bounds"], art["a_half_width"])
            gate.append({"patient": i, "abs_diff_vs_committed_evaluator": abs(margin - m_committed),
                         "abs_diff_vs_simulation": abs(margin - m_sim)})
            assert abs(margin - m_committed) <= FIDELITY_TOL, f"k={degree_k}: instrumented margin disagrees with the committed evaluator"
            assert abs(margin - m_sim) <= FIDELITY_TOL, f"k={degree_k}: instrumented margin disagrees with the simulation"
    rss_peak = ci.proc_memory_kb()  # peak since the reset if reset_ok, else whole-process peak

    numeric = [k_ for k_ in recs[0] if isinstance(recs[0][k_], float)]
    summary = {k_: {"mean_s": float(np.mean([r[k_] for r in recs])), "sd_s": float(np.std([r[k_] for r in recs], ddof=1))}
               for k_ in numeric}
    return {
        "k": degree_k, "n_warmup_discarded": N_WARMUP, "n_timed": N_TIMED,
        "context_and_keygen_s_one_off": t_keygen, "key_sizes_bytes": key_sizes,
        "ciphertext_bytes": {"input_each_level": recs[0]["input_ciphertext_bytes_each"],
                             "input_total_per_patient": recs[0]["input_ciphertext_bytes_total"],
                             "output_per_patient": recs[0]["output_ciphertext_bytes"]},
        "timing_per_patient": summary,
        "memory_kb": {"rss_at_process_start": rss0["rss_kb"], "rss_after_context_and_keygen": rss_after_keygen["rss_kb"],
                      "rss_after_model_load": rss_ready["rss_kb"], "rss_immediately_before_evaluation": rss_before_eval["rss_kb"],
                      "peak_rss_reset_supported": reset_ok,
                      "peak_rss_during_evaluation_phase" if reset_ok else "peak_rss_over_whole_process": rss_peak["peak_rss_kb"],
                      "evaluation_phase_increment_over_pre_evaluation_rss": rss_peak["peak_rss_kb"] - rss_before_eval["rss_kb"],
                      "context_and_keys_increment": rss_after_keygen["rss_kb"] - rss0["rss_kb"],
                      "note": ("evaluation_phase_increment = peak RSS reached during the warm-up and timed evaluations minus RSS immediately before them; "
                               "context_and_keys_increment is the one-off resident cost of the context, public/relinearization/Galois keys. "
                               "Loading the cohort table and the model before evaluation is NOT counted in either.")},
        "fidelity_gate": gate,
    }


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--child":
        print(MARK + json.dumps(child(int(sys.argv[2]))))
        return 0

    results = {}
    for k in K_VALUES:
        print(f"k={k}: fresh process...", flush=True)
        proc = subprocess.run([sys.executable, "-u", os.path.abspath(__file__), "--child", str(k)],
                              capture_output=True, text=True, cwd=_HERE)
        if proc.returncode != 0:
            print(proc.stdout[-2000:], proc.stderr[-2000:])
            raise SystemExit(f"child for k={k} failed")
        line = next(ln for ln in proc.stdout.splitlines() if ln.startswith(MARK))
        results[f"k={k}"] = json.loads(line[len(MARK):])
        t = results[f"k={k}"]["timing_per_patient"]
        m = results[f"k={k}"]["memory_kb"]
        print(f"  wall_clock {t['wall_clock_total_s']['mean_s']:.3f} +/- {t['wall_clock_total_s']['sd_s']:.3f}s  "
              f"eval-phase +{m['evaluation_phase_increment_over_pre_evaluation_rss'] / 1024:.0f} MiB (keys +{m['context_and_keys_increment'] / 1024:.0f} MiB)  "
              f"input ct {results[f'k={k}']['ciphertext_bytes']['input_total_per_patient'] / 1e6:.2f} MB/patient")

    from ckks_instrumented import LOCKED_CHAIN, LOCKED_N, LOCKED_SCALE_BITS
    out = {
        "protocol": "closure guide 4.1; see src/measure_ckks_latency.py docstring",
        "backend": "REAL TenSEAL/SEAL ciphertext execution",
        "locked_parameters": {"N": LOCKED_N, "coeff_modulus_bit_sizes": LOCKED_CHAIN, "scale_bits": LOCKED_SCALE_BITS},
        "hardware": hardware(),
        "protocol_detail": {"model": {"seed": SEED, "label": LABEL}, "patients": "frozen seed-42 test-split sample (data/m1_real_ckks_samples/seed42.csv)",
                            "warmup": f"{N_WARMUP} evaluations on patients {25 - N_WARMUP}-24, discarded",
                            "timed": f"{N_TIMED} evaluations, one per distinct patient 0-{N_TIMED - 1}",
                            "process_isolation": "one fresh process per k for per-k peak memory",
                            "decomposition_note": ("encode, encrypt, decrypt and decode are measured by pushing the same data through SEAL "
                                                   "directly after the online path; they are not part of wall_clock_total_s"),
                            "k9": ("measured with the coefficient-folding evaluator" if FOLDED else "not measured: does not fit the locked chain (results/m1_depth_probe.json)"),
                            "polynomial_evaluator": "coefficient-folding (ckks_circuit.eval_poly_tenseal_folded)" if FOLDED else "original (ckks_circuit.eval_poly_tenseal)"},
        "by_k": results,
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"written -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
