"""
M1 -- Multi-seed REAL CKKS validation (closure guide 3.5). Executes docs/real_ckks_multiseed_protocol.md.

REAL TenSEAL/SEAL ciphertext execution, compared against the circuit-faithful numpy SIMULATION. It is evidence that the
simulation stays faithful across independently trained models; it is not an estimate of encrypted-model accuracy.

Scope
  * seeds 42, 123, 7, 456, 789; ONE label (hypertension) for all five; k in {3, 5, 7}; n = 25 per seed, the same 25
    patients for every k. k = 9 is excluded by design (does not fit the locked chain, m1-0-implementation-lock.md sec 18).
  * The 25 patients per seed are read from the FROZEN samples in data/m1_real_ckks_samples/ (drawn by
    src/draw_real_ckks_samples.py, hashes committed in results/m1_real_ckks_sample_manifest.json before this ran).
  * Population: TEST split. Used to measure fidelity; nothing is selected from it. No label column is read.
  * The evaluator is the committed ckks_circuit.evaluate_circuit_tenseal, unmodified, so results are comparable with the
    original single-seed arm (results/m1_encrypted_subsample.json, kept untouched).

Reported per (seed, k): agreement (sign) of real vs simulated margin, normalized MAE, max |error|, and per-patient wall-clock
latency. Then the pooled mean +/- SD across the 5 seeds (SD across seeds, not patients). Real and simulated are labelled.

Output: results/m1_encrypted_multiseed.json
"""
from __future__ import annotations

import hashlib
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
from ckks_instrumented import LOCKED_CHAIN, LOCKED_N, LOCKED_SCALE_BITS, make_ckks_context  # noqa: E402
from splits import MODEL_SEEDS  # noqa: E402

COHORT_PATH = os.path.join(_HERE, "..", "data", "m1_cohort.parquet")
FEATURES_PATH = os.path.join(_HERE, "..", "data", "m1_features.parquet")
MODEL_DIR = os.path.join(_HERE, "..", "data", "models")
SAMPLE_DIR = os.path.join(_HERE, "..", "data", "m1_real_ckks_samples")
MANIFEST_PATH = os.path.join(_HERE, "..", "results", "m1_real_ckks_sample_manifest.json")
OUT_PATH = os.path.join(_HERE, "..", "results", "m1_encrypted_multiseed.json")
DEPTH_PROBE = os.path.join(_HERE, "..", "results", "m1_depth_probe.json")

LABEL = "label_hypertension_uncomplicated"  # one label for every seed (guide 3.5)
K_EXECUTED = (3, 5, 7)
K_EXCLUDED = 9
N_PER_SEED = 25


def sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main() -> int:
    t_run_start = time.time()
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["label"] == LABEL, "R3/design: the frozen samples were drawn for a different label"

    # R1 -- every sample is exactly the frozen one, and predates this run
    for seed in MODEL_SEEDS:
        path = os.path.join(SAMPLE_DIR, f"seed{seed}.csv")
        assert sha256(path) == manifest["seeds"][str(seed)]["sha256"], f"R1 FAILED: seed {seed} sample differs from the frozen manifest"
        assert os.path.getmtime(path) < t_run_start, f"R1 FAILED: seed {seed} sample was written after this run started"
    print("R1 all five samples match the committed SHA-256 manifest and predate this run: PASS")

    cohort = pd.read_parquet(COHORT_PATH)
    features = pd.read_parquet(FEATURES_PATH)
    feature_cols = [c for c in features.columns if c != "stay_id"]
    # R3 -- id columns only: no label is ever loaded into the frame used for evaluation
    frame = features.merge(cohort[["stay_id", "subject_id"]], on="stay_id", how="inner").set_index("stay_id")
    assert not any(c.startswith("label_") for c in frame.columns), "R3 FAILED: a label column is present"
    print("R3 no label column is present in the evaluation frame: PASS")

    art = cc.load_circuit_frozen_artifacts()
    coeff_path = os.path.abspath(cc.POLY_COEFFS_PATH)
    with open(coeff_path, "r", encoding="utf-8") as f:
        coeff_meta = json.load(f)

    ctx = make_ckks_context()  # R4: locked N, chain and scale, one place
    print(f"R4 CKKS context from the locked parameters: N={LOCKED_N} chain={LOCKED_CHAIN} scale=2^{LOCKED_SCALE_BITS}; "
          f"k={K_EXCLUDED} not executed: PASS")

    cells = []
    for seed in MODEL_SEEDS:
        sample = pd.read_csv(os.path.join(SAMPLE_DIR, f"seed{seed}.csv"))
        assert len(sample) == N_PER_SEED
        X = frame.loc[sample["stay_id"].to_numpy(), feature_cols].reset_index(drop=True)

        clf = xgb.XGBClassifier()
        clf.load_model(os.path.join(MODEL_DIR, f"xgb_float_seed{seed}_{LABEL}.json"))
        booster = clf.get_booster()
        booster.feature_names = feature_cols
        plain = booster.predict(xgb.DMatrix(X, feature_names=feature_cols), output_margin=True)
        assert np.allclose(plain, sample["plaintext_margin"].to_numpy(), atol=1e-6), f"seed {seed}: frozen sample margins do not reproduce"
        structure = cc.extract_ensemble_leaf_circuit(booster, art["continuous_features"], tree_depth=6)

        # one untimed warm-up evaluation so the first cell does not carry cold-start cost
        warm = cc.assemble_coefficient_vectors(structure, art["poly_coeffs"]["targets"]["soft_step"], K_EXECUTED[0])
        cc.evaluate_circuit_tenseal(ctx, structure, warm, X.iloc[0], art["norm_bounds"], art["a_half_width"])

        for k in K_EXECUTED:
            coeffs = cc.assemble_coefficient_vectors(structure, art["poly_coeffs"]["targets"]["soft_step"], k)
            sim, enc, wall, phase_sum = [], [], [], []
            for i in range(len(X)):
                sim.append(cc.evaluate_circuit_numpy(structure, coeffs, X.iloc[i], art["norm_bounds"], art["a_half_width"]))
                t0 = time.perf_counter()
                margin, timing = cc.evaluate_circuit_tenseal(ctx, structure, coeffs, X.iloc[i], art["norm_bounds"], art["a_half_width"])
                wall.append(time.perf_counter() - t0)
                enc.append(margin)
                phase_sum.append(sum(timing.values()))
            sim, enc = np.array(sim), np.array(enc)
            err = np.abs(enc - sim)
            span = float(sim.max() - sim.min()) or 1.0
            cells.append({
                "seed": seed, "k": k, "n": len(X), "label": LABEL,
                "agreement_real_vs_simulated": float(np.mean(np.sign(enc) == np.sign(sim))),
                "agreement_real_vs_plaintext_xgboost": float(np.mean(np.sign(enc) == np.sign(plain))),
                "normalized_mae_real_vs_simulated": float(err.mean() / span),
                "max_abs_error_real_vs_simulated": float(err.max()),
                "latency_wall_clock_s_mean": float(np.mean(wall)),
                "latency_wall_clock_s_std_within_cell": float(np.std(wall, ddof=1)),
                "latency_phase_sum_s_mean": float(np.mean(phase_sum)),
            })
            c = cells[-1]
            print(f"seed={seed:>3d} k={k}: agreement(real,sim)={c['agreement_real_vs_simulated']:.4f} "
                  f"nMAE={c['normalized_mae_real_vs_simulated']:.2e} latency={c['latency_wall_clock_s_mean']:.3f}s", flush=True)

    df = pd.DataFrame(cells)
    pooled = {}
    for k in K_EXECUTED:
        g = df[df.k == k]
        pooled[f"k={k}"] = {"n_seeds": int(len(g)), "n_patient_evaluations": int(g["n"].sum())}
        for col in ("agreement_real_vs_simulated", "agreement_real_vs_plaintext_xgboost", "normalized_mae_real_vs_simulated",
                    "max_abs_error_real_vs_simulated", "latency_wall_clock_s_mean"):
            pooled[f"k={k}"][col] = {"mean": float(g[col].mean()), "sd_across_seeds": float(g[col].std(ddof=1)),
                                     "min": float(g[col].min()), "max": float(g[col].max())}

    with open(DEPTH_PROBE, "r", encoding="utf-8") as f:
        k9 = json.load(f)["per_degree"]["k=9"]
    out = {
        "protocol": "docs/real_ckks_multiseed_protocol.md",
        "backend": "REAL TenSEAL/SEAL ciphertext execution, compared against the SIMULATED circuit-faithful numpy evaluator",
        "label_same_for_all_seeds": LABEL, "seeds": list(MODEL_SEEDS), "k_executed": list(K_EXECUTED),
        "n_per_seed": N_PER_SEED, "population": "TEST split, frozen stratified samples (12 pred-positive + 13 pred-negative)",
        "k_excluded": K_EXCLUDED,
        "k_excluded_reason": (f"code-derived depth {k9['code_derived']['total_multiplicative_depth']} > "
                              f"{k9['levels_available_on_locked_chain']} levels on the locked chain "
                              "(results/m1_depth_probe.json; m1-0-implementation-lock.md sec 18)"),
        "locked_parameters": {"N": LOCKED_N, "coeff_modulus_bit_sizes": LOCKED_CHAIN, "scale_bits": LOCKED_SCALE_BITS},
        "coefficients": {"file": os.path.basename(coeff_path), "sha256": sha256(coeff_path), "version": coeff_meta.get("version"),
                         "gamma": coeff_meta.get("gamma"), "gamma_anchor_percentile": coeff_meta.get("gamma_anchor_percentile")},
        "sample_manifest_sha256": sha256(MANIFEST_PATH),
        "per_cell": cells,
        "pooled_across_seeds": pooled,
        "limitation": ("n=25 per seed: coarse agreement resolution (1/25); this supports 'the simulation is faithful across "
                       "independently trained models', not an estimate of encrypted accuracy."),
        "runtime_seconds": round(time.time() - t_run_start, 1),
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\nPooled across the 5 seeds (mean +/- SD across seeds):")
    for k in K_EXECUTED:
        p = pooled[f"k={k}"]
        print(f"  k={k}: agreement(real,sim)={p['agreement_real_vs_simulated']['mean']:.4f} +/- {p['agreement_real_vs_simulated']['sd_across_seeds']:.4f}   "
              f"nMAE={p['normalized_mae_real_vs_simulated']['mean']:.2e} +/- {p['normalized_mae_real_vs_simulated']['sd_across_seeds']:.2e}   "
              f"latency={p['latency_wall_clock_s_mean']['mean']:.3f} +/- {p['latency_wall_clock_s_mean']['sd_across_seeds']:.3f}s")
    print(f"written -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
