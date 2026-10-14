"""
M1 revision E1 -- multi-seed REAL CKKS validation with the coefficient-folding evaluator, k in {3, 5, 7, 9}.
Executes docs/revision_prespec_m1.md (E1), which re-runs the design of docs/real_ckks_multiseed_protocol.md unchanged
except for the polynomial evaluator (ckks_circuit.eval_poly_tenseal_folded) and the added degree k = 9.

Same 5 seeds, same hypertension label, same frozen 25 stratified test patients per seed (hash-checked against
results/m1_real_ckks_sample_manifest.json), same fixed CKKS parameters. No label column is read. The original-evaluator
arm (results/m1_encrypted_multiseed.json) is not touched.

Criterion for "k = 9 executable", fixed in the pre-specification before this ran:
  (a) all 125 k = 9 evaluations complete;
  (b) sign agreement between real and simulated margins is 125/125;
  (c) normalized MAE is within the range observed for k = 3/5/7 under the original evaluator.
The same three checks are reported for k = 3/5/7 under the folding evaluator. Per-cell failures are recorded, not retried.

Output: results/m1_encrypted_multiseed_folded.json
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
OUT_PATH = os.path.join(_HERE, "..", "results", "m1_encrypted_multiseed_folded.json")
ORIGINAL_PATH = os.path.join(_HERE, "..", "results", "m1_encrypted_multiseed.json")

LABEL = "label_hypertension_uncomplicated"  # one label for every seed (guide 3.5)
K_EXECUTED = (3, 5, 7, 9)
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
          f"polynomial evaluator: coefficient-folding: PASS")

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
        cc.evaluate_circuit_tenseal(ctx, structure, warm, X.iloc[0], art["norm_bounds"], art["a_half_width"],
                                    poly_evaluator=cc.eval_poly_tenseal_folded)

        for k in K_EXECUTED:
            coeffs = cc.assemble_coefficient_vectors(structure, art["poly_coeffs"]["targets"]["soft_step"], k)
            sim, enc, wall, phase_sum, plain_ok, failures = [], [], [], [], [], []
            for i in range(len(X)):
                s_i = cc.evaluate_circuit_numpy(structure, coeffs, X.iloc[i], art["norm_bounds"], art["a_half_width"])
                t0 = time.perf_counter()
                try:
                    margin, timing = cc.evaluate_circuit_tenseal(ctx, structure, coeffs, X.iloc[i], art["norm_bounds"],
                                                                 art["a_half_width"], poly_evaluator=cc.eval_poly_tenseal_folded)
                except Exception as exc:  # recorded, never retried with other parameters (pre-specification)
                    failures.append({"patient_index": i, "exception_type": type(exc).__name__, "exception_message": str(exc)})
                    continue
                wall.append(time.perf_counter() - t0)
                sim.append(s_i)
                enc.append(margin)
                plain_ok.append(plain[i])
                phase_sum.append(sum(timing.values()))
            sim, enc, plain_ok = np.array(sim), np.array(enc), np.array(plain_ok)
            n_done = len(enc)
            if n_done:
                err = np.abs(enc - sim)
                span = float(sim.max() - sim.min()) or 1.0
            cells.append({
                "seed": seed, "k": k, "n": len(X), "n_completed": n_done, "n_failed": len(failures), "failures": failures,
                "label": LABEL, "polynomial_evaluator": "coefficient-folding",
                "agreement_real_vs_simulated": float(np.mean(np.sign(enc) == np.sign(sim))) if n_done else None,
                "n_sign_agree_real_vs_simulated": int(np.sum(np.sign(enc) == np.sign(sim))) if n_done else 0,
                "agreement_real_vs_plaintext_xgboost": float(np.mean(np.sign(enc) == np.sign(plain_ok))) if n_done else None,
                "normalized_mae_real_vs_simulated": float(err.mean() / span) if n_done else None,
                "max_abs_error_real_vs_simulated": float(err.max()) if n_done else None,
                "latency_wall_clock_s_mean": float(np.mean(wall)) if n_done else None,
                "latency_wall_clock_s_std_within_cell": float(np.std(wall, ddof=1)) if n_done > 1 else None,
                "latency_phase_sum_s_mean": float(np.mean(phase_sum)) if n_done else None,
            })
            c = cells[-1]
            if c["n_completed"]:
                print(f"seed={seed:>3d} k={k}: completed {c['n_completed']}/{c['n']} agreement(real,sim)="
                      f"{c['agreement_real_vs_simulated']:.4f} nMAE={c['normalized_mae_real_vs_simulated']:.2e} "
                      f"latency={c['latency_wall_clock_s_mean']:.3f}s", flush=True)
            else:
                print(f"seed={seed:>3d} k={k}: NO evaluation completed ({c['failures'][0]['exception_type']})", flush=True)

    df = pd.DataFrame(cells)

    def stat(series):
        v = series.dropna()
        return {"mean": float(v.mean()), "sd_across_seeds": float(v.std(ddof=1)) if len(v) > 1 else None,
                "min": float(v.min()), "max": float(v.max())} if len(v) else None

    pooled = {}
    for k in K_EXECUTED:
        g = df[df.k == k]
        pooled[f"k={k}"] = {"n_seeds": int(len(g)), "n_patient_evaluations": int(g["n"].sum()),
                            "n_completed": int(g["n_completed"].sum()), "n_failed": int(g["n_failed"].sum()),
                            "n_sign_agree_real_vs_simulated": int(g["n_sign_agree_real_vs_simulated"].sum())}
        for col in ("agreement_real_vs_simulated", "agreement_real_vs_plaintext_xgboost", "normalized_mae_real_vs_simulated",
                    "max_abs_error_real_vs_simulated", "latency_wall_clock_s_mean"):
            pooled[f"k={k}"][col] = stat(g[col])

    # Criterion (pre-specification): (a) every evaluation completes, (b) every sign agrees, (c) normalized MAE within the
    # range of the original evaluator's k = 3/5/7 cells. "Within" is read as no larger than the largest original cell
    # (a smaller error is not a failure); the original range itself is reported so the reading can be checked.
    with open(ORIGINAL_PATH, "r", encoding="utf-8") as f:
        orig_cells = json.load(f)["per_cell"]
    orig_nmae = [c["normalized_mae_real_vs_simulated"] for c in orig_cells]
    lo, hi = float(min(orig_nmae)), float(max(orig_nmae))
    criterion = {"original_evaluator_nmae_range_k357": {"min": lo, "max": hi, "n_cells": len(orig_nmae)}, "per_k": {}}
    for k in K_EXECUTED:
        g = df[df.k == k]
        done_all = bool((g["n_failed"] == 0).all() and (g["n_completed"] == g["n"]).all())
        agree_all = bool(int(g["n_sign_agree_real_vs_simulated"].sum()) == int(g["n"].sum()))
        nmae_cells = [float(v) for v in g["normalized_mae_real_vs_simulated"].dropna()]
        nmae_ok = bool(len(nmae_cells) == len(g) and max(nmae_cells) <= hi)
        criterion["per_k"][f"k={k}"] = {
            "a_all_evaluations_complete": done_all, "b_all_signs_agree": agree_all,
            "c_nmae_no_larger_than_original_range_max": nmae_ok,
            "c_nmae_inside_original_range_both_ends": bool(nmae_ok and min(nmae_cells) >= lo),
            "executable_by_criterion": bool(done_all and agree_all and nmae_ok)}

    out = {
        "protocol": "docs/revision_prespec_m1.md (E1); design of docs/real_ckks_multiseed_protocol.md unchanged except the evaluator and k = 9",
        "backend": "REAL TenSEAL/SEAL ciphertext execution, compared against the SIMULATED circuit-faithful numpy evaluator",
        "polynomial_evaluator": "ckks_circuit.eval_poly_tenseal_folded (c_p z^p computed as (c_p z) z^(p-1); same polynomial)",
        "label_same_for_all_seeds": LABEL, "seeds": list(MODEL_SEEDS), "k_executed": list(K_EXECUTED),
        "n_per_seed": N_PER_SEED, "population": "TEST split, frozen stratified samples (12 pred-positive + 13 pred-negative)",
        "locked_parameters": {"N": LOCKED_N, "coeff_modulus_bit_sizes": LOCKED_CHAIN, "scale_bits": LOCKED_SCALE_BITS},
        "coefficients": {"file": os.path.basename(coeff_path), "sha256": sha256(coeff_path), "version": coeff_meta.get("version"),
                         "gamma": coeff_meta.get("gamma"), "gamma_anchor_percentile": coeff_meta.get("gamma_anchor_percentile")},
        "sample_manifest_sha256": sha256(MANIFEST_PATH),
        "per_cell": cells,
        "pooled_across_seeds": pooled,
        "k9_executable_criterion": criterion,
        "limitation": ("n=25 per seed: coarse agreement resolution (1/25); this supports 'the simulation is faithful across "
                       "independently trained models', not an estimate of encrypted accuracy."),
        "runtime_seconds": round(time.time() - t_run_start, 1),
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\nPooled across the 5 seeds (mean +/- SD across seeds):")
    for k in K_EXECUTED:
        p = pooled[f"k={k}"]
        a, n, lt = p["agreement_real_vs_simulated"], p["normalized_mae_real_vs_simulated"], p["latency_wall_clock_s_mean"]
        print(f"  k={k}: completed {p['n_completed']}/{p['n_patient_evaluations']}  agreement(real,sim)={a['mean']:.4f}   "
              f"nMAE={n['mean']:.2e}   latency={lt['mean']:.3f}s  criterion: {criterion['per_k'][f'k={k}']}")
    print(f"written -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
