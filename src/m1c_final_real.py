#!/usr/bin/env python3
"""
Final run, real ciphertexts (Analysis-plan gate in GATES.md): for every seed, the exact 5-bin circuit (degree 9, folding evaluator,
locked context) on a frozen stratified test sample of 200 stays (100 plaintext-positive, 100 plaintext-negative at the F1 threshold,
generator seed 42), margin gate 5e-4. Needs the artifacts written by m1c_final_run.py. Aggregate output only:
results/m1c_final_real.json.
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

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import ckks_circuit as cc  # noqa: E402
import m1c_final_common as fc  # noqa: E402
from m1c_final_run import build_exact_circuit  # noqa: E402
from m1c_pilot_validity import simulate  # noqa: E402

OUT = os.path.join(fc.RESULTS, "m1c_final_real.json")
PER_CLASS, GATE = 100, 5e-4
N_VALUE, COEFF_MOD_BIT_SIZES, SCALE = 16384, [60, 40, 40, 40, 40, 40, 40, 40, 60], 2**40


def main() -> int:
    t_start = time.perf_counter()
    fc.require_signed_gates()
    with open(fc.DECISION_PATH, "r", encoding="utf-8") as f:
        use_acuity = bool(json.load(f)["use_acuity"])
    fr = fc.load_frames(use_acuity, need_test=True)
    feats = fr["feats"]
    ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=N_VALUE, coeff_mod_bit_sizes=COEFF_MOD_BIT_SIZES)
    ctx.global_scale = SCALE
    ctx.generate_galois_keys()
    per_seed, all_err, all_secs, n_agree, n_total = {}, [], [], 0, 0
    for seed in fc.SEEDS:
        with open(os.path.join(fc.ARTIFACT_DIR, f"seed{seed}_q5_params.json"), "r", encoding="utf-8") as f:
            par = json.load(f)
        assert par["feats"] == feats and par["use_acuity"] == use_acuity
        edges = {c: np.asarray(e) for c, e in par["edges"].items()}

        def binned(X):
            cols = {}
            for c in feats:
                v = X[c].to_numpy(dtype=float)
                cols[c] = (np.clip(np.round(v) - 1, 0, 4) if c == fc.ACUITY else np.searchsorted(edges[c], v, side="right").astype(float))
            return pd.DataFrame(cols)
        Xte_b = binned(fr["Xte"])
        clf = xgb.XGBClassifier()
        clf.load_model(os.path.join(fc.ARTIFACT_DIR, f"seed{seed}_q5.json"))
        bst = clf.get_booster()
        m_te = bst.predict(xgb.DMatrix(Xte_b, feature_names=feats), output_margin=True)
        thr = par["ops"]["f1"]
        pp = (m_te >= thr).astype(int)
        so, ss, norm, coeffs, k = build_exact_circuit(bst, feats, par["B"])
        bl = cc.prepare_batch_levels(ss, feats, norm, 1.0)
        rng = np.random.default_rng(42)
        idx = np.concatenate([rng.choice(np.flatnonzero(pp == 1), PER_CLASS, replace=False),
                              rng.choice(np.flatnonzero(pp == 0), PER_CLASS, replace=False)])
        sim = simulate(bl, coeffs, ss.leaf_value, Xte_b.to_numpy(dtype=np.float64)[idx], 1.0)
        errs, secs, agree, sim_agree = [], [], 0, 0
        for j, i in enumerate(idx):
            t0 = time.perf_counter()
            real, _ = cc.evaluate_circuit_tenseal(ctx, ss, coeffs, Xte_b.iloc[i], norm, 1.0, poly_evaluator=cc.eval_poly_tenseal_folded)
            secs.append(time.perf_counter() - t0)
            errs.append(abs(real - sim[j]))
            agree += int((real >= thr) == (m_te[i] >= thr))
            sim_agree += int((sim[j] >= thr) == (m_te[i] >= thr))
        per_seed[str(seed)] = {"n_stays": int(len(idx)), "decisions_agreeing_with_booster": agree,
                               "simulated_decisions_agreeing_with_booster": sim_agree,
                               "max_abs_real_minus_sim": float(max(errs)), "mean_abs_real_minus_sim": float(np.mean(errs)),
                               "gate_passed": bool(max(errs) <= GATE), "mean_seconds_per_stay": float(np.mean(secs)),
                               "sd_seconds_per_stay": float(np.std(secs, ddof=1))}
        all_err += errs
        all_secs += secs
        n_agree += agree
        n_total += len(idx)
        print(f"seed {seed}: agree {agree}/{len(idx)}, max err {max(errs):.2e}, {np.mean(secs):.2f}s/stay "
              f"[{time.perf_counter() - t_start:.0f}s]", flush=True)
        with open(OUT.replace(".json", "_partial.json"), "w", encoding="utf-8") as f:
            json.dump(per_seed, f)
    out = {"protocol": "Analysis-plan gate in GATES.md (M1 continuous-feature task)",
           "backend": "REAL TenSEAL/SEAL ciphertext execution, locked parameters", "evaluator": "eval_poly_tenseal_folded",
           "degree": 2 * fc.B_PRIMARY - 1, "bins": fc.B_PRIMARY, "gate_tolerance": GATE,
           "sample": f"per seed {2 * PER_CLASS} test stays ({PER_CLASS} plaintext-positive, {PER_CLASS} plaintext-negative at the F1 threshold), generator seed 42",
           "per_seed": per_seed, "n_stays": n_total, "decisions_agreeing_with_booster": n_agree,
           "max_abs_real_minus_sim": float(max(all_err)), "all_gates_passed": bool(max(all_err) <= GATE),
           "mean_seconds_per_stay": float(np.mean(all_secs)), "sd_seconds_per_stay": float(np.std(all_secs, ddof=1)),
           "runtime_seconds": time.perf_counter() - t_start}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"real: agree {n_agree}/{n_total}; all gates passed: {out['all_gates_passed']}; written -> {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
