#!/usr/bin/env python3
"""
Thermometer final run, stage 2 (docs/m1c_thermo_final_prespec.md): real ciphertexts on a frozen stratified test sample (100 stays) per B: broadcast and packed
thermometer, client-indicator baseline (four-level chain); B = 5 also the exact interpolation circuit (locked chain, folding evaluator). One seed per call.
Aggregate output only: results/m1c_thermo_final_real_seed{S}.json.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import tenseal as ts

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import ckks_circuit as cc  # noqa: E402
import m1c_thermo as th  # noqa: E402
import m1c_thermo_final_common as tc  # noqa: E402
from m1c_final_run import build_exact_circuit  # noqa: E402


def main() -> int:
    a = tc.parse_args()
    seed, smoke = a.seed, a.smoke
    per_class = 2 if smoke else tc.PER_CLASS
    t_start = time.perf_counter()
    tc.require_gates()
    fr = tc.load_frames(smoke)
    feats = fr["feats"]
    ctx4 = tc.make_ctx(16384, tc.CHAIN4, 40)
    ctx7 = tc.make_ctx(16384, tc.CHAIN7, 40)
    out = {"seed": seed, "smoke": smoke, "sample": f"{2 * per_class} stays ({per_class} plaintext-positive, {per_class} plaintext-negative at the F1 threshold), generator seed 42",
           "chains": {"four_level": tc.CHAIN4, "locked_seven_level": tc.CHAIN7},
           "key_bytes_one_off": {"n16384_four_level": tc.key_bytes(ctx4), "n16384_locked_seven_level": tc.key_bytes(ctx7)}, "by_B": {}}
    for B in tc.LEVELS_B:
        bst, Xte_b, m_te, ops = tc.load_model(fr, seed, B, smoke)
        thr = ops["f1"]
        idx = tc.stratified_sample(m_te, thr, per_class)
        m_sel = m_te[idx]
        structure = cc.extract_ensemble_leaf_circuit(bst, set(feats), tree_depth=6)
        levels = th.level_arrays(structure, feats, B)
        bits = th.thermometer_bits(Xte_b.to_numpy(dtype=np.float64), B)[idx]
        R = {"B": B}
        # broadcast
        pv = th.folded_plain_vectors(levels, structure.leaf_value, B)
        errs, secs, reals, up = [], [], [], None
        for j in range(len(idx)):
            t0 = time.perf_counter()
            cts = th.encrypt_bits(ctx4, bits[j])
            real, _ = th.eval_thermo_tenseal(pv, cts)
            secs.append(time.perf_counter() - t0)
            errs.append(abs(real - m_sel[j])); reals.append(real)
            up = up or th.serialized_bytes(cts)
        R["broadcast_thermometer_4level"] = tc.circuit_summary(errs, reals, m_sel, secs, up, ops)
        # packed
        mats = th.build_packed(levels, structure.leaf_value, B, len(feats))
        errs, secs, reals = [], [], []
        for j in range(len(idx)):
            t0 = time.perf_counter()
            real, _ = th.eval_thermo_packed_tenseal(ctx4, mats, bits[j])
            secs.append(time.perf_counter() - t0)
            errs.append(abs(real - m_sel[j])); reals.append(real)
        up = len(ts.ckks_vector(ctx4, [float(b) for b in bits[0]]).serialize())
        R["packed_thermometer_4level"] = tc.circuit_summary(errs, reals, m_sel, secs, up, ops)
        # client-indicator baseline
        ind = th.client_indicators(levels, bits, B)
        errs, secs, reals = [], [], []
        for j in range(len(idx)):
            t0 = time.perf_counter()
            real, _ = th.eval_client_indicator_tenseal(ctx4, ind[j], structure.leaf_value)
            secs.append(time.perf_counter() - t0)
            errs.append(abs(real - m_sel[j])); reals.append(real)
        vec = np.zeros(th.N_SLOTS); vec[: ind.shape[1]] = ind[0]
        R["client_indicator_baseline_4level"] = tc.circuit_summary(errs, reals, m_sel, secs, len(ts.ckks_vector(ctx4, vec.tolist()).serialize()), ops)
        if B == 5:
            so, ss, norm, coeffs, k = build_exact_circuit(bst, feats, B)
            errs, secs, reals = [], [], []
            for j, i in enumerate(idx):
                t0 = time.perf_counter()
                real, _ = cc.evaluate_circuit_tenseal(ctx7, ss, coeffs, Xte_b.iloc[i], norm, 1.0, poly_evaluator=cc.eval_poly_tenseal_folded)
                secs.append(time.perf_counter() - t0)
                errs.append(abs(real - m_sel[j])); reals.append(real)
            one = len(ts.ckks_vector(ctx7, [0.0] * structure.leaf_value.shape[0]).serialize())
            R["exact_interpolation_7level"] = tc.circuit_summary(errs, reals, m_sel, secs, 6 * one, ops)
            R["exact_interpolation_7level"]["degree"] = k
        for key, r in R.items():
            if isinstance(r, dict) and "max_abs_margin_error" in r:
                print(f"seed {seed} B={B} {key}: max err {r['max_abs_margin_error']:.2e}, F1-point agree {r['decisions_agreeing']}/{r['n_stays']}, "
                      f"80%-sens-point agree {r['decisions_agreeing_sens80']}/{r['n_stays']}, {r['mean_seconds_per_stay']:.2f} s/stay, "
                      f"upload {r['client_upload_bytes'] / 1e6:.2f} MB [{time.perf_counter() - t_start:.0f}s]", flush=True)
        out["by_B"][f"B={B}"] = R
        with open(tc.out_path("real", seed, smoke).replace(".json", "_partial.json"), "w", encoding="utf-8") as f:
            json.dump(out, f)
    out["runtime_seconds"] = time.perf_counter() - t_start
    with open(tc.out_path("real", seed, smoke), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"written -> {tc.out_path('real', seed, smoke)} ({out['runtime_seconds']:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
