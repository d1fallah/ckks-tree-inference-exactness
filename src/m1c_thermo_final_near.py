#!/usr/bin/env python3
"""
Thermometer final run, stage 4 (docs/m1c_thermo_final_prespec.md; author's Validity-gate decision (a), with repetition): every test stay whose plaintext margin lies
within 5e-4 of either threshold, evaluated on real ciphertexts (broadcast variant, four-level chain) with R fresh encryptions each (R = 20 for seed 42, else 5).
Stop rule: if the union exceeds 100 stays for any B, write the counts and exit before evaluating anything. One seed per call.
Aggregate output only (counts, anonymous sorted flip rates, maxima): results/m1c_thermo_final_near_seed{S}.json.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import ckks_circuit as cc  # noqa: E402
import m1c_thermo as th  # noqa: E402
import m1c_thermo_final_common as tc  # noqa: E402

CAP = 100


def main() -> int:
    a = tc.parse_args()
    seed, smoke = a.seed, a.smoke
    reps = 2 if smoke else (20 if seed == 42 else 5)
    t_start = time.perf_counter()
    tc.require_gates()
    fr = tc.load_frames(smoke)
    feats = fr["feats"]
    models, plan = {}, {}
    for B in tc.LEVELS_B:
        bst, Xte_b, m_te, ops = tc.load_model(fr, seed, B, smoke)
        near = {n: np.flatnonzero(np.abs(m_te - ops[n]) <= tc.GATE) for n in ("f1", "sens80")}
        union = np.union1d(near["f1"], near["sens80"])
        if smoke:
            union = np.flatnonzero(np.abs(m_te - ops["f1"]) <= 0.02)[:4]  # engineering check: nearest-ish stays only
            near = {n: np.intersect1d(union, np.flatnonzero(np.abs(m_te - ops[n]) <= 0.02)) for n in near}
        models[B] = (bst, Xte_b, m_te, ops)
        plan[B] = {"n_f1": int(len(near["f1"])), "n_sens80": int(len(near["sens80"])), "n_union": int(len(union))}
        plan[B]["_union"], plan[B]["_near"] = union, near
    counts = {f"B={B}": {k: v for k, v in p.items() if not k.startswith("_")} for B, p in plan.items()}
    print(f"seed {seed}: stays within the gate of a threshold {counts}; R = {reps}", flush=True)
    out = {"seed": seed, "smoke": smoke, "repetitions": reps, "gate": tc.GATE, "counts": counts, "by_B": {}}
    if any(p["n_union"] > CAP for p in plan.values()):
        out["stopped"] = f"a union of near-threshold stays exceeds {CAP}; nothing evaluated (stop rule of the gate); ask the author"
        with open(tc.out_path("near", seed, smoke), "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print("STOP: " + out["stopped"], flush=True)
        return 2
    ctx4 = tc.make_ctx(16384, tc.CHAIN4, 40)
    for B in tc.LEVELS_B:
        bst, Xte_b, m_te, ops = models[B]
        union, near = plan[B]["_union"], plan[B]["_near"]
        R = {"B": B, "n_union": int(len(union))}
        if len(union):
            structure = cc.extract_ensemble_leaf_circuit(bst, set(feats), tree_depth=6)
            levels = th.level_arrays(structure, feats, B)
            pv = th.folded_plain_vectors(levels, structure.leaf_value, B)
            bits = th.thermometer_bits(Xte_b.to_numpy(dtype=np.float64), B)
            sim = th.simulate_margin(levels, structure.leaf_value, bits[union], B)
            real = np.empty((len(union), reps))
            secs = []
            for j, i in enumerate(union):
                for r in range(reps):
                    t0 = time.perf_counter()
                    real[j, r], _ = th.eval_thermo_tenseal(pv, th.encrypt_bits(ctx4, bits[i]))
                    secs.append(time.perf_counter() - t0)
            err = np.abs(real - m_te[union][:, None])
            R["max_abs_margin_error"] = float(err.max())
            R["mean_seconds_per_evaluation"] = float(np.mean(secs))
            for name in ("f1", "sens80"):
                sel = np.isin(union, near[name])
                pos = np.flatnonzero(sel)
                flips = np.array([[(real[j, r] >= ops[name]) != (m_te[union[j]] >= ops[name]) for r in range(reps)] for j in pos]) if len(pos) else np.zeros((0, reps), bool)
                rates = np.sort(flips.mean(axis=1)) if len(pos) else np.array([])
                R[name] = {"n_stays_within_gate": int(len(pos)), "n_flipped_in_at_least_one_repetition": int((flips.any(axis=1)).sum()) if len(pos) else 0,
                           "flip_rates_sorted": [float(x) for x in rates],
                           "n_simulated_decision_differs_from_booster": int(((sim[pos] >= ops[name]) != (m_te[union[pos]] >= ops[name])).sum()) if len(pos) else 0,
                           "min_abs_gap": float(np.min(np.abs(m_te[union[pos]] - ops[name]))) if len(pos) else None}
        out["by_B"][f"B={B}"] = R
        print(f"seed {seed} B={B}: " + ", ".join(f"{n}: {R[n]['n_flipped_in_at_least_one_repetition']}/{R[n]['n_stays_within_gate']} flipped" for n in ("f1", "sens80") if n in R)
              + f" [{time.perf_counter() - t_start:.0f}s]", flush=True)
    out["runtime_seconds"] = time.perf_counter() - t_start
    with open(tc.out_path("near", seed, smoke), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"written -> {tc.out_path('near', seed, smoke)} ({out['runtime_seconds']:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
