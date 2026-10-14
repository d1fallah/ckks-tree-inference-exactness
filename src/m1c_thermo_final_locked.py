#!/usr/bin/env python3
"""
Thermometer final run, stage 3 (added arm, docs/m1c_thermo_final_prespec.md): broadcast and packed thermometer on the LOCKED seven-level chain, seed 42,
B = 5, 8, 16, on the first 25 plaintext-positive and first 25 plaintext-negative stays of the stage-2 sample. Isolates the protocol from the parameters.
Aggregate output only: results/m1c_thermo_final_locked.json.
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


def main() -> int:
    smoke = "--smoke" in sys.argv
    seed = 42
    half = 2 if smoke else 25
    t_start = time.perf_counter()
    tc.require_gates()
    fr = tc.load_frames(smoke)
    feats = fr["feats"]
    ctx7 = tc.make_ctx(16384, tc.CHAIN7, 40)
    out = {"seed": seed, "smoke": smoke, "chain": tc.CHAIN7, "key_bytes_one_off": tc.key_bytes(ctx7),
           "sample": f"{2 * half} stays: first {half} plaintext-positive and first {half} plaintext-negative stays of the stage-2 sample", "by_B": {}}
    for B in tc.LEVELS_B:
        bst, Xte_b, m_te, ops = tc.load_model(fr, seed, B, smoke)
        full = tc.stratified_sample(m_te, ops["f1"], 2 if smoke else tc.PER_CLASS)
        n = len(full) // 2
        idx = np.concatenate([full[:half], full[n : n + half]])
        m_sel = m_te[idx]
        structure = cc.extract_ensemble_leaf_circuit(bst, set(feats), tree_depth=6)
        levels = th.level_arrays(structure, feats, B)
        bits = th.thermometer_bits(Xte_b.to_numpy(dtype=np.float64), B)[idx]
        R = {"B": B}
        pv = th.folded_plain_vectors(levels, structure.leaf_value, B)
        errs, secs, reals, up = [], [], [], None
        for j in range(len(idx)):
            t0 = time.perf_counter()
            cts = th.encrypt_bits(ctx7, bits[j])
            real, _ = th.eval_thermo_tenseal(pv, cts)
            secs.append(time.perf_counter() - t0)
            errs.append(abs(real - m_sel[j])); reals.append(real)
            up = up or th.serialized_bytes(cts)
        R["broadcast_thermometer_locked_chain"] = tc.circuit_summary(errs, reals, m_sel, secs, up, ops)
        mats = th.build_packed(levels, structure.leaf_value, B, len(feats))
        errs, secs, reals = [], [], []
        for j in range(len(idx)):
            t0 = time.perf_counter()
            real, _ = th.eval_thermo_packed_tenseal(ctx7, mats, bits[j])
            secs.append(time.perf_counter() - t0)
            errs.append(abs(real - m_sel[j])); reals.append(real)
        up = len(ts.ckks_vector(ctx7, [float(b) for b in bits[0]]).serialize())
        R["packed_thermometer_locked_chain"] = tc.circuit_summary(errs, reals, m_sel, secs, up, ops)
        for key in ("broadcast_thermometer_locked_chain", "packed_thermometer_locked_chain"):
            r = R[key]
            print(f"B={B} {key}: max err {r['max_abs_margin_error']:.2e}, agree {r['decisions_agreeing']}/{r['n_stays']}, "
                  f"{r['mean_seconds_per_stay']:.2f} s/stay, upload {r['client_upload_bytes'] / 1e6:.2f} MB [{time.perf_counter() - t_start:.0f}s]", flush=True)
        out["by_B"][f"B={B}"] = R
    out["runtime_seconds"] = time.perf_counter() - t_start
    with open(tc.out_path("locked", None, smoke), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"written -> {tc.out_path('locked', None, smoke)} ({out['runtime_seconds']:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
