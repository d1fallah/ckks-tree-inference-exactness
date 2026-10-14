#!/usr/bin/env python3
r"""
Same-chain comparison of the three circuits (docs/m1c_thermo_samechain_prespec.md; GATES.md entries "Question gate", "Validity gate" and "Analysis-plan gate, M1 same-chain
comparison of the three circuits", signed by the author 2026-10-13).

Packed thermometer, broadcast thermometer and (B = 5 only) exact interpolation on the LOCKED seven-level chain, on the thermometer study's own frozen stratified sample
(tc.stratified_sample: 50 plaintext-positive then 50 plaintext-negative stays at the F1 threshold, generator seed 42), all in one session per seed, the circuits alternating
in a rotating order per stay. One seed per call (full run) or seeds 42 and 123 at B = 5 with four stays (--smoke, an engineering check and not a result).
No stay identifier is printed or stored; the output holds aggregates plus per-stay timing and error arrays in sample order without identifiers.
Output: results/m1c_thermo_samechain_seed{S}.json (full), results/m1c_thermo_samechain_smoke.json (smoke).
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time

import numpy as np
import tenseal as ts

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import ckks_circuit as cc  # noqa: E402
import m1c_final_common as fc  # noqa: E402
import m1c_thermo as th  # noqa: E402
import m1c_thermo_final_common as tc  # noqa: E402
from m1c_final_run import build_exact_circuit  # noqa: E402

GATE_TITLES = ("Question gate, M1 same-chain comparison of the three circuits",
               "Validity gate, M1 same-chain comparison of the three circuits",
               "Analysis-plan gate, M1 same-chain comparison of the three circuits")


def require_gates() -> None:
    tc.require_gates()
    for t in GATE_TITLES:
        fc.require_signed_entry(t)


def reference_timing(ref_ct, ref_plain, reps: int = 20) -> float:
    """Fixed micro-benchmark for background load: mean seconds of one ciphertext-by-plaintext multiplication. Reported, never used to adjust a result."""
    ts_ = []
    for _ in range(reps):
        t0 = time.perf_counter()
        _ = ref_ct * ref_plain
        ts_.append(time.perf_counter() - t0)
    return float(np.mean(ts_))


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def run_seed(seed: int, ctx7, Bs, smoke: bool, ref_ct, ref_plain, fr, t_start: float) -> dict:
    feats = fr["feats"]
    res = {"seed": seed, "by_B": {}}
    for B in Bs:
        bst, Xte_b, m_te, ops = tc.load_model(fr, seed, B, False)
        full = tc.stratified_sample(m_te, ops["f1"], tc.PER_CLASS)
        pp = (m_te[full] >= ops["f1"])
        counts = {"n_sample": int(len(full)), "n_plaintext_positive": int(pp.sum()), "n_plaintext_negative": int((~pp).sum()),
                  "positives_first_then_negatives": bool(pp[: tc.PER_CLASS].all() and (~pp[tc.PER_CLASS:]).all())}
        assert counts["n_plaintext_positive"] == tc.PER_CLASS and counts["n_plaintext_negative"] == tc.PER_CLASS and counts["positives_first_then_negatives"], counts
        if smoke:
            half = 2
            idx = np.concatenate([full[:half], full[tc.PER_CLASS: tc.PER_CLASS + half]])
        else:
            idx = full
        m_sel = m_te[idx]
        structure = cc.extract_ensemble_leaf_circuit(bst, set(feats), tree_depth=6)
        levels = th.level_arrays(structure, feats, B)
        bits = th.thermometer_bits(Xte_b.to_numpy(dtype=np.float64), B)[idx]
        pv = th.folded_plain_vectors(levels, structure.leaf_value, B)
        mats = th.build_packed(levels, structure.leaf_value, B, len(feats))
        names = ["broadcast", "packed"]
        interp = None
        if B == 5:
            so, ss, norm, coeffs, k = build_exact_circuit(bst, feats, B)
            interp = (ss, coeffs, norm, k)
            names.append("interpolation")
        rec = {n: {"secs": [], "errs": [], "reals": []} for n in names}
        up = {}
        ref_before = reference_timing(ref_ct, ref_plain)
        for j in range(len(idx)):
            r = j % len(names)
            order = names[r:] + names[:r]
            for n in order:
                t0 = time.perf_counter()
                if n == "broadcast":
                    cts = th.encrypt_bits(ctx7, bits[j])
                    real, _ = th.eval_thermo_tenseal(pv, cts)
                elif n == "packed":
                    real, _ = th.eval_thermo_packed_tenseal(ctx7, mats, bits[j])
                else:
                    ss, coeffs, norm, k = interp
                    real, _ = cc.evaluate_circuit_tenseal(ctx7, ss, coeffs, Xte_b.iloc[int(idx[j])], norm, 1.0, poly_evaluator=cc.eval_poly_tenseal_folded)
                dt = time.perf_counter() - t0
                rec[n]["secs"].append(dt)
                rec[n]["errs"].append(abs(real - float(m_sel[j])))
                rec[n]["reals"].append(real)
                if n == "broadcast" and "broadcast" not in up:
                    up["broadcast"] = th.serialized_bytes(cts)
        ref_after = reference_timing(ref_ct, ref_plain)
        up["packed"] = len(ts.ckks_vector(ctx7, [float(b) for b in bits[0]]).serialize())
        if interp is not None:
            one = len(ts.ckks_vector(ctx7, [0.0] * structure.leaf_value.shape[0]).serialize())
            up["interpolation"] = 6 * one
        R = {"B": B, "sample_counts": counts, "n_stays_run": int(len(idx)), "circuit_order_rule": "rotating by stay index" if len(names) == 3 else "alternating by stay index",
             "reference_timing_seconds_before": ref_before, "reference_timing_seconds_after": ref_after, "peak_rss_mb_so_far": peak_rss_mb()}
        for n in names:
            s = tc.circuit_summary(rec[n]["errs"], rec[n]["reals"], m_sel, rec[n]["secs"], up[n], ops)
            s["per_stay_seconds"] = [float(x) for x in rec[n]["secs"]]
            s["per_stay_abs_margin_error"] = [float(x) for x in rec[n]["errs"]]
            if n == "interpolation":
                s["degree"] = interp[3]
            R[n] = s
            print(f"seed {seed} B={B} {n}: {s['mean_seconds_per_stay']:.2f} s/stay (SD {s['sd_seconds_per_stay']:.2f}), upload {s['client_upload_bytes'] / 1e6:.2f} MB, "
                  f"max err {s['max_abs_margin_error']:.2e}, F1-point agree {s['decisions_agreeing']}/{s['n_stays']} [{time.perf_counter() - t_start:.0f}s]", flush=True)
        print(f"seed {seed} B={B}: reference timing {ref_before * 1e3:.2f} ms before, {ref_after * 1e3:.2f} ms after; sample {counts['n_plaintext_positive']} + {counts['n_plaintext_negative']}; peak RSS {peak_rss_mb():.0f} MB", flush=True)
        res["by_B"][f"B={B}"] = R
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if not a.smoke:
        assert a.seed in fc.SEEDS, f"--seed must be one of {fc.SEEDS}"
    t_start = time.perf_counter()
    require_gates()
    fr = tc.load_frames(False)  # the thermometer study's own frozen test sample, as in its stage 2 and locked arm
    ctx7 = tc.make_ctx(16384, tc.CHAIN7, 40)
    ref_ct = ts.ckks_vector(ctx7, [1.0] * th.N_SLOTS)
    ref_plain = [0.5] * th.N_SLOTS
    out = {"smoke": a.smoke, "chain": tc.CHAIN7, "key_bytes_one_off": tc.key_bytes(ctx7), "prespec": "docs/m1c_thermo_samechain_prespec.md"}
    if a.smoke:
        out["seeds"] = {}
        for seed in (42, 123):
            out["seeds"][str(seed)] = run_seed(seed, ctx7, (5,), True, ref_ct, ref_plain, fr, t_start)
        path = os.path.join(fc.RESULTS, "m1c_thermo_samechain_smoke.json")
    else:
        out["seed"] = a.seed
        path = os.path.join(fc.RESULTS, f"m1c_thermo_samechain_seed{a.seed}.json")
        out["by_B"] = {}
        for B in tc.LEVELS_B:
            part = run_seed(a.seed, ctx7, (B,), False, ref_ct, ref_plain, fr, t_start)
            out["by_B"].update(part["by_B"])
            with open(path.replace(".json", "_partial.json"), "w", encoding="utf-8") as f:
                json.dump(out, f)
    out["runtime_seconds"] = time.perf_counter() - t_start
    out["peak_rss_mb"] = peak_rss_mb()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"written -> {path} ({out['runtime_seconds']:.0f}s, peak RSS {out['peak_rss_mb']:.0f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
