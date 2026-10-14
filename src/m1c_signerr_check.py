#!/usr/bin/env python3
r"""
Sign of the margin error on the one stay that disagreed at the 80%-sensitivity threshold (seed 456, 8 bins). Question, Validity and Analysis-plan gates "M1 sign of the margin error",
signed by the author 2026-10-13; the pilot is src/m1c_signerr_pilot.py (results/m1c_signerr_pilot.json).

The stay is found as in the pilot (stored-run rule: the stored absolute error of both circuits is at least the stay's distance to the threshold; exactly one such stay is required).
Broadcast and packed thermometer circuits on the locked seven-level chain, the same code as the same-chain study (th.eval_thermo_tenseal, th.eval_thermo_packed_tenseal, no offset),
R = 20 fresh encryptions per circuit in alternating order, one key context shared by both circuits (scope choice (a)). Recorded per circuit: the 20 signed errors (decrypted margin minus the
exact quantized model's margin), counts of each sign, mean and SD, whether each error pushes toward or away from the threshold, repetitions whose decision differs from the plaintext model;
and the stay's signed gap. Nothing that identifies the stay (feature, identifier, timestamp, position in the sample) is printed or stored. Output: results/m1c_signerr_check.json. Run in WSL.
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
import m1c_final_common as fc  # noqa: E402
import m1c_thermo as th  # noqa: E402
import m1c_thermo_final_common as tc  # noqa: E402

GATE_TITLES = ("Question gate, M1 sign of the margin error", "Validity gate, M1 sign of the margin error", "Analysis-plan gate, M1 sign of the margin error")
SEED, B, R_REPS = 456, 8, 20
STORED = os.path.join(fc.RESULTS, "m1c_thermo_samechain_seed456.json")
PILOT = os.path.join(fc.RESULTS, "m1c_signerr_pilot.json")
OUT = os.path.join(fc.RESULTS, "m1c_signerr_check.json")


def main() -> int:
    t_start = time.perf_counter()
    tc.require_gates()
    for t in GATE_TITLES:
        fc.require_signed_entry(t)
    with open(STORED, "r", encoding="utf-8") as f:
        stored = json.load(f)["by_B"][f"B={B}"]
    with open(PILOT, "r", encoding="utf-8") as f:
        pilot = json.load(f)
    fr = tc.load_frames(False)
    feats = fr["feats"]
    bst, Xte_b, m_te, ops = tc.load_model(fr, SEED, B, False)
    full = tc.stratified_sample(m_te, ops["f1"], tc.PER_CLASS)
    m_sel = m_te[full]
    gap80 = np.abs(m_sel - ops["sens80"])
    cand = {n: set(int(i) for i in np.where(np.asarray(stored[n]["per_stay_abs_margin_error"]) >= gap80)[0]) for n in ("broadcast", "packed")}
    inter = sorted(cand["broadcast"] & cand["packed"])
    assert pilot["data_realism"]["passes"] and len(inter) == 1, "the stay is not uniquely identified; stop and report (nothing is chosen by hand)"
    j = inter[0]
    thr = float(ops["sens80"])
    g = float(m_sel[j] - thr)  # signed gap: margin minus threshold
    side = 1 if g >= 0 else -1
    plain_dec = bool(m_sel[j] >= thr)

    ctx7 = tc.make_ctx(16384, tc.CHAIN7, 40)
    structure = cc.extract_ensemble_leaf_circuit(bst, set(feats), tree_depth=6)
    levels = th.level_arrays(structure, feats, B)
    bits = th.thermometer_bits(Xte_b.to_numpy(dtype=np.float64), B)[full][j]
    pv = th.folded_plain_vectors(levels, structure.leaf_value, B)
    mats = th.build_packed(levels, structure.leaf_value, B, len(feats))
    errs = {"broadcast": [], "packed": []}
    for rep in range(R_REPS):
        order = ("broadcast", "packed") if rep % 2 == 0 else ("packed", "broadcast")
        for n in order:
            if n == "broadcast":
                real, _ = th.eval_thermo_tenseal(pv, th.encrypt_bits(ctx7, bits))
            else:
                real, _ = th.eval_thermo_packed_tenseal(ctx7, mats, bits)
            errs[n].append(real - float(m_sel[j]))
        print(f"  repetition {rep + 1}/{R_REPS} [{time.perf_counter() - t_start:.0f}s]", flush=True)
    out = {"seed": SEED, "B": B, "chain": tc.CHAIN7, "reps": R_REPS, "threshold": "80%-sensitivity (post hoc midpoint rule of the study)", "key_contexts": 1,
           "stay_signed_gap_margin_minus_threshold": g, "stay_plaintext_decision_positive": plain_dec,
           "pilot_zero_offset_context": {n: {"mean": pilot["positive_control"][n]["zero_offset_signed_error_mean"], "sd": pilot["positive_control"][n]["zero_offset_signed_error_sd"]} for n in ("broadcast", "packed")}}
    for n in ("broadcast", "packed"):
        e = np.asarray(errs[n])
        dec_differs = int((((m_sel[j] + e) >= thr) != plain_dec).sum())
        out[n] = {"signed_errors": [float(x) for x in e], "n_positive": int((e > 0).sum()), "n_negative": int((e < 0).sum()), "mean": float(e.mean()), "sd": float(e.std(ddof=1)),
                  "n_toward_threshold": int((np.sign(e) == -side).sum()), "n_away_from_threshold": int((np.sign(e) == side).sum()),
                  "n_decision_differs_from_plaintext": dec_differs, "one_signed": bool((e > 0).all() or (e < 0).all()),
                  "abs_error_max": float(np.abs(e).max()), "abs_error_exceeds_gap": int((np.abs(e) >= abs(g)).sum())}
        print(n, json.dumps({k: v for k, v in out[n].items() if k != "signed_errors"}), flush=True)
    both = out["broadcast"]["one_signed"] and out["packed"]["one_signed"]
    same = both and np.sign(out["broadcast"]["mean"]) == np.sign(out["packed"]["mean"])
    out["outcome"] = ("(i) all errors of both circuits have one sign" + (", the same sign in both" if same else ", opposite signs in the two circuits")) if both else \
        ("(ii) signs vary within both circuits" if not (out["broadcast"]["one_signed"] or out["packed"]["one_signed"]) else "(iii) one circuit one-signed, the other mixed")
    out["runtime_seconds"] = time.perf_counter() - t_start
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("outcome:", out["outcome"], f"\nwritten -> {OUT} ({out['runtime_seconds']:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
