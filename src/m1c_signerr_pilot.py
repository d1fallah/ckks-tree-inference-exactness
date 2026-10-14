#!/usr/bin/env python3
r"""
Validity-gate pilot of the sign-of-error check (GATES.md: "Question gate, M1 sign of the margin error" and "Validity gate, M1 sign of the margin error", both signed by the author 2026-10-13).

Seed 456, 8 bins, locked seven-level chain, the thermometer study's own frozen stratified sample (tc.stratified_sample). Three parts, aggregate output only:
  1. Data realism: the regenerated sample must have the counts stored in results/m1c_thermo_samechain_seed456.json, and exactly one sampled stay must be a candidate for the
     80%-sensitivity disagreement of both circuits. The stored results hold the absolute margin error of every sampled stay in sample order (no identifiers) but not the
     decrypted values, so a stay can have disagreed in the stored run only if its stored error is at least its distance to the threshold; candidates are those stays, per circuit.
  2. Positive control: three sampled stays far from both thresholds (largest smaller gap, never the stay of interest). For the broadcast and the packed circuit a known offset
     (0, +1e-4, -1e-4) is added to the encrypted margin before decryption (th._balanced_product is wrapped so that .sum() adds the plaintext constant; the circuit code itself is unchanged),
     R = 5 fresh encryptions per condition. The recorded signed error is decrypted margin minus the exact quantized model's margin. Per repetition the shift is the signed error
     minus the mean zero-offset signed error of the same stay and circuit; the pre-specified criterion is: the shift has the sign of the offset and differs from the offset by at most
     the largest absolute error stored for that circuit. Counted per repetition (the gate's wording, "in every repetition") and, as an added summary, pooled.
  3. Nothing is printed or stored that identifies a stay: no feature, identifier, timestamp or position in the sample.
Output: results/m1c_signerr_pilot.json (aggregate). Run in WSL.
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
import m1c_final_common as fc  # noqa: E402
import m1c_thermo as th  # noqa: E402
import m1c_thermo_final_common as tc  # noqa: E402
import ckks_circuit as cc  # noqa: E402

GATE_TITLES = ("Question gate, M1 sign of the margin error", "Validity gate, M1 sign of the margin error")
SEED, B, R_REPS, OFFSET, N_PILOT_STAYS = 456, 8, 5, 1e-4, 3
STORED = os.path.join(fc.RESULTS, "m1c_thermo_samechain_seed456.json")
OUT = os.path.join(fc.RESULTS, "m1c_signerr_pilot.json")


class _Offset:
    """Wraps the product ciphertext so that .sum() returns the sum plus a plaintext constant (the positive-control offset)."""

    def __init__(self, prod, off):
        self.prod, self.off = prod, off

    def sum(self):
        s = self.prod.sum()
        return s + self.off if self.off else s


_ORIG_PRODUCT = th._balanced_product


_INSIDE = [False]


def with_offset(off: float) -> None:
    """Wrap only the outermost call: _balanced_product recurses through the module name, and nested calls must return plain ciphertexts."""
    def wrapped(f, _o=off):
        if _INSIDE[0]:
            return _ORIG_PRODUCT(f)
        _INSIDE[0] = True
        try:
            r = _ORIG_PRODUCT(f)
        finally:
            _INSIDE[0] = False
        return _Offset(r, _o)
    th._balanced_product = wrapped


def main() -> int:
    t_start = time.perf_counter()
    tc.require_gates()
    for t in GATE_TITLES:
        fc.require_signed_entry(t)
    with open(STORED, "r", encoding="utf-8") as f:
        stored = json.load(f)["by_B"][f"B={B}"]
    fr = tc.load_frames(False)
    feats = fr["feats"]
    bst, Xte_b, m_te, ops = tc.load_model(fr, SEED, B, False)
    full = tc.stratified_sample(m_te, ops["f1"], tc.PER_CLASS)
    m_sel = m_te[full]
    pp = m_sel >= ops["f1"]
    counts = {"n_sample": int(len(full)), "n_plaintext_positive": int(pp.sum()), "n_plaintext_negative": int((~pp).sum())}
    counts_match = all(counts[k] == stored["sample_counts"][k] for k in counts)
    gap80 = np.abs(m_sel - ops["sens80"])
    gapf1 = np.abs(m_sel - ops["f1"])
    cand = {}
    for n in ("broadcast", "packed"):
        e = np.asarray(stored[n]["per_stay_abs_margin_error"])
        assert len(e) == len(m_sel)
        cand[n] = set(int(i) for i in np.where(e >= gap80)[0])
    inter = sorted(cand["broadcast"] & cand["packed"])
    realism = {"sample_counts": counts, "sample_counts_match_stored": bool(counts_match),
               "stored_agreeing_sens80": {n: int(stored[n]["decisions_agreeing_sens80"]) for n in ("broadcast", "packed")},
               "n_candidates_per_circuit": {n: len(cand[n]) for n in cand}, "n_candidates_both_circuits": len(inter)}
    realism["passes"] = bool(counts_match and len(inter) == 1 and all(v == len(m_sel) - 1 for v in realism["stored_agreeing_sens80"].values()))
    print("data realism:", json.dumps(realism), flush=True)
    out = {"seed": SEED, "B": B, "chain": tc.CHAIN7, "reps": R_REPS, "offset": OFFSET, "gates": list(GATE_TITLES), "data_realism": realism}
    if not realism["passes"]:
        out["stopped"] = "data realism failed; the positive control was not run (gate: stop and report, nothing chosen by hand)"
        json.dump(out, open(OUT, "w", encoding="utf-8"), indent=2)
        print("STOP: data realism failed; see", OUT)
        return 1
    interest = inter[0]
    far = [int(i) for i in np.argsort(-np.minimum(gapf1, gap80)) if int(i) != interest][:N_PILOT_STAYS]
    out["pilot_stays_min_gap_range"] = [float(np.minimum(gapf1, gap80)[far].min()), float(np.minimum(gapf1, gap80)[far].max())]

    ctx7 = tc.make_ctx(16384, tc.CHAIN7, 40)
    structure = cc.extract_ensemble_leaf_circuit(bst, set(feats), tree_depth=6)
    levels = th.level_arrays(structure, feats, B)
    bits = th.thermometer_bits(Xte_b.to_numpy(dtype=np.float64), B)[full]
    pv = th.folded_plain_vectors(levels, structure.leaf_value, B)
    mats = th.build_packed(levels, structure.leaf_value, B, len(feats))
    tol = {n: float(stored[n]["max_abs_margin_error"]) for n in ("broadcast", "packed")}
    res = {}
    for n in ("broadcast", "packed"):
        rec = {off: [[] for _ in far] for off in (0.0, OFFSET, -OFFSET)}
        for rep in range(R_REPS):
            for off in (0.0, OFFSET, -OFFSET):
                with_offset(off)
                for k, j in enumerate(far):
                    if n == "broadcast":
                        real, _ = th.eval_thermo_tenseal(pv, th.encrypt_bits(ctx7, bits[j]))
                    else:
                        real, _ = th.eval_thermo_packed_tenseal(ctx7, mats, bits[j])
                    rec[off][k].append(real - float(m_sel[j]))
            print(f"  {n}: repetition {rep + 1}/{R_REPS} [{time.perf_counter() - t_start:.0f}s]", flush=True)
        th._balanced_product = _ORIG_PRODUCT
        shifts = {+1: [], -1: []}
        for k in range(len(far)):
            base = float(np.mean(rec[0.0][k]))
            for sgn, off in ((+1, OFFSET), (-1, -OFFSET)):
                shifts[sgn] += [e - base for e in rec[off][k]]
        ok = {}
        for sgn, off in ((+1, OFFSET), (-1, -OFFSET)):
            s = np.asarray(shifts[sgn])
            right_sign = (np.sign(s) == sgn)
            within = (np.abs(s - off) <= tol[n])
            ok[f"{'+' if sgn > 0 else '-'}{OFFSET:g}"] = {"n_reps": int(len(s)), "n_right_sign": int(right_sign.sum()), "n_within_tolerance": int(within.sum()),
                                                    "n_pass_both": int((right_sign & within).sum()), "mean_shift": float(s.mean()), "sd_shift": float(s.std(ddof=1)),
                                                    "all_pass_both": bool((right_sign & within).all()), "pooled_mean_has_right_sign": bool(np.sign(s.mean()) == sgn)}
        zero = np.concatenate([np.asarray(rec[0.0][k]) for k in range(len(far))])
        res[n] = {"tolerance_max_abs_error_stored": tol[n], "zero_offset_signed_error_mean": float(zero.mean()), "zero_offset_signed_error_sd": float(zero.std(ddof=1)),
                  "by_offset": ok, "passes_every_repetition": bool(all(v["all_pass_both"] for v in ok.values())),
                  "passes_pooled": bool(all(v["pooled_mean_has_right_sign"] for v in ok.values()))}
        print(n, json.dumps(res[n]), flush=True)
    out["positive_control"] = res
    out["runtime_seconds"] = time.perf_counter() - t_start
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=2)
    print(f"written -> {OUT} ({out['runtime_seconds']:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
