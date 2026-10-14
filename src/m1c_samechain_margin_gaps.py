#!/usr/bin/env python3
r"""
Added analysis of the Analysis-plan gate "M1 same-chain comparison of the three circuits" (docs/m1c_thermo_samechain_prespec.md, section 7): margin gaps.
For each seed, bin count and threshold (F1 and 80% sensitivity): the smallest distance between an attained plaintext margin and the threshold over (a) the 100
sampled stays and (b) the full test split, and the number of full-split stays within the 5e-4 gate of the threshold. Read-only; aggregate output only, no per-stay value.
Output: results/m1c_samechain_margin_gaps.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import m1c_final_common as fc  # noqa: E402
import m1c_thermo_final_common as tc  # noqa: E402

GATES = ("Question gate, M1 same-chain comparison of the three circuits",
         "Validity gate, M1 same-chain comparison of the three circuits",
         "Analysis-plan gate, M1 same-chain comparison of the three circuits")


def main() -> int:
    tc.require_gates()
    for t in GATES:
        fc.require_signed_entry(t)
    fr = tc.load_frames(False)
    out = {"note": "docs/m1c_thermo_samechain_prespec.md section 7 (added analysis); distances are between plaintext margins and the decision thresholds",
           "gate": tc.GATE, "n_test_stays": int(len(fr["Xte"])), "by_seed": {}}
    for seed in fc.SEEDS:
        out["by_seed"][str(seed)] = {}
        for B in tc.LEVELS_B:
            bst, Xte_b, m_te, ops = tc.load_model(fr, seed, B, False)
            idx = tc.stratified_sample(m_te, ops["f1"], tc.PER_CLASS)
            R = {}
            for name in ("f1", "sens80"):
                gap = np.abs(m_te - ops[name])
                R[name] = {"threshold": float(ops[name]),
                           "min_gap_sample_100": float(gap[idx].min()),
                           "min_gap_full_test_split": float(gap.min()),
                           "n_full_split_within_gate": int((gap <= tc.GATE).sum())}
            out["by_seed"][str(seed)][f"B={B}"] = R
            print(f"seed {seed} B={B}: min gap sample {R['f1']['min_gap_sample_100']:.2e} (F1), {R['sens80']['min_gap_sample_100']:.2e} (80%); "
                  f"full split {R['f1']['min_gap_full_test_split']:.2e} / {R['sens80']['min_gap_full_test_split']:.2e}; "
                  f"within gate {R['f1']['n_full_split_within_gate']} / {R['sens80']['n_full_split_within_gate']}", flush=True)
    path = os.path.join(fc.RESULTS, "m1c_samechain_margin_gaps.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"written -> {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
