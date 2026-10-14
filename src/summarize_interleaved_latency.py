#!/usr/bin/env python3
"""
Summarize results/m1_interleaved_latency.json (added analysis S2.3a): per degree and evaluator, the mean wall-clock
latency in each of the two interleaved rounds and across them, and the folding/original ratio per round.
Aggregate only. Output: results/m1_interleaved_latency_summary.json
"""
from __future__ import annotations

import json
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(_HERE, "..", "results")


def main() -> int:
    with open(os.path.join(RES, "m1_interleaved_latency.json"), "r", encoding="utf-8") as f:
        d = json.load(f)
    by = {}
    for r in d["runs"]:
        w = r["timing_per_patient"]["wall_clock_total_s"]
        by.setdefault((r["evaluator"], r["k"]), {})[r["round"]] = (w["mean_s"], w["sd_s"], r["n_timed"])
    out = {"protocol": "docs/step2_prespec_m1.md section C (added analysis S2.3a)", "order": d["order"], "by_k": {}}
    for k in (3, 5, 7, 9):
        entry = {}
        for ev in ("original", "folding"):
            if (ev, k) in by:
                rounds = by[(ev, k)]
                means = [rounds[1][0], rounds[2][0]]
                entry[ev] = {"round_1_mean_s": rounds[1][0], "round_1_sd_s": rounds[1][1],
                             "round_2_mean_s": rounds[2][0], "round_2_sd_s": rounds[2][1],
                             "mean_of_round_means_s": float(np.mean(means)), "n_timed_per_round": rounds[1][2]}
        if "original" in entry and "folding" in entry:
            ratios = [by[("folding", k)][r][0] / by[("original", k)][r][0] for r in (1, 2)]
            entry["folding_over_original_ratio_round_1"] = ratios[0]
            entry["folding_over_original_ratio_round_2"] = ratios[1]
            entry["folding_over_original_ratio_mean"] = float(np.mean(ratios))
        out["by_k"][f"k={k}"] = entry
    with open(os.path.join(RES, "m1_interleaved_latency_summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    for k, e in out["by_k"].items():
        print(k, {n: (round(v["mean_of_round_means_s"], 3) if isinstance(v, dict) else round(v, 3)) for n, v in e.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
