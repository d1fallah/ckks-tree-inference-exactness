#!/usr/bin/env python3
r"""
Summary of the same-chain run (docs/m1c_thermo_samechain_prespec.md, sections 6 and 10). Reads results/m1c_thermo_samechain_seed{S}.json for the five seeds and
results/m1c_samechain_margin_gaps.json; applies the decision rule fixed in the pre-specification:
  "A faster than B" only if A's per-seed mean latency is lower in all five seeds and the mean per-seed difference is at least 10% of B's mean; reversed likewise;
  otherwise "not distinguishable on this sample".
Aggregate output only: results/m1c_thermo_samechain_summary.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import m1c_final_common as fc  # noqa: E402

SEEDS = tuple(fc.SEEDS)
BS = (5, 8, 16)
RELATIVE_SIZE = 0.10  # fixed in the pre-specification before the run


def rule(a: np.ndarray, b: np.ndarray) -> str:
    """a, b: per-seed mean seconds of circuits A and B."""
    d = b - a  # positive when A is faster
    if np.all(d > 0) and d.mean() >= RELATIVE_SIZE * b.mean():
        return "A faster than B"
    if np.all(d < 0) and (-d).mean() >= RELATIVE_SIZE * a.mean():
        return "B faster than A"
    return "not distinguishable on this sample"


def main() -> int:
    runs = {}
    for s in SEEDS:
        p = os.path.join(fc.RESULTS, f"m1c_thermo_samechain_seed{s}.json")
        with open(p, "r", encoding="utf-8") as f:
            runs[s] = json.load(f)
    gaps_path = os.path.join(fc.RESULTS, "m1c_samechain_margin_gaps.json")
    gaps = json.load(open(gaps_path, encoding="utf-8")) if os.path.exists(gaps_path) else None
    out = {"prespec": "docs/m1c_thermo_samechain_prespec.md", "seeds": list(SEEDS), "n_stays_per_seed_and_B": 100, "relative_size_rule": RELATIVE_SIZE, "by_B": {}}
    for B in BS:
        k = f"B={B}"
        circuits = ["broadcast", "packed"] + (["interpolation"] if B == 5 else [])
        R = {"circuits": {}, "comparisons": {}}
        for c in circuits:
            per_seed = [runs[s]["by_B"][k][c] for s in SEEDS]
            m = np.array([r["mean_seconds_per_stay"] for r in per_seed])
            R["circuits"][c] = {
                "mean_seconds_per_seed": [float(x) for x in m],
                "sd_within_seed_seconds": [float(r["sd_seconds_per_stay"]) for r in per_seed],
                "mean_seconds": {"mean": float(m.mean()), "sd": float(m.std(ddof=1))},
                "client_upload_mb_per_seed": [r["client_upload_bytes"] / 1e6 for r in per_seed],
                "client_upload_mb": float(np.mean([r["client_upload_bytes"] for r in per_seed]) / 1e6),
                "max_abs_margin_error_per_seed": [float(r["max_abs_margin_error"]) for r in per_seed],
                "max_abs_margin_error_over_seeds": float(max(r["max_abs_margin_error"] for r in per_seed)),
                "meets_gate_all_seeds": bool(all(r["all_within_gate"] for r in per_seed)),
                "decisions_agreeing_f1_total": int(sum(r["decisions_agreeing"] for r in per_seed)),
                "decisions_agreeing_sens80_total": int(sum(r["decisions_agreeing_sens80"] for r in per_seed)),
                "n_stays_total": int(sum(r["n_stays"] for r in per_seed)),
                "degree": per_seed[0].get("degree"),
            }
        m = {c: np.array(R["circuits"][c]["mean_seconds_per_seed"]) for c in circuits}
        R["comparisons"]["packed_vs_broadcast"] = {"A": "packed", "B": "broadcast", "result": rule(m["packed"], m["broadcast"]),
                                                  "mean_difference_seconds_B_minus_A": float((m["broadcast"] - m["packed"]).mean())}
        if B == 5:
            R["comparisons"]["packed_vs_interpolation"] = {"A": "packed", "B": "interpolation", "result": rule(m["packed"], m["interpolation"]),
                                                          "mean_difference_seconds_B_minus_A": float((m["interpolation"] - m["packed"]).mean())}
            R["comparisons"]["broadcast_vs_interpolation"] = {"A": "broadcast", "B": "interpolation", "result": rule(m["broadcast"], m["interpolation"]),
                                                             "mean_difference_seconds_B_minus_A": float((m["interpolation"] - m["broadcast"]).mean())}
        R["reference_timing_seconds_before_after_per_seed"] = [[runs[s]["by_B"][k]["reference_timing_seconds_before"], runs[s]["by_B"][k]["reference_timing_seconds_after"]] for s in SEEDS]
        R["sample_counts_ok"] = bool(all(runs[s]["by_B"][k]["sample_counts"]["positives_first_then_negatives"] for s in SEEDS))
        if gaps:
            R["margin_gaps"] = {str(s): gaps["by_seed"][str(s)][k] for s in SEEDS}
            R["smallest_gap_sample_over_seeds"] = float(min(min(gaps["by_seed"][str(s)][k][t]["min_gap_sample_100"] for t in ("f1", "sens80")) for s in SEEDS))
            R["smallest_gap_full_split_over_seeds"] = float(min(min(gaps["by_seed"][str(s)][k][t]["min_gap_full_test_split"] for t in ("f1", "sens80")) for s in SEEDS))
        out["by_B"][k] = R
        for name, c in R["comparisons"].items():
            print(f"B={B} {name}: {c['result']} (mean difference B - A {c['mean_difference_seconds_B_minus_A']:.3f} s)")
    out["max_abs_margin_error_over_everything"] = float(max(v["max_abs_margin_error_over_seeds"] for R in out["by_B"].values() for v in R["circuits"].values()))
    out["peak_rss_mb_per_seed"] = [runs[s]["peak_rss_mb"] for s in SEEDS]
    path = os.path.join(fc.RESULTS, "m1c_thermo_samechain_summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"written -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
