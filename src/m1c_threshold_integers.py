#!/usr/bin/env python3
r"""
Added check (2026-10-13, author's approval of point 5b of a pasted review): are the split thresholds of the saved thermometer-study boosters integers, and how small is the smallest?
Reads only the saved booster files of the study (data/final/thermo_s{seed}_B{B}.json: split thresholds and tree structure, no patient data); prints and stores counts only.
Why it matters for Lemma 1, Remark 1: the code uses c = ceil(t - 1e-9); for an integer threshold t = k >= 1 this equals ceil(t) = k, so the tolerance changes no decision, and the
case k = 0 does not occur if the smallest threshold is at least 1.
Output: results/m1c_threshold_integers.json.
"""
from __future__ import annotations

import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "final")
OUT = os.path.join(HERE, "..", "results", "m1c_threshold_integers.json")
SEEDS, BINS = (42, 123, 7, 456, 789), (5, 8, 16)


def main() -> int:
    n_boosters = n_splits = n_nonint = n_tol_differs = 0
    t_min = math.inf
    for seed in SEEDS:
        for B in BINS:
            with open(os.path.join(DATA, f"thermo_s{seed}_B{B}.json"), "r", encoding="utf-8") as f:
                d = json.load(f)
            n_boosters += 1
            for tree in d["learner"]["gradient_booster"]["model"]["trees"]:
                for t, left in zip(tree["split_conditions"], tree["left_children"]):
                    if left == -1:  # a leaf
                        continue
                    n_splits += 1
                    t_min = min(t_min, t)
                    n_nonint += int(t != math.floor(t))
                    n_tol_differs += int(math.ceil(t - 1e-9) != math.ceil(t))
    out = {"n_boosters": n_boosters, "n_splits": n_splits, "n_non_integer_thresholds": n_nonint, "smallest_threshold": t_min,
           "n_splits_where_ceil_with_tolerance_differs_from_ceil": n_tol_differs,
           "note": "thresholds read from the saved boosters of the thermometer study; counts only"}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
