"""
M1 -- Descriptive checks behind docs/framing_brief_for_supervisor.md (no new experiment, no new model, no CKKS).

Reads only committed result artifacts and computes, in one place, every number the framing brief quotes that is not already
stored verbatim in an artifact:

  1. How much of the utility tolerance is used: the largest |dF1| in the k sweep, at k=1, and in the scaling grid, against 0.03.
  2. Whether the single-seed scaling pilot's pattern survives the 5-seed run, as PAIRED per-seed differences (the same seed and
     the same test split under both configurations), never as a difference of seed-averaged values:
       - tree count: 5 vs 50 trees at depth 3
       - depth: 3 vs 6 at 50 trees
     reported per seed, as a mean and SD of the paired difference, and as the number of seeds in the same direction with the exact
     one-sided sign-test probability under "either direction equally likely".
  3. How extreme the pilot's seed-42 5-tree cell is among the 40 scaling cells.

STATUS OF THESE STATISTICS: exploratory and post hoc. Nothing here was pre-registered, five seeds is the whole sample, and the
sign test is uncorrected for the two contrasts examined. They describe the evidence; they are not a significance claim, and the
protocol's locked statistic (|dF1|, per-seed then mean) is unchanged.

Output: results/m1_framing_evidence.json
"""
from __future__ import annotations

import json
import os
import statistics
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from notation import K_APPROX_DEGREE_VALUES, MAX_ACCEPTABLE_F1_LOSS  # noqa: E402

RES = os.path.join(_HERE, "..", "results")
OUT_PATH = os.path.join(RES, "m1_framing_evidence.json")


def load(name: str) -> dict:
    with open(os.path.join(RES, name), "r", encoding="utf-8") as f:
        return json.load(f)


def paired_contrast(cells: list[dict], seeds: list[int], a: tuple[int, int], b: tuple[int, int], key: str) -> dict:
    """Per-seed paired difference cell(a) - cell(b) of `key`, where a and b are (n_estimators, max_depth)."""
    def value(seed: int, cfg: tuple[int, int]) -> float:
        hit = [c for c in cells if c["seed"] == seed and (c["n_estimators"], c["max_depth"]) == cfg]
        assert len(hit) == 1, (seed, cfg, len(hit))
        return hit[0][key]

    diffs = [value(s, a) - value(s, b) for s in seeds]
    n_pos = sum(d > 0 for d in diffs)
    n_neg = sum(d < 0 for d in diffs)
    n = len(diffs)
    same_direction = max(n_pos, n_neg)
    # exact one-sided sign-test probability of >= same_direction seeds in one pre-named direction under p = 1/2
    from math import comb
    p_one_sided = sum(comb(n, i) for i in range(same_direction, n + 1)) / 2 ** n
    return {
        "contrast": f"{key}: (trees={a[0]}, depth={a[1]}) minus (trees={b[0]}, depth={b[1]})",
        "per_seed_values_a": {str(s): value(s, a) for s in seeds},
        "per_seed_values_b": {str(s): value(s, b) for s in seeds},
        "per_seed_difference": {str(s): d for s, d in zip(seeds, diffs)},
        "mean_paired_difference": statistics.mean(diffs),
        "sd_paired_difference": statistics.stdev(diffs),
        "seeds_a_greater_than_b": n_pos,
        "seeds_a_less_than_b": n_neg,
        "n_seeds": n,
        "exact_one_sided_sign_test_p_if_direction_named_in_advance": p_one_sided,
        "median_paired_difference": statistics.median(diffs),
        "largest_single_seed_share_of_summed_difference": max(abs(d) for d in diffs) / sum(abs(d) for d in diffs),
    }


def main() -> int:
    sweep = load("m1_ckks_sweep_simulated.json")
    k1 = load("m1_ckks_sweep_k1_simulated.json")
    scal = load("m1_scaling_multiseed.json")
    cells, seeds = scal["per_cell"], scal["seeds"]
    tol = MAX_ACCEPTABLE_F1_LOSS

    sweep_means = {f"k={k}": sweep["k_sweep"][f"k={k}"]["delta_f1_mean_over_seeds"] for k in K_APPROX_DEGREE_VALUES}
    sweep_per_seed_max = max(r["delta_f1"] for k in K_APPROX_DEGREE_VALUES for r in sweep["k_sweep"][f"k={k}"]["per_seed"])
    k1_mean = k1["delta_f1_mean_over_seeds"]
    scaling_cell_max = max(c["abs_delta_f1"] for c in cells)
    worst = max(cells, key=lambda c: c["abs_delta_f1"])
    five_tree = sorted(c["abs_delta_f1"] for c in cells if (c["n_estimators"], c["max_depth"]) == (5, 3))
    others_5 = sorted(c["abs_delta_f1"] for c in cells if (c["n_estimators"], c["max_depth"]) == (5, 3) and c["seed"] != 42)

    out = {
        "status": "descriptive, exploratory, post hoc; not pre-registered; 5 seeds; sign test uncorrected for the two contrasts",
        "source_artifacts": ["m1_ckks_sweep_simulated.json", "m1_ckks_sweep_k1_simulated.json", "m1_scaling_multiseed.json"],
        "utility_tolerance": tol,
        "utility_headroom": {
            "k_sweep_delta_f1_mean_by_k": sweep_means,
            "k_sweep_largest_mean": max(sweep_means.values()),
            "k_sweep_largest_mean_as_fraction_of_tolerance": max(sweep_means.values()) / tol,
            "k_sweep_largest_per_seed_value": sweep_per_seed_max,
            "k1_delta_f1_mean": k1_mean,
            "k1_as_fraction_of_tolerance": k1_mean / tol,
            "scaling_grid_largest_cell": scaling_cell_max,
            "scaling_grid_largest_cell_as_fraction_of_tolerance": scaling_cell_max / tol,
            "scaling_grid_largest_cell_is": {"seed": worst["seed"], "n_estimators": worst["n_estimators"], "max_depth": worst["max_depth"]},
            "n_scaling_cells": len(cells),
            "n_scaling_cells_at_or_above_tolerance": sum(c["abs_delta_f1"] >= tol for c in cells),
        },
        "pilot_cell_extremeness": {
            "pilot_seed42_5tree_depth3": worst["abs_delta_f1"] if (worst["seed"], worst["n_estimators"], worst["max_depth"]) == (42, 5, 3) else None,
            "five_tree_depth3_values_all_seeds_sorted": five_tree,
            "five_tree_depth3_values_excluding_seed42_sorted": others_5,
            "pilot_value_over_largest_other_seed_value": (
                next(c["abs_delta_f1"] for c in cells if c["seed"] == 42 and (c["n_estimators"], c["max_depth"]) == (5, 3)) / max(others_5)),
        },
        "paired_scaling_checks": {
            "tree_count_5_vs_50_at_depth3_abs_delta_f1": paired_contrast(cells, seeds, (5, 3), (50, 3), "abs_delta_f1"),
            "depth_6_vs_3_at_50_trees_abs_delta_f1": paired_contrast(cells, seeds, (50, 6), (50, 3), "abs_delta_f1"),
        },
        "read_this_as": ("Direction and size are separate questions. The direction of the tree-count effect may hold in every seed while "
                         "its size is a fraction of what the single-seed pilot showed; the smallest one-sided sign-test probability 5 seeds can give is 1/32 = 0.031, "
                         "so it is not a substitute for the locked bootstrap."),
    }

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: out[k] for k in ("utility_headroom", "pilot_cell_extremeness")}, indent=2))
    for name, c in out["paired_scaling_checks"].items():
        print(name, "mean", round(c["mean_paired_difference"], 5), "sd", round(c["sd_paired_difference"], 5),
              f"a>b in {c['seeds_a_greater_than_b']}/{c['n_seeds']} seeds, sign-test p1={c['exact_one_sided_sign_test_p_if_direction_named_in_advance']:.4f}")
    print(f"written -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
