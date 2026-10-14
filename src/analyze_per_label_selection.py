"""
M1 -- Does the selected degree ever change with the label? (supervisor brief, decision D4; closure log Entry 12 open item.)

Reads only results/m1_ckks_sweep_simulated.json (per_label_detail: 5 seeds x 5 labels x k in {3,5,7,9}, SIMULATED utility, full test split) and
results/m1_encrypted_multiseed.json (real latency, hypertension label only). Applies the locked rule to EACH label separately:

    feasible_k(label)  = { k : mean over seeds of |dF1(label, k)| <= 0.03 }        (per-seed |dF1| first, then mean: the locked statistic)
    k*(label)          = the feasible k with the lowest real latency

Latency is only measured for the hypertension label. The selection therefore uses the measured ordering of latency over k (k=3 < 5 < 7 in every
seed) and ASSUMES it holds for the other labels. That is a stated assumption, not a measurement: the circuit's depth, and so the level structure that
drives latency, is set by k and the tree depth, not by the label, but per-label latency was not run. k = 9 has no real latency (depth 8 > 7 levels)
and cannot be selected on latency.

This is a per-label view stricter than the locked macro criterion; it is supplementary and changes no locked protocol. Output: results/m1_per_label_selection.json
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
OUT_PATH = os.path.join(RES, "m1_per_label_selection.json")


def main() -> int:
    with open(os.path.join(RES, "m1_ckks_sweep_simulated.json"), encoding="utf-8") as f:
        detail = json.load(f)["per_label_detail"]
    with open(os.path.join(RES, "m1_encrypted_multiseed.json"), encoding="utf-8") as f:
        enc = json.load(f)
    tol = MAX_ACCEPTABLE_F1_LOSS
    executable = {int(k.split("=")[1]): v["latency_wall_clock_s_mean"]["mean"] for k, v in enc["pooled_across_seeds"].items()}
    per_seed_lat = {}
    for c in enc["per_cell"]:
        per_seed_lat.setdefault(c["seed"], {})[c["k"]] = c["latency_wall_clock_s_mean"]
    ordering_holds_every_seed = all(lat[3] < lat[5] < lat[7] for lat in per_seed_lat.values())
    assert ordering_holds_every_seed, "latency ordering k=3<5<7 does not hold in every seed; the per-label selection assumption fails"

    labels = sorted({r["label"] for r in detail})
    out_labels, changed = {}, []
    for lab in labels:
        rows = [r for r in detail if r["label"] == lab]
        assert len(rows) == 5, (lab, len(rows))
        by_k, feasible = {}, []
        for k in K_APPROX_DEGREE_VALUES:
            absd = [r[f"k={k}"]["delta_f1"] for r in rows]
            signed = [r[f"k={k}"]["f1_ckks"] - r[f"k={k}"]["f1_plain"] for r in rows]
            mean = statistics.mean(absd)
            by_k[f"k={k}"] = {"abs_delta_f1_mean": mean, "abs_delta_f1_sd_across_seeds": statistics.stdev(absd), "abs_delta_f1_max_seed": max(absd),
                              "signed_delta_f1_mean_ckks_minus_plain": statistics.mean(signed), "n_seeds_ckks_above_plain": sum(s > 0 for s in signed),
                              "fraction_of_tolerance_mean": mean / tol, "meets_tolerance": mean <= tol}
            if mean <= tol:
                feasible.append(k)
        selectable = [k for k in feasible if k in executable]
        k_star = min(selectable, key=lambda k: executable[k]) if selectable else None
        out_labels[lab] = {"n_leaves_seed42_model": rows[0]["n_leaves"], "by_k": by_k, "feasible_k": feasible, "k_star": k_star}
        if k_star != 3:
            changed.append(lab)

    all_cells = [(lab, k, r[f"k={k}"]["delta_f1"]) for lab in labels for r in detail if r["label"] == lab for k in K_APPROX_DEGREE_VALUES]
    worst = max(all_cells, key=lambda t: t[2])
    result = {
        "status": "supplementary per-label view of the locked rule; SIMULATED utility; latency measured for hypertension only",
        "utility_tolerance": tol,
        "assumption": ("per-label latency ordering k=3<5<7 is assumed equal to the measured hypertension ordering (holds in each of the 5 seeds there); "
                       "per-label latency was not measured"),
        "latency_ordering_holds_every_seed_hypertension": ordering_holds_every_seed,
        "real_latency_s_by_k": executable,
        "labels": out_labels,
        "k_star_by_label": {lab: v["k_star"] for lab, v in out_labels.items()},
        "labels_where_k_star_differs_from_3": changed,
        "n_label_seed_k_cells": len(all_cells),
        "n_cells_at_or_above_tolerance": sum(c[2] >= tol for c in all_cells),
        "largest_cell": {"label": worst[0], "k": worst[1], "abs_delta_f1": worst[2], "fraction_of_tolerance": worst[2] / tol},
        "largest_label_mean": max(((lab, k, v["by_k"][f"k={k}"]["abs_delta_f1_mean"]) for lab, v in out_labels.items() for k in K_APPROX_DEGREE_VALUES),
                                  key=lambda t: t[2]),
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print("k* by label:", result["k_star_by_label"])
    print("labels where k* != 3:", changed or "none")
    print(f"cells at or above tolerance: {result['n_cells_at_or_above_tolerance']}/{result['n_label_seed_k_cells']}; largest cell {result['largest_cell']}")
    print("largest label mean:", result["largest_label_mean"])
    print(f"written -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
