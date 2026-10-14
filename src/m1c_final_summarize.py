#!/usr/bin/env python3
"""
Collapse results/m1c_final_results.json (per seed) into results/m1c_final_summary.json: mean and SD over the five seeds for each
quantity the manuscript reports, plus the per-seed values. Reads stored results only; no data. The paper assets read this summary
through paper_data.Evidence.
"""
from __future__ import annotations

import json
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
IN = os.path.join(_HERE, "..", "results", "m1c_final_results.json")
OUT = os.path.join(_HERE, "..", "results", "m1c_final_summary.json")


def ms(vals) -> dict:
    v = np.asarray(vals, dtype=float)
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "per_seed": [float(x) for x in v]}


def main() -> int:
    with open(IN, "r", encoding="utf-8") as f:
        d = json.load(f)
    seeds = [str(s) for s in d["seeds"]]
    ps = d["per_seed"]

    def col(fn):
        return [fn(ps[s]) for s in seeds]

    def plain_block(key, util_key):
        g = lambda s: ps[s][key][util_key]  # noqa: E731
        return {
            "auroc": ms(col(lambda s: s[key][util_key]["auroc"])),
            "auprc": ms(col(lambda s: s[key][util_key]["auprc"])),
            "f1op_f1": ms(col(lambda s: s[key][util_key]["at_f1_threshold"]["f1"])),
            "f1op_sensitivity": ms(col(lambda s: s[key][util_key]["at_f1_threshold"]["sensitivity"])),
            "f1op_specificity": ms(col(lambda s: s[key][util_key]["at_f1_threshold"]["specificity"])),
            "f1op_ppv": ms(col(lambda s: s[key][util_key]["at_f1_threshold"]["ppv"])),
            "sens80_sensitivity": ms(col(lambda s: s[key][util_key]["at_sens80_threshold"]["sensitivity"])),
            "sens80_specificity": ms(col(lambda s: s[key][util_key]["at_sens80_threshold"]["specificity"])),
            "sens80_ppv": ms(col(lambda s: s[key][util_key]["at_sens80_threshold"]["ppv"])),
        }

    def exact_block(key):
        out = {"degree": ps[seeds[0]][key]["degree"], "bins": ps[seeds[0]][key]["B"],
               "circuit_auroc": ms(col(lambda s: s[key]["circuit_auroc_test"]))}
        for op in ("f1", "sens80"):
            c = lambda s, op=op: s[key]["circuit_vs_plain"][op]  # noqa: E731
            out[op] = {"n_differing": ms(col(lambda s: c(s)["n_differing"])),
                       "differing_per_1000": ms(col(lambda s: c(s)["differing_per_1000"])),
                       "plain_positive_decisions": ms(col(lambda s: c(s)["plain_positive_decisions"])),
                       "lost_share_of_plain_positives": ms(col(lambda s: c(s)["lost_share_of_plain_positives"])),
                       "max_abs_margin_difference": ms(col(lambda s: c(s)["max_abs_margin_difference"])),
                       "n_differing_max_over_seeds": int(max(c(ps[s])["n_differing"] for s in seeds))}
        return out

    floor = {}
    for k in ps[seeds[0]]["floor_reference"].keys():
        fl = lambda s, k=k: s["floor_reference"][k]  # noqa: E731
        floor[k] = {"circuit_auroc": ms(col(lambda s: fl(s)["circuit_auroc_test"])),
                    "auroc_delta": ms(col(lambda s: fl(s)["auroc_delta_plain_minus_circuit"])),
                    "selected_percentile_per_seed": [fl(ps[s])["selected_percentile"] for s in seeds]}
        for op in ("f1", "sens80"):
            floor[k][op] = {
                "differing_per_1000": ms(col(lambda s: fl(s)["vs_plain"][op]["differing_per_1000"])),
                "lost_share_of_plain_positives": ms(col(lambda s: fl(s)["vs_plain"][op]["lost_share_of_plain_positives"])),
                "plain_neg_to_circuit_pos": ms(col(lambda s: fl(s)["vs_plain"][op]["plain_neg_to_circuit_pos"])),
                "plain_positive_decisions": ms(col(lambda s: fl(s)["vs_plain"][op]["plain_positive_decisions"]))}

    qc = lambda s: s["quantization_cost"]  # noqa: E731
    out = {
        "source": "results/m1c_final_results.json", "seeds": d["seeds"], "n_seeds": len(seeds), "inputs": d["inputs"],
        "n_test_stays": d["n_test_stays"], "n_test_subjects": d["n_test_subjects"], "cohort_flow": d["cohort_flow"],
        "use_acuity": d["acuity_decision"]["use_acuity"], "acuity_auroc_gain_validation": d["acuity_decision"]["auroc_gain"],
        "bootstrap": d["bootstrap"],
        "exact_in_simulation_all_seeds_both_operating_points": d["exact_in_simulation_all_seeds_both_operating_points"],
        "plaintext_full_precision": plain_block("full_precision", "utility_test"),
        "plaintext_quantized_B5": plain_block("primary_exact_B5", "utility_plain_test"),
        "plaintext_quantized_B8": plain_block("sensitivity_exact_B8", "utility_plain_test"),
        "exact_B5": exact_block("primary_exact_B5"), "exact_B8": exact_block("sensitivity_exact_B8"),
        "floor": floor,
        "quantization_cost": {
            "auroc": ms(col(lambda s: qc(s)["auroc_cost"])),
            "auroc_boot95_per_seed": [qc(ps[s])["auroc_cost_boot95"] for s in seeds],
            "f1_full_precision": ms(col(lambda s: qc(s)["f1_full_precision"])),
            "f1_quantized": ms(col(lambda s: qc(s)["f1_quantized"])),
            "f1_cost_boot95_per_seed": [qc(ps[s])["f1_cost_boot95"] for s in seeds]},
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"written -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
