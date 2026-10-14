#!/usr/bin/env python3
"""
Thermometer final run, stage 5 (docs/m1c_thermo_final_prespec.md): aggregates the per-seed result files over the five seeds (mean and SD over seeds; the
patient-cluster bootstrap intervals stay in the per-seed files and are reported separately). Reads aggregate JSON only.
Output: results/m1c_thermo_final_summary.json.
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


def load(stage, seed):
    with open(tc.out_path(stage, seed, False), "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    sim = {s: load("sim", s) for s in fc.SEEDS}
    real = {s: load("real", s) for s in fc.SEEDS}
    near = {s: load("near", s) for s in fc.SEEDS}
    with open(tc.out_path("locked", None, False), "r", encoding="utf-8") as f:
        locked = json.load(f)

    def over(get):
        return fc.ms([get(s) for s in fc.SEEDS])
    out = {"protocol": "Analysis-plan gate, M1 model-private thermometer protocol (GATES.md)", "seeds": list(fc.SEEDS), "by_B": {}}
    out["full_precision"] = {"auroc_test": over(lambda s: sim[s]["full_precision"]["utility_test"]["auroc"]),
                             "f1_f1_op": over(lambda s: sim[s]["full_precision"]["utility_test"]["at_f1_threshold"]["f1"])}
    for B in tc.LEVELS_B:
        k = f"B={B}"
        g = lambda s, *p: _walk(sim[s]["by_B"][k], p)  # noqa: E731
        O = {"quantized_plain": {
                "auroc_test": over(lambda s: g(s, "utility_plain_test", "auroc")),
                "auprc_test": over(lambda s: g(s, "utility_plain_test", "auprc")),
                "f1_f1_op": over(lambda s: g(s, "utility_plain_test", "at_f1_threshold", "f1")),
                "sensitivity_sens80_op": over(lambda s: g(s, "utility_plain_test", "at_sens80_threshold", "sensitivity")),
                "specificity_sens80_op": over(lambda s: g(s, "utility_plain_test", "at_sens80_threshold", "specificity")),
                "ppv_sens80_op": over(lambda s: g(s, "utility_plain_test", "at_sens80_threshold", "ppv")),
                "auroc_cost_vs_full_precision": over(lambda s: g(s, "quantization_cost", "auroc_cost")),
                "auroc_cost_boot95_per_seed": [g(s, "quantization_cost", "auroc_cost_boot95") for s in fc.SEEDS]},
             "thermometer_simulated": {
                "n_differing_f1_op_per_seed": [g(s, "thermometer_sim_vs_booster", "f1", "n_differing") for s in fc.SEEDS],
                "n_differing_sens80_op_per_seed": [g(s, "thermometer_sim_vs_booster", "sens80", "n_differing") for s in fc.SEEDS],
                "max_abs_margin_difference_over_seeds": float(max(g(s, "thermometer_sim_vs_booster", "f1", "max_abs_margin_difference") for s in fc.SEEDS)),
                "exact_in_simulation_all_seeds_both_ops": bool(all(g(s, "thermometer_sim_vs_booster", n, "n_differing") == 0 for s in fc.SEEDS for n in ("f1", "sens80"))
                                                                 and max(g(s, "thermometer_sim_vs_booster", "f1", "max_abs_margin_difference") for s in fc.SEEDS) <= 1e-5),
                "stays_near_either_threshold_per_seed": [g(s, "stays_near_either_threshold") for s in fc.SEEDS]},
             "real": {}, "near_threshold": {}}
        if B == 5:
            O["interpolation_simulated"] = {
                "n_differing_f1_op_per_seed": [g(s, "interpolation_sim_vs_booster", "f1", "n_differing") for s in fc.SEEDS],
                "n_differing_sens80_op_per_seed": [g(s, "interpolation_sim_vs_booster", "sens80", "n_differing") for s in fc.SEEDS]}
        for circuit in real[fc.SEEDS[0]]["by_B"][k]:
            if circuit == "B":
                continue
            cs = [real[s]["by_B"][k][circuit] for s in fc.SEEDS]
            O["real"][circuit] = {
                "n_stays_total": int(sum(c["n_stays"] for c in cs)),
                "max_abs_margin_error": float(max(c["max_abs_margin_error"] for c in cs)),
                "decisions_agreeing_f1_op": int(sum(c["decisions_agreeing"] for c in cs)),
                "decisions_agreeing_sens80_op": int(sum(c["decisions_agreeing_sens80"] for c in cs)),
                "all_seeds_pass_circuit_gate": bool(all(c["passes_circuit_gate"] for c in cs)),
                "min_abs_gap_sampled": float(min(min(c["min_abs_gap_to_f1_threshold_sampled"], c["min_abs_gap_to_sens80_threshold_sampled"]) for c in cs)),
                "seconds_per_stay": fc.ms([c["mean_seconds_per_stay"] for c in cs]),
                "client_upload_bytes": int(cs[0]["client_upload_bytes"]),
                "client_upload_mb": cs[0]["client_upload_bytes"] / 1e6,
                "chain_levels": len(real[fc.SEEDS[0]]["chains"]["locked_seven_level" if "7level" in circuit else "four_level"]) - 2}
        for name in ("f1", "sens80"):
            ns = [near[s]["by_B"].get(k, {}).get(name, {}).get("n_stays_within_gate", 0) for s in fc.SEEDS]
            fl = [near[s]["by_B"].get(k, {}).get(name, {}).get("n_flipped_in_at_least_one_repetition", 0) for s in fc.SEEDS]
            sm = [near[s]["by_B"].get(k, {}).get(name, {}).get("n_simulated_decision_differs_from_booster", 0) for s in fc.SEEDS]
            O["near_threshold"][name] = {"n_stays_within_gate_per_seed": ns, "n_flipped_per_seed": fl, "n_simulated_differing_per_seed": sm,
                                         "total_stays": int(sum(ns)), "total_flipped": int(sum(fl)),
                                         "repetitions_per_seed": [near[s]["repetitions"] for s in fc.SEEDS],
                                         "flip_rates_seed42": near[42]["by_B"].get(k, {}).get(name, {}).get("flip_rates_sorted", [])}
        O["near_threshold"]["max_abs_margin_error_over_repetitions"] = float(max(near[s]["by_B"].get(k, {}).get("max_abs_margin_error", 0.0) for s in fc.SEEDS))
        out["by_B"][k] = O
    kb = real[fc.SEEDS[0]]["key_bytes_one_off"]
    for v in kb.values():
        v.update({k + "_mb": v[k] / 1e6 for k in list(v)})
    out["key_bytes_one_off"] = kb
    out["chains"] = real[fc.SEEDS[0]]["chains"]
    out["chain_levels"] = {k: len(v) - 2 for k, v in out["chains"].items()}
    out["n_seeds"] = len(fc.SEEDS)
    out["n_real_stays_per_seed_per_B"] = real[fc.SEEDS[0]]["by_B"]["B=5"]["broadcast_thermometer_4level"]["n_stays"]
    out["n_test_stays"] = sim[fc.SEEDS[0]]["n_test_stays"]
    out["n_test_subjects"] = sim[fc.SEEDS[0]]["n_test_subjects"]
    out["source"] = "m1c_thermo_final_*.json (Analysis-plan gate, M1 model-private thermometer protocol)"
    out["locked_chain_levels"] = len(locked["chain"]) - 2
    lk = locked["key_bytes_one_off"]
    lk.update({k + "_mb": lk[k] / 1e6 for k in list(lk)})
    for R in locked["by_B"].values():
        for v in R.values():
            if isinstance(v, dict) and "client_upload_bytes" in v:
                v["client_upload_mb"] = v["client_upload_bytes"] / 1e6
    out["locked_chain_arm"] = {"seed": 42, "key_bytes_one_off": lk, "by_B": locked["by_B"], "sample": locked["sample"]}
    with open(os.path.join(fc.RESULTS, "m1c_thermo_final_summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("written -> results/m1c_thermo_final_summary.json")
    return 0


def _walk(d, path):
    for p in path:
        d = d[p]
    return d


if __name__ == "__main__":
    raise SystemExit(main())
