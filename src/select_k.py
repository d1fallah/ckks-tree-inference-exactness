#!/usr/bin/env python3
"""
The primary research contribution's selection rule, per M1-0 Sec 0's
recorded supervisor override of the roadmap's original formulation:

    ROADMAP (superseded):  maximize F1(k)  s.t. L(k) <= L_max, DeltaF1(k) <= 0.03
    GUIDE v4.1 (governing): minimize L(k)  s.t. DeltaF1(k) <= 0.03

This retires L_max entirely (m1-0-implementation-lock.md Sec 0) -- there is
no latency ceiling to calibrate. The feasible set is simply every k whose
measured DeltaF1 (results/m1_ckks_sweep_simulated.json) is within the frozen
0.03 predictive-utility tolerance; k* is the CHEAPEST (lowest measured
latency) feasible k. If latency has not yet been measured for a given
k (real-encrypted latency comes from eval_ckks_encrypted.py, not the
simulation), this reports the feasible set using DeltaF1 alone and flags
latency as pending rather than guessing an ordering.

Latency source (closure log, 2026-09-21): the MULTI-SEED real-ciphertext arm
(results/m1_encrypted_multiseed.json: 5 seeds x 25 frozen test patients, one label,
REAL TenSEAL/SEAL). The earlier single-seed arm (results/m1_encrypted_subsample.json,
one model, n=30) fed the first selection and is preserved unchanged; the selection it
produced is kept in results/m1_k_selection_v1_singleseed_latency.json. k=9 has no real
latency because its code-derived depth (8) exceeds the 7 levels on the locked chain
(m1-0-implementation-lock.md sec 18); it is simulation-only, so it is excluded from the
latency-minimising choice ON THAT GROUND, stated in the output, not silently dropped by a
missing value. Its utility row (simulated) is still reported.

Usage: python select_k.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from notation import K_APPROX_DEGREE_VALUES, MAX_ACCEPTABLE_F1_LOSS  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP_PATH = os.path.join(_HERE, "..", "results", "m1_ckks_sweep_simulated.json")
LATENCY_PATH = os.path.join(_HERE, "..", "results", "m1_encrypted_multiseed.json")
OUT_PATH = os.path.join(_HERE, "..", "results", "m1_k_selection.json")


def main() -> int:
    with open(SWEEP_PATH, "r", encoding="utf-8") as f:
        sweep = json.load(f)

    latency_by_k, latency_sd_by_k, per_seed_latency, k_not_executable = {}, {}, {}, {}
    if os.path.exists(LATENCY_PATH):
        with open(LATENCY_PATH, "r", encoding="utf-8") as f:
            enc = json.load(f)
        for k_key, block in enc["pooled_across_seeds"].items():
            lat = block["latency_wall_clock_s_mean"]  # mean over seeds of the per-seed wall-clock mean (25 patients each)
            latency_by_k[k_key] = lat["mean"]
            latency_sd_by_k[k_key] = lat["sd_across_seeds"]
        for cell in enc["per_cell"]:
            per_seed_latency.setdefault(f"k={cell['k']}", {})[cell["seed"]] = cell["latency_wall_clock_s_mean"]
        k_not_executable[f"k={enc['k_excluded']}"] = enc["k_excluded_reason"]

    feasible = []
    infeasible = []
    for k in K_APPROX_DEGREE_VALUES:
        row = sweep["k_sweep"][f"k={k}"]
        delta = row["delta_f1_mean_over_seeds"]
        ci = row["paired_bootstrap"]
        entry = {
            "k": k,
            "delta_f1_mean": delta,
            "delta_f1_95ci": [ci["ci_95_lower"], ci["ci_95_upper"]],
            "online_latency_seconds_per_patient": latency_by_k.get(f"k={k}"),
            "latency_sd_across_seeds": latency_sd_by_k.get(f"k={k}"),
            "latency_not_measured_reason": k_not_executable.get(f"k={k}"),
        }
        if delta <= MAX_ACCEPTABLE_F1_LOSS:
            feasible.append(entry)
        else:
            infeasible.append(entry)

    # Is the latency ordering a property of the mean or of every seed? k* per seed, on the same feasible set.
    feasible_keys = [f"k={e['k']}" for e in feasible if e["online_latency_seconds_per_patient"] is not None]
    seeds = sorted({sd for k_key in feasible_keys for sd in per_seed_latency.get(k_key, {})})
    k_star_per_seed = {
        str(sd): int(min(feasible_keys, key=lambda kk: per_seed_latency[kk][sd]).split("=")[1]) for sd in seeds
    } if feasible_keys else {}

    result = {
        "rule": "minimize L(k) s.t. DeltaF1(k) <= 0.03 (M1-0 Sec 0, supervisor override recorded)",
        "utility_tolerance": MAX_ACCEPTABLE_F1_LOSS,
        "feasible_set": feasible,
        "infeasible_set": infeasible,
        "backend_for_delta_f1": sweep.get("backend", "unknown"),
        "latency_source": {
            "file": os.path.basename(LATENCY_PATH),
            "backend": "REAL TenSEAL/SEAL ciphertext execution",
            "design": "5 seeds x 25 frozen test patients per k, one label (hypertension); value = mean over seeds of per-seed mean wall-clock; SD is across seeds",
            "k_without_real_latency": k_not_executable,
        },
        "k_star_per_seed_by_measured_latency": k_star_per_seed,
        "note_on_selection": ("The utility constraint is slack for every k in the locked family (and at k=1, "
                              "docs/k1_compatibility.md): k* is the cheapest executable degree, i.e. the lower bound of the "
                              "family, not a feasibility boundary."),
    }

    if not feasible:
        result["k_star"] = None
        result["outcome"] = "EMPTY FEASIBLE SET -- no k in {3,5,7,9} meets DeltaF1 <= 0.03. Reported plainly, not relaxed after the fact."
    else:
        have_latency = [e for e in feasible if e["online_latency_seconds_per_patient"] is not None]
        if have_latency:
            k_star = min(have_latency, key=lambda e: e["online_latency_seconds_per_patient"])
            result["k_star"] = k_star["k"]
            result["outcome"] = (f"k*={k_star['k']} selected: cheapest measured-latency k among the feasible set "
                                 f"(k with no real latency are listed in latency_source and cannot be selected on latency).")
        else:
            k_star = min(feasible, key=lambda e: e["k"])
            result["k_star"] = k_star["k"]
            result["outcome"] = (
                f"Feasible set is non-empty but real encrypted latency has not been measured yet "
                f"(run eval_ckks_encrypted.py first) -- reporting the smallest feasible k ({k_star['k']}) "
                f"as a provisional k*, NOT a latency-minimizing choice. Re-run this script after "
                f"{os.path.basename(LATENCY_PATH)} exists."
            )

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    print(f"\nwritten -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
