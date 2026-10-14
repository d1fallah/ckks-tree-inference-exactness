"""
M1 -- Verification of the degree-sweep reproduction (closure guide 3.4). A REPRODUCTION, not a new protocol.

`src/eval_ckks.py` was re-run unchanged (only its output path was redirected, via M1_EVAL_CKKS_OUT) against the committed data,
coefficients and thresholds; the ORIGINAL committed result (results/m1_ckks_sweep_simulated.json) is never touched. This script
compares the two, checks that the run really used the intended anchor / coefficient version / differencing order, and records the
hashes and commit needed to say exactly what was run.

Outputs results/m1_sweep_reproduction_check.json (SIMULATED circuit-faithful backend throughout).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda *p: os.path.join(_HERE, "..", *p)  # noqa: E731
ORIGINAL = R("results", "m1_ckks_sweep_simulated.json")
REPRO = R("results", "m1_ckks_sweep_simulated_repro.json")
OUT = R("results", "m1_sweep_reproduction_check.json")

# Reference values from the supervisor's closure guide (3.4), 4 decimal places.
GUIDE_REFERENCE = {"k=3": (0.0016, 0.0010, 0.0025), "k=5": (0.0019, None, None), "k=7": (0.0022, None, None), "k=9": (0.0026, None, None)}
EXACT_TOL = 1e-12        # bit-for-bit up to floating-point serialization
MINOR_TOL = 5e-4         # tiny bootstrap-level differences are acceptable per the guide; material differences stop the line


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def sha256_dir(pattern_dir: str) -> str:
    h = hashlib.sha256()
    for name in sorted(os.listdir(pattern_dir)):
        if name.endswith(".json"):
            with open(os.path.join(pattern_dir, name), "rb") as f:
                h.update(name.encode() + f.read())
    return h.hexdigest()


def git(*args) -> str:
    try:
        return subprocess.check_output(["git", "-C", R(), *args], text=True).strip()
    except Exception:
        return "unavailable"


def main() -> int:
    with open(ORIGINAL, "r", encoding="utf-8") as f:
        orig = json.load(f)
    with open(REPRO, "r", encoding="utf-8") as f:
        rep = json.load(f)
    with open(R("data", "poly_coeffs_v1.json"), "r", encoding="utf-8") as f:
        coeffs = json.load(f)

    checks = {}
    # (a) the anchor / coefficient version the scripts actually use
    checks["coefficient_file_version"] = coeffs.get("version")
    checks["gamma_anchor_percentile_in_coefficient_file"] = coeffs.get("gamma_anchor_percentile")
    checks["gamma_in_coefficient_file"] = coeffs.get("gamma")
    sel_path = R("results", "m1_gamma_anchor_selection.json")
    if os.path.exists(sel_path):
        with open(sel_path, "r", encoding="utf-8") as f:
            sel = json.load(f)
        checks["selection_artifact_selected_percentile"] = sel["selected_percentile"]
        checks["coefficient_file_matches_selection_artifact"] = bool(coeffs["gamma_anchor_percentile"] == sel["selected_percentile"])

    # (b) per-seed differencing order: delta is |f1_plain - f1_ckks| per seed, headline is the MEAN of per-seed deltas
    order_ok = True
    for k, v in rep["k_sweep"].items():
        deltas = [abs(r["f1_plain"] - r["f1_ckks"]) for r in v["per_seed"]]
        order_ok &= bool(np.allclose(deltas, [r["delta_f1"] for r in v["per_seed"]], atol=1e-15))
        order_ok &= bool(abs(np.mean(deltas) - v["delta_f1_mean_over_seeds"]) < 1e-15)
    checks["per_seed_differencing_then_averaging"] = order_ok

    # (c) comparison of the reproduction to the original
    comparison, worst = {}, 0.0
    for k in orig["k_sweep"]:
        o, r = orig["k_sweep"][k], rep["k_sweep"][k]
        d_mean = abs(o["delta_f1_mean_over_seeds"] - r["delta_f1_mean_over_seeds"])
        d_lo = abs(o["paired_bootstrap"]["ci_95_lower"] - r["paired_bootstrap"]["ci_95_lower"])
        d_hi = abs(o["paired_bootstrap"]["ci_95_upper"] - r["paired_bootstrap"]["ci_95_upper"])
        d_seed = max(abs(a["delta_f1"] - b["delta_f1"]) for a, b in zip(o["per_seed"], r["per_seed"]))
        worst = max(worst, d_mean, d_lo, d_hi, d_seed)
        signed = [b["f1_ckks"] - b["f1_plain"] for b in r["per_seed"]]
        comparison[k] = {
            "original": {"delta_f1_mean": o["delta_f1_mean_over_seeds"], "ci": [o["paired_bootstrap"]["ci_95_lower"], o["paired_bootstrap"]["ci_95_upper"]]},
            "reproduction": {"delta_f1_mean": r["delta_f1_mean_over_seeds"], "ci": [r["paired_bootstrap"]["ci_95_lower"], r["paired_bootstrap"]["ci_95_upper"]]},
            "abs_diff_mean": d_mean, "abs_diff_ci_lower": d_lo, "abs_diff_ci_upper": d_hi, "max_abs_diff_per_seed_delta": d_seed,
            "matches_guide_reference_at_4dp": bool(round(r["delta_f1_mean_over_seeds"], 4) == GUIDE_REFERENCE[k][0]),
            "signed_delta_f1_per_seed_ckks_minus_plain": signed,
            "all_seeds_ckks_above_plain": bool(all(s > 0 for s in signed)),
        }
    all_above = all(c["all_seeds_ckks_above_plain"] for c in comparison.values())
    verdict = "IDENTICAL" if worst <= EXACT_TOL else ("WITHIN TOLERANCE (bootstrap-level)" if worst <= MINOR_TOL else "MATERIAL DIFFERENCE - STOP AND DIAGNOSE")

    out = {
        "protocol": "closure guide 3.4 (verification, not a new protocol); originals never overwritten",
        "backend": "SIMULATED circuit-faithful numpy evaluator",
        "provenance": {
            "git_head_commit": git("rev-parse", "HEAD"), "git_working_tree_clean_for_src": git("status", "--porcelain", "--", "src") == "",
            "code_sha256": {n: sha256(R("src", n)) for n in ("eval_ckks.py", "ckks_circuit.py", "eval_metrics.py", "splits.py", "notation.py")},
            "data_sha256": {"poly_coeffs_v1.json": sha256(R("data", "poly_coeffs_v1.json")),
                            "m1_cohort.parquet": sha256(R("data", "m1_cohort.parquet")),
                            "m1_features.parquet": sha256(R("data", "m1_features.parquet")),
                            "frozen_models_25_json": sha256_dir(R("data", "models"))},
            "results_inputs_sha256": {"m1_plaintext_baseline.json": sha256(R("results", "m1_plaintext_baseline.json")),
                                      "m1_circuit_params.json": sha256(R("results", "m1_circuit_params.json"))},
            "original_result_sha256": sha256(ORIGINAL), "reproduction_result_sha256": sha256(REPRO),
        },
        "intended_configuration_checks": checks,
        "comparison": comparison,
        "worst_absolute_difference_anywhere": worst,
        "verdict": verdict,
        "sign_disclosure": (
            "|dF1| is the locked statistic and hides direction. CKKS macro-F1 is above plaintext macro-F1 in all 5 seeds at every k: "
            + str(all_above) + ". If true, the reported utility 'loss' is a small utility GAIN at the macro level "
            "(docs/leakage_and_threshold_audit.md, section D)."),
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"verdict: {verdict}  (worst abs difference {worst:.3e})")
    for k, c in comparison.items():
        print(f"  {k}: original {c['original']['delta_f1_mean']:.6f} vs reproduction {c['reproduction']['delta_f1_mean']:.6f}   "
              f"guide 4dp match: {c['matches_guide_reference_at_4dp']}   CKKS above plain in all seeds: {c['all_seeds_ckks_above_plain']}")
    print(f"intended-configuration checks: {checks}")
    print(f"written -> {OUT}")
    return 0 if verdict != "MATERIAL DIFFERENCE - STOP AND DIAGNOSE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
