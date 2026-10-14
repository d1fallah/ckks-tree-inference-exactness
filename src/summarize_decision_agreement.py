#!/usr/bin/env python3
"""
Combine the five per-seed files written by eval_decision_agreement.py into the aggregate result for added
analysis S2.1 (docs/step2_prespec_m1.md, section B), and cross-check it against the stored k-sweep.

Per (label, k): per-seed disagreement rates, their mean and SD over seeds (training variability), the
patient-cluster bootstrap 95% interval of the seed-mean rate (sampling uncertainty; the same resampled
subjects in every seed), flip counts, probability-space error, and AUROC difference. Also the mean over the
five labels per k. Reported separately, never merged (CLAUDE.md 1.4.1).

Consistency gate: AUROC_plain and AUROC_ckks must equal the values stored by eval_ckks.py for k = 3..9 (the
same margins), which shows this re-run reproduced the reported simulator output.

Output: results/m1_decision_agreement.json
"""
from __future__ import annotations

import json
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(_HERE, "..", "results")
SEEDS = (42, 123, 7, 456, 789)


def main() -> int:
    per_seed = {}
    for s in SEEDS:
        with open(os.path.join(RES, f"m1_decision_agreement_seed{s}.json"), "r", encoding="utf-8") as f:
            per_seed[s] = json.load(f)
        assert per_seed[s]["smoke_test_rows"] is None, f"seed {s} file is a smoke test"
    with open(os.path.join(RES, "m1_ckks_sweep_simulated.json"), "r", encoding="utf-8") as f:
        stored = json.load(f)
    stored_idx = {(r["seed"], r["label"]): r for r in stored["per_label_detail"]}

    cells = {}
    max_auroc_gap = 0.0
    for s in SEEDS:
        for c in per_seed[s]["cells"]:
            cells.setdefault((c["label"], c["k"]), {})[s] = c
            if c["k"] != 1:
                ref = stored_idx[(s, c["label"])][f"k={c['k']}"]
                max_auroc_gap = max(max_auroc_gap, abs(ref["auroc_plain"] - c["auroc_plain"]), abs(ref["auroc_ckks"] - c["auroc_ckks"]))
    assert max_auroc_gap < 1e-9, f"re-run AUROC differs from the stored sweep by {max_auroc_gap}"

    labels = sorted({k[0] for k in cells})
    degrees = sorted({k[1] for k in cells})
    n_test = per_seed[SEEDS[0]]["n_test_stays"]
    out_cells = []
    for label in labels:
        for k in degrees:
            by_seed = cells[(label, k)]
            rates = np.array([by_seed[s]["disagree_rate"] for s in SEEDS])
            boot = np.mean([by_seed[s]["boot_disagree_rates"] for s in SEEDS], axis=0)
            out_cells.append({
                "label": label, "k": k,
                "disagree_rate_per_seed": {str(s): float(by_seed[s]["disagree_rate"]) for s in SEEDS},
                "disagree_rate_mean": float(rates.mean()), "disagree_rate_sd_across_seeds": float(rates.std(ddof=1)),
                "disagree_rate_cluster_bootstrap_ci95_of_seed_mean": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                "disagree_stays_per_1000_mean": float(1000 * rates.mean()),
                "mean_flips_plain_pos_to_ckks_neg": float(np.mean([by_seed[s]["n_plain_pos_to_ckks_neg"] for s in SEEDS])),
                "mean_flips_plain_neg_to_ckks_pos": float(np.mean([by_seed[s]["n_plain_neg_to_ckks_pos"] for s in SEEDS])),
                "prob_abs_error_mean_over_seeds": float(np.mean([by_seed[s]["prob_abs_error_mean"] for s in SEEDS])),
                "prob_abs_error_max_over_seeds": float(np.max([by_seed[s]["prob_abs_error_max"] for s in SEEDS])),
                "auroc_diff_plain_minus_ckks_mean": float(np.mean([by_seed[s]["auroc_plain"] - by_seed[s]["auroc_ckks"] for s in SEEDS])),
                "auroc_diff_plain_minus_ckks_sd": float(np.std([by_seed[s]["auroc_plain"] - by_seed[s]["auroc_ckks"] for s in SEEDS], ddof=1)),
            })
    by_k = {}
    for k in degrees:
        per_label_rates = np.array([[cells[(lab, k)][s]["disagree_rate"] for s in SEEDS] for lab in labels])  # (labels, seeds)
        boot_all = np.mean([[cells[(lab, k)][s]["boot_disagree_rates"] for s in SEEDS] for lab in labels], axis=(0, 1))
        seed_means = per_label_rates.mean(axis=0)
        by_k[f"k={k}"] = {
            "mean_over_labels_and_seeds": float(per_label_rates.mean()),
            "sd_across_seeds_of_label_mean": float(seed_means.std(ddof=1)),
            "cluster_bootstrap_ci95_of_label_seed_mean": [float(np.percentile(boot_all, 2.5)), float(np.percentile(boot_all, 97.5))],
            "max_cell_rate": float(per_label_rates.max()),
            "stays_per_1000_mean": float(1000 * per_label_rates.mean()),
            "prob_abs_error_mean_over_labels_and_seeds": float(np.mean([[cells[(lab, k)][s]["prob_abs_error_mean"] for s in SEEDS] for lab in labels])),
            "prob_abs_error_max_over_all_cells": float(np.max([[cells[(lab, k)][s]["prob_abs_error_max"] for s in SEEDS] for lab in labels])),
            "auroc_diff_plain_minus_ckks_mean_over_labels_and_seeds": float(np.mean([[cells[(lab, k)][s]["auroc_plain"] - cells[(lab, k)][s]["auroc_ckks"] for s in SEEDS] for lab in labels])),
            "auroc_diff_abs_max_over_all_cells": float(np.max([[abs(cells[(lab, k)][s]["auroc_plain"] - cells[(lab, k)][s]["auroc_ckks"]) for s in SEEDS] for lab in labels])),
        }
    out = {
        "protocol": "docs/step2_prespec_m1.md section B (added analysis S2.1)",
        "backend": "SIMULATED (circuit-faithful NumPy), plaintext vs simulated CKKS at the frozen per-(seed, label) margin thresholds",
        "n_test_stays": n_test, "seeds": list(SEEDS), "degrees": degrees, "labels": labels,
        "uncertainty": {"training_variability": "SD across the five seeds",
                        "sampling": "patient-cluster bootstrap, 1,000 resamples, seed 42, same subjects in every seed; 95% percentile interval of the seed-mean rate"},
        "consistency_gate": {"max_abs_auroc_difference_vs_stored_sweep": max_auroc_gap, "tolerance": 1e-9, "passed": True},
        "cells": out_cells, "by_k": by_k,
        "total_runtime_seconds": float(sum(per_seed[s]["runtime_seconds"] for s in SEEDS)),
    }
    with open(os.path.join(RES, "m1_decision_agreement.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    for k in degrees:
        r = by_k[f"k={k}"]
        print(f"k={k}: mean disagreement {r['mean_over_labels_and_seeds']:.5f} ({r['stays_per_1000_mean']:.2f} per 1000 stays) "
              f"SD over seeds {r['sd_across_seeds_of_label_mean']:.5f}  bootstrap 95% CI {r['cluster_bootstrap_ci95_of_label_seed_mean']}  max cell {r['max_cell_rate']:.5f}")
    print(f"AUROC consistency vs stored sweep: max gap {max_auroc_gap:.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
