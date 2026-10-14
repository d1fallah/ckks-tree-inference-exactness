#!/usr/bin/env python3
"""
Second validity pilot for the M1 continuous-feature task (added analysis). Executes
docs/m1c_pilot2_prespec.md, committed (b1704a7) before this file existed. Validation split only; the test
split is never selected. Simulated circuit. Aggregate output only: results/m1c_pilot2.json.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score, roc_auc_score

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import ckks_circuit as cc  # noqa: E402
import m1c_cohort as mc  # noqa: E402
from derive_circuit_params import pooled_interval  # noqa: E402
from generate_poly_coefficients import SIGMOID_EDGE_LEVEL  # noqa: E402
from m1c_pilot_validity import LIMITS, fit_model, simulate  # noqa: E402
from select_gamma_anchor import fit_power_coefficients  # noqa: E402
from splits import patient_level_split  # noqa: E402
from train_plaintext import tune_thresholds  # noqa: E402

OUT = os.path.join(_HERE, "..", "results", "m1c_pilot2.json")
PILOT1 = os.path.join(_HERE, "..", "results", "m1c_pilot_validity.json")
SEED = 42
GRID = [5, 10, 25, 50, 75, 90, 95, 99.9]
DEGREES = (3, 5, 7, 9)
SCORE_K = 5
SCORE_FRACTION = 0.25
TIE_TOLERANCE = 0.002
N_BOOT, BOOT_SEED = 1000, 42
REFERENCE_MAX_PER_1000 = 50.0


def pooled_abs_z(clf, Xtr: pd.DataFrame, norm: dict) -> tuple[np.ndarray, dict]:
    """|normalized x - normalized t| over the model's splits and the same 4,000 training rows that
    derive_circuit_params.pooled_interval draws (same generator, same loop). Returns the pooled array and the
    per-feature 99.9th percentile. Checked against pooled_interval below."""
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(Xtr), size=min(4000, len(Xtr)), replace=False)
    Xs = Xtr.iloc[idx]
    df = clf.get_booster().trees_to_dataframe()
    internal = df[df["Feature"] != "Leaf"]
    by_feat: dict[str, list[float]] = {}
    for feat, thr in zip(internal["Feature"], internal["Split"]):
        by_feat.setdefault(feat, []).append(float(thr))
    pooled, per_feat = [], {}
    for feat, thrs in by_feat.items():
        lo, hi = norm[feat]["min"], norm[feat]["max"]
        span = (hi - lo) if hi > lo else 1.0
        x = ((Xs[feat].to_numpy(dtype=float) - lo) / span)[:, None]
        t = ((np.asarray(thrs) - lo) / span)[None, :]
        d = np.abs((x - t).ravel())
        pooled.append(d)
        per_feat[feat] = float(np.percentile(d, 99.9))
    return np.concatenate(pooled), per_feat


def main() -> int:
    t_start = time.perf_counter()
    df, _ = mc.build_cohort()
    train_ids, val_ids, test_ids = patient_level_split(df)
    tr = df[df["subject_id"].isin(train_ids)].reset_index(drop=True)
    va = df[df["subject_id"].isin(val_ids)].reset_index(drop=True)
    assert set(va["subject_id"]).isdisjoint(test_ids) and set(tr["subject_id"]).isdisjoint(test_ids)
    del df
    feats = mc.FEATURE_COLS
    med = {c: float(tr[c].median()) for c in mc.VITAL_FEATURES}
    Xtr, Xva = tr[feats].copy(), va[feats].copy()
    for c in mc.VITAL_FEATURES:
        Xtr[c] = Xtr[c].fillna(med[c])
        Xva[c] = Xva[c].fillna(med[c])
    ytr, yva = tr[mc.OUTCOME_COL].to_numpy(dtype=int), va[mc.OUTCOME_COL].to_numpy(dtype=int)

    clf = fit_model(Xtr, ytr)
    booster = clf.get_booster()
    plain = booster.predict(xgb.DMatrix(Xva, feature_names=feats), output_margin=True)
    auc_full = float(roc_auc_score(yva, plain))
    with open(PILOT1, "r", encoding="utf-8") as f:
        p1 = json.load(f)
    assert abs(auc_full - p1["positive_control"]["auroc_full_val"]) < 1e-9, "model differs from the first pilot"
    proba = 1.0 / (1.0 + np.exp(-plain))
    thr = tune_thresholds(yva.reshape(-1, 1), proba.reshape(-1, 1))[0]
    thr_margin = float(np.log(thr / (1.0 - thr)))
    assert abs(thr_margin - p1["positive_control"]["tuned_threshold_margin"]) < 1e-12
    pp = (plain >= thr_margin).astype(int)
    print(f"model reproduced: AUROC {auc_full:.4f}, threshold margin {thr_margin:.4f}", flush=True)

    structure = cc.extract_ensemble_leaf_circuit(booster, set(feats), tree_depth=6)
    Xva_mat = Xva.to_numpy(dtype=np.float64)
    codes, uniques = pd.factorize(va["subject_id"])
    n_subj = len(uniques)
    rng = np.random.default_rng(BOOT_SEED)
    w = rng.multinomial(n_subj, np.full(n_subj, 1.0 / n_subj), size=N_BOOT).astype(np.int64)
    n_boot_stays = w @ np.bincount(codes, minlength=n_subj).astype(np.int64)

    # fixed scoring subsample: 25% of validation patients
    srng = np.random.default_rng(SEED)
    keep_subj = srng.choice(n_subj, size=int(round(SCORE_FRACTION * n_subj)), replace=False)
    sub = np.isin(codes, keep_subj)
    Xsub, plain_sub, pp_sub = Xva_mat[sub], plain[sub], pp[sub]
    span_plain = float(plain_sub.max() - plain_sub.min()) or 1.0

    norm_B = {c: {"min": LIMITS[c][0], "max": LIMITS[c][1]} for c in feats}
    # equivalence check of the local pooled computation against derive_circuit_params.pooled_interval
    arr_B, perfeat_B = pooled_abs_z(clf, Xtr, norm_B)
    ref = pooled_interval({"pilot": clf}, Xtr, norm_B, set(), seed=SEED)
    for q in (50, 75, 90, 95, 99.9):
        assert abs(float(np.percentile(arr_B, q)) - ref["abs_diff_quantiles"][q]) < 1e-5, f"pooled mismatch p{q}"
    # per-feature scaling: divide each feature's difference by its own p99.9, then a common interval of 1
    norm_F = {c: {"min": LIMITS[c][0], "max": LIMITS[c][0] + (LIMITS[c][1] - LIMITS[c][0]) * perfeat_B[c]}
              for c in perfeat_B}
    for c in feats:
        norm_F.setdefault(c, norm_B[c])
    arr_F, _ = pooled_abs_z(clf, Xtr, norm_F)
    intervals = {
        "pooled": {"norm": norm_B, "a": float(ref["a_half_width"]), "abs_z": arr_B},
        "per_feature": {"norm": norm_F, "a": 1.0, "abs_z": arr_F},
    }

    cache: dict = {}

    def sweep_cell(iname: str, P, gamma: float, k: int) -> dict:
        key = (iname, P, k)
        if key in cache:
            return cache[key]
        spec = intervals[iname]
        bl = cc.prepare_batch_levels(structure, feats, spec["norm"], spec["a"])
        ck = cc.assemble_coefficient_vectors(structure, fit_power_coefficients(gamma, spec["a"], k), k)
        sim = simulate(bl, ck, structure.leaf_value, Xva_mat, spec["a"])
        pc = (sim >= thr_margin).astype(int)
        diff = pp != pc
        per_subj = np.bincount(codes[diff], minlength=n_subj).astype(np.int64)
        boot = (w @ per_subj) / n_boot_stays
        cell = {
            "disagree_per_1000": float(1000 * diff.mean()),
            "disagree_boot95_per_1000": [float(1000 * np.percentile(boot, 2.5)), float(1000 * np.percentile(boot, 97.5))],
            "n_plain_pos_to_ckks_neg": int(((pp == 1) & (pc == 0)).sum()),
            "n_plain_neg_to_ckks_pos": int(((pp == 0) & (pc == 1)).sum()),
            "f1_ckks_val": float(f1_score(yva, pc, zero_division=0)),
            "auroc_ckks_val": float(roc_auc_score(yva, sim)),
            "auroc_delta_plain_minus_ckks": float(auc_full - roc_auc_score(yva, sim)),
        }
        cache[key] = cell
        return cell

    out_cfg = {}
    for iname, spec in intervals.items():
        t0 = time.perf_counter()
        bl_sub = cc.prepare_batch_levels(structure, feats, spec["norm"], spec["a"])
        cands = []
        for P in GRID:
            z_ref = float(np.percentile(spec["abs_z"], P))
            gamma = float(np.log(SIGMOID_EDGE_LEVEL / (1 - SIGMOID_EDGE_LEVEL)) / z_ref)
            c5 = cc.assemble_coefficient_vectors(structure, fit_power_coefficients(gamma, spec["a"], SCORE_K), SCORE_K)
            sim5 = simulate(bl_sub, c5, structure.leaf_value, Xsub, spec["a"])
            cands.append({"percentile": P, "z_ref": z_ref, "gamma": gamma,
                          "sign_agreement": float(np.mean(np.sign(plain_sub) == np.sign(sim5))),
                          "decision_agreement": float(np.mean(pp_sub == (sim5 >= thr_margin).astype(int))),
                          "nmae": float(np.mean(np.abs(sim5 - plain_sub)) / span_plain)})
            print(f"  {iname} p{P}: gamma={gamma:.1f} sign={cands[-1]['sign_agreement']:.3f} "
                  f"decision={cands[-1]['decision_agreement']:.3f}", flush=True)
        for rule, metric in (("old_sign_at_0", "sign_agreement"), ("threshold_decision", "decision_agreement")):
            top = max(r[metric] for r in cands)
            tied = [r for r in cands if top - r[metric] <= TIE_TOLERANCE]
            win = sorted(tied, key=lambda r: (r["nmae"], -r["gamma"]))[0]
            cells = {f"k={k}": sweep_cell(iname, win["percentile"], win["gamma"], k) for k in DEGREES}
            out_cfg[f"{iname}|{rule}"] = {
                "interval": iname, "rule": rule, "selected_percentile": win["percentile"], "gamma": win["gamma"],
                "tied_percentiles": [r["percentile"] for r in tied], "on_grid_edge_low": win["percentile"] == GRID[0],
                "a": spec["a"], "cells": cells,
            }
            print(f"{iname}|{rule}: p{win['percentile']} " +
                  " ".join(f"k{k}={cells[f'k={k}']['disagree_per_1000']:.1f}" for k in DEGREES), flush=True)
        out_cfg[f"{iname}|candidates"] = cands
        print(f"{iname} done in {time.perf_counter() - t0:.0f}s", flush=True)

    best_k9 = min((v["cells"]["k=9"]["disagree_per_1000"], n) for n, v in out_cfg.items() if "cells" in v)
    out = {
        "protocol": "docs/m1c_pilot2_prespec.md (commit b1704a7)", "backend": "SIMULATED (circuit-faithful NumPy)",
        "scope": "seed 42, validation split only, test split not read; anchor scored on a fixed 25% of validation patients",
        "grid": GRID, "degrees": list(DEGREES), "model_auroc_val": auc_full,
        "n_val_stays": int(len(va)), "n_scoring_subsample_stays": int(sub.sum()),
        "configs": out_cfg,
        "best_config_at_k9": {"name": best_k9[1], "disagree_per_1000": best_k9[0]},
        "decision_rule_reference_per_1000": REFERENCE_MAX_PER_1000,
        "best_meets_reference_at_k9": bool(best_k9[0] <= REFERENCE_MAX_PER_1000),
        "runtime_seconds": time.perf_counter() - t_start,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"best at k=9: {best_k9[1]} {best_k9[0]:.1f} per 1000 (reference <= {REFERENCE_MAX_PER_1000}); "
          f"written -> {OUT} ({out['runtime_seconds']:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
