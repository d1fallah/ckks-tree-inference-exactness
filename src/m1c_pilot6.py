#!/usr/bin/env python3
"""
Sixth pilot for the M1 continuous-feature task (added analysis): input quantization. Executes
docs/m1c_pilot6_prespec.md (committed before this file existed). Train and validation splits only; the test split is
never selected. Simulated circuit. Aggregate output only: results/m1c_pilot6.json.
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from numpy.polynomial import chebyshev as C
from sklearn.metrics import f1_score, roc_auc_score

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import ckks_circuit as cc  # noqa: E402
import m1c_cohort as mc  # noqa: E402
from generate_poly_coefficients import SIGMOID_EDGE_LEVEL, soft_step  # noqa: E402
from m1c_pilot_validity import fit_model  # noqa: E402
from splits import patient_level_split  # noqa: E402
from train_plaintext import tune_thresholds  # noqa: E402

OUT = os.path.join(_HERE, "..", "results", "m1c_pilot6.json")
PILOT1 = os.path.join(_HERE, "..", "results", "m1c_pilot_validity.json")
LEVELS = (4, 8, 16)
DEGREES = (3, 5, 7, 9, 15)
GAMMA_GRID = [1, 5, 25]
SEED = 42
SCORE_FRACTION = 0.25
TIE_TOLERANCE = 0.002
N_BOOT, BOOT_SEED = 1000, 42
FID_PER_1000, FID_LOST_POS = 50.0, 0.10
CHUNK = 4000


def shifted_threshold(t: float) -> float:
    return t - 0.5 if abs(t - round(t)) <= 1e-6 else t


def pooled_abs_z_q(clf, Xtr: pd.DataFrame, norm: dict, shift: bool) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(Xtr), size=min(4000, len(Xtr)), replace=False)
    Xs = Xtr.iloc[idx]
    df = clf.get_booster().trees_to_dataframe()
    internal = df[df["Feature"] != "Leaf"]
    by_feat: dict[str, list[float]] = {}
    for feat, thr in zip(internal["Feature"], internal["Split"]):
        t = float(thr)
        by_feat.setdefault(feat, []).append(shifted_threshold(t) if shift else t)
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
    Xtr_raw, Xva_raw = tr[feats].copy(), va[feats].copy()
    for c in mc.VITAL_FEATURES:
        Xtr_raw[c] = Xtr_raw[c].fillna(med[c])
        Xva_raw[c] = Xva_raw[c].fillna(med[c])
    ytr, yva = tr[mc.OUTCOME_COL].to_numpy(dtype=int), va[mc.OUTCOME_COL].to_numpy(dtype=int)
    with open(PILOT1, "r", encoding="utf-8") as f:
        p1 = json.load(f)
    auc_full_precision = p1["positive_control"]["auroc_full_val"]
    f1_full_precision = p1["positive_control"]["f1_plain_val"]
    codes, uniques = pd.factorize(va["subject_id"])
    n_subj = len(uniques)
    rng = np.random.default_rng(BOOT_SEED)
    w = rng.multinomial(n_subj, np.full(n_subj, 1.0 / n_subj), size=N_BOOT).astype(np.int64)
    n_boot_stays = w @ np.bincount(codes, minlength=n_subj).astype(np.int64)
    srng = np.random.default_rng(SEED)
    keep_subj = srng.choice(n_subj, size=int(round(SCORE_FRACTION * n_subj)), replace=False)
    sub = np.isin(codes, keep_subj)

    out_levels = {}
    for b in LEVELS:
        t_b = time.perf_counter()
        # ---- public quantile-bin table from the training split
        edges = {}
        for c in feats:
            e = np.unique(np.quantile(Xtr_raw[c].to_numpy(dtype=float), np.arange(1, b) / b))
            edges[c] = e
        def binned(X: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame({c: np.searchsorted(edges[c], X[c].to_numpy(dtype=float), side="right").astype(float)
                                 for c in feats})
        Xtr, Xva = binned(Xtr_raw), binned(Xva_raw)
        nb = {c: int(Xtr[c].max()) + 1 for c in feats}
        # ---- positive control
        clf = fit_model(Xtr, ytr)
        booster = clf.get_booster()
        plain = booster.predict(xgb.DMatrix(Xva, feature_names=feats), output_margin=True)
        auc = float(roc_auc_score(yva, plain))
        clf_age = fit_model(Xtr[["age"]], ytr)
        auc_age = float(roc_auc_score(yva, clf_age.predict_proba(Xva[["age"]])[:, 1]))
        proba = 1.0 / (1.0 + np.exp(-plain))
        thr = tune_thresholds(yva.reshape(-1, 1), proba.reshape(-1, 1))[0]
        thr_margin = float(np.log(thr / (1.0 - thr)))
        pp = (plain >= thr_margin).astype(int)
        n_pos_plain = int(pp.sum())
        f1_plain = float(f1_score(yva, pp, zero_division=0))
        pc_info = {"auroc_val": auc, "auroc_age_only_val": auc_age, "P1": bool(auc >= 0.60),
                   "P2": bool(auc - auc_age >= 0.03), "auroc_cost_vs_full_precision": auc_full_precision - auc,
                   "f1_plain_val": f1_plain, "f1_cost_vs_full_precision": f1_full_precision - f1_plain,
                   "tuned_threshold": thr, "n_plain_positive_decisions": n_pos_plain,
                   "bins_per_feature": nb}
        print(f"b={b}: AUROC {auc:.4f} (age-only {auc_age:.4f}, cost {auc_full_precision - auc:.4f}); "
              f"F1 {f1_plain:.3f}; bins {nb}", flush=True)

        # ---- circuit structures
        struct_orig = cc.extract_ensemble_leaf_circuit(booster, set(feats), tree_depth=6)
        struct_shift = cc.TreeCircuitStructure(
            n_leaves=struct_orig.n_leaves, leaf_value=struct_orig.leaf_value,
            levels=[[dataclasses.replace(r, threshold=shifted_threshold(r.threshold)) if r.kind != "pad" else r
                     for r in lvl] for lvl in struct_orig.levels])
        yes_masks = [np.array([r.kind != "pad" and r.is_yes_edge for r in lvl]) for lvl in struct_orig.levels]
        pad_masks = [np.array([r.kind == "pad" for r in lvl]) for lvl in struct_orig.levels]
        norm_q = {c: {"min": 0.0, "max": float(max(nb[c] - 1, 1))} for c in feats}
        _, perfeat = pooled_abs_z_q(clf, Xtr, norm_q, True)
        norm_F = {c: {"min": 0.0, "max": norm_q[c]["max"] * perfeat[c]} for c in perfeat}
        for c in feats:
            norm_F.setdefault(c, norm_q[c])
        abs_z, _ = pooled_abs_z_q(clf, Xtr, norm_F, True)
        bl_orig = cc.prepare_batch_levels(struct_orig, feats, norm_F, 1.0)
        bl = cc.prepare_batch_levels(struct_shift, feats, norm_F, 1.0)
        Xva_mat = Xva.to_numpy(dtype=np.float64)
        Xsub, plain_sub, pp_sub = Xva_mat[sub], plain[sub], pp[sub]
        span_plain = float(plain_sub.max() - plain_sub.min()) or 1.0

        def margin(st, blv, hfun, X):
            o = np.empty(X.shape[0])
            for s in range(0, X.shape[0], CHUNK):
                Xc = X[s : s + CHUNK]
                prod = np.ones((Xc.shape[0], st.n_leaves))
                for lvl, yes, pad in zip(blv, yes_masks, pad_masks):
                    z = np.clip((Xc[:, lvl.feature_col_idx] - lvl.threshold[None, :]) / lvl.span[None, :], -1.0, 1.0)
                    h = hfun(z)
                    f = np.where(yes[None, :], h, 1.0 - h)
                    prod *= np.where(pad[None, :], 1.0, f)
                o[s : s + CHUNK] = prod @ st.leaf_value
            return o

        hard = lambda z: (z < 0).astype(float)  # noqa: E731
        m_o = margin(struct_orig, bl_orig, hard, Xva_mat)
        m_s = margin(struct_shift, bl, hard, Xva_mat)
        assert np.max(np.abs(m_o - m_s)) <= 1e-9, f"b={b}: shifted thresholds change the exact-step circuit"
        assert np.max(np.abs(m_s - plain)) <= 1e-4, f"b={b}: exact-step circuit differs from the booster"
        assert np.array_equal(m_s >= thr_margin, plain >= thr_margin), f"b={b}: decisions differ from the booster"
        n_cmp = n_tie = 0
        for s in range(0, Xva_mat.shape[0], CHUNK):
            Xc = Xva_mat[s : s + CHUNK]
            for d, lvl in enumerate(bl_orig):
                valid = ~pad_masks[d]
                tie = (np.abs(Xc[:, lvl.feature_col_idx] - lvl.threshold[None, :]) <= 1e-9)[:, valid]
                n_cmp += tie.size
                n_tie += int(tie.sum())
        pc_info["exact_tie_share_original_thresholds"] = n_tie / n_cmp
        print(f"b={b}: lossless OK; exact ties {n_tie / n_cmp:.4f} of {n_cmp} comparisons", flush=True)

        cache: dict = {}

        def coef_for(P_: float, k: int):
            z_ref = float(np.percentile(abs_z, P_))
            with np.errstate(divide="ignore"):
                gamma = float(np.log(SIGMOID_EDGE_LEVEL / (1 - SIGMOID_EDGE_LEVEL)) / z_ref) if z_ref > 0 else float("inf")
            return C.Chebyshev.interpolate(lambda z: soft_step(z, gamma), deg=k, domain=[-1.0, 1.0]).coef, gamma

        def full_cell(P_: float, k: int) -> dict:
            if (P_, k) in cache:
                return cache[(P_, k)]
            coef, gamma = coef_for(P_, k)
            sim = margin(struct_shift, bl, lambda z: C.chebval(z, coef), Xva_mat)
            pc = (sim >= thr_margin).astype(int)
            diff = pp != pc
            per_subj = np.bincount(codes[diff], minlength=n_subj).astype(np.int64)
            boot = (w @ per_subj) / n_boot_stays
            lost = int(((pp == 1) & (pc == 0)).sum())
            gained = int(((pp == 0) & (pc == 1)).sum())
            pr = 1.0 / (1.0 + np.exp(-sim))
            own = tune_thresholds(yva.reshape(-1, 1), pr.reshape(-1, 1))[0]
            cell = {
                "selected_percentile": P_, "gamma": gamma,
                "disagree_per_1000": float(1000 * diff.mean()),
                "disagree_boot95_per_1000": [float(1000 * np.percentile(boot, 2.5)), float(1000 * np.percentile(boot, 97.5))],
                "n_plain_pos_to_ckks_neg": lost, "n_plain_neg_to_ckks_pos": gained,
                "lost_share_of_plain_positives": lost / n_pos_plain,
                "auroc_ckks_val": float(roc_auc_score(yva, sim)),
                "auroc_delta_plain_minus_ckks": float(auc - roc_auc_score(yva, sim)),
                "f1_plain_val": f1_plain,
                "f1_ckks_at_plain_threshold_val": float(f1_score(yva, pc, zero_division=0)),
                "f1_ckks_at_own_threshold_val": float(f1_score(yva, (pr >= own).astype(int), zero_division=0)),
            }
            cache[(P_, k)] = cell
            return cell

        cells = {}
        for k in DEGREES:
            t0 = time.perf_counter()
            cands = []
            for P_ in GAMMA_GRID:
                coef, gamma = coef_for(P_, k)
                sim = margin(struct_shift, bl, lambda z: C.chebval(z, coef), Xsub)
                cands.append({"percentile": P_, "gamma": gamma,
                              "decision_agreement": float(np.mean(pp_sub == (sim >= thr_margin).astype(int))),
                              "nmae": float(np.mean(np.abs(sim - plain_sub)) / span_plain)})
            top = max(r["decision_agreement"] for r in cands)
            tied = [r for r in cands if top - r["decision_agreement"] <= TIE_TOLERANCE]
            win = sorted(tied, key=lambda r: (r["nmae"], -r["gamma"]))[0]
            cell = dict(full_cell(win["percentile"], k))
            cell["scoring_candidates"] = cands
            cells[f"k={k}"] = cell
            print(f"b={b} k={k}: p{win['percentile']} {cell['disagree_per_1000']:.1f}/1000 "
                  f"{[round(x, 1) for x in cell['disagree_boot95_per_1000']]} lost {cell['lost_share_of_plain_positives']:.3f} "
                  f"AUROCd {cell['auroc_delta_plain_minus_ckks']:.4f} [{time.perf_counter() - t0:.0f}s]", flush=True)
        out_levels[f"b={b}"] = {"positive_control": pc_info, "cells": cells, "runtime_seconds": time.perf_counter() - t_b}
        with open(OUT.replace(".json", "_partial.json"), "w", encoding="utf-8") as f:
            json.dump(out_levels, f)

    meets = [{"b": b, "k": k, "meets_fidelity": bool(c["disagree_per_1000"] <= FID_PER_1000
                                                      and c["lost_share_of_plain_positives"] <= FID_LOST_POS),
              "auroc_within_0.03_of_full_precision": bool(v["positive_control"]["auroc_cost_vs_full_precision"] <= 0.03)}
             for b, v in out_levels.items() for k, c in v["cells"].items()]
    out = {
        "protocol": "docs/m1c_pilot6_prespec.md", "backend": "SIMULATED (circuit-faithful NumPy, Chebyshev-basis comparison)",
        "scope": "seed 42, train and validation splits only, test split not read; gamma scored on the fixed 25% subsample",
        "full_precision_reference": {"auroc_val": auc_full_precision, "f1_val": f1_full_precision,
                                     "pilot_k9_disagree_per_1000": 35.02},
        "n_val_stays": int(len(va)), "levels": out_levels,
        "fidelity_criteria": {"max_disagree_per_1000": FID_PER_1000, "max_lost_share_of_plain_positives": FID_LOST_POS},
        "meets_fidelity": meets, "runtime_seconds": time.perf_counter() - t_start,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"written -> {OUT} ({out['runtime_seconds']:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
