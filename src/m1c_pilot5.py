#!/usr/bin/env python3
"""
Fifth pilot for the M1 continuous-feature task (added analysis): exact ties between inputs and thresholds.
Executes docs/m1c_pilot5_prespec.md (committed before this file existed). Validation split only; the test split is
never selected. Simulated circuit. Aggregate output only: results/m1c_pilot5.json.
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
from m1c_pilot2 import pooled_abs_z  # noqa: E402
from m1c_pilot_validity import LIMITS, fit_model  # noqa: E402
from splits import patient_level_split  # noqa: E402
from train_plaintext import tune_thresholds  # noqa: E402

OUT = os.path.join(_HERE, "..", "results", "m1c_pilot5.json")
PILOT1 = os.path.join(_HERE, "..", "results", "m1c_pilot_validity.json")
PILOT4 = os.path.join(_HERE, "..", "results", "m1c_pilot4.json")
SEED = 42
DEGREES = (5, 7, 9, 11, 13, 15)
GAMMA_GRID = [1, 5, 25]
SCORE_FRACTION = 0.25
TIE_TOLERANCE = 0.002
N_BOOT, BOOT_SEED = 1000, 42
FID_PER_1000, FID_LOST_POS = 50.0, 0.10
CHUNK = 4000
LATTICE = {"triage_temperature": 1.0 / 18.0, "triage_heartrate": 1.0, "triage_resprate": 1.0, "triage_o2sat": 1.0,
           "triage_sbp": 1.0, "triage_dbp": 1.0, "triage_pain": 1.0, "age": 1.0}


def shifted_threshold(feature: str, t: float) -> float:
    s = LATTICE[feature]
    return t - s / 2.0 if abs(t / s - round(t / s)) <= 1e-6 else t


def pooled_abs_z_thr(clf, Xtr: pd.DataFrame, norm: dict, shift: bool) -> tuple[np.ndarray, dict]:
    """pooled_abs_z (m1c_pilot2), with the thresholds optionally shifted. Same 4,000 training rows, same loop."""
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(Xtr), size=min(4000, len(Xtr)), replace=False)
    Xs = Xtr.iloc[idx]
    df = clf.get_booster().trees_to_dataframe()
    internal = df[df["Feature"] != "Leaf"]
    by_feat: dict[str, list[float]] = {}
    for feat, thr in zip(internal["Feature"], internal["Split"]):
        t = float(thr)
        by_feat.setdefault(feat, []).append(shifted_threshold(feat, t) if shift else t)
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
    with open(PILOT4, "r", encoding="utf-8") as f:
        p4 = json.load(f)
    assert abs(auc_full - p1["positive_control"]["auroc_full_val"]) < 1e-9, "model differs from earlier pilots"
    proba = 1.0 / (1.0 + np.exp(-plain))
    thr = tune_thresholds(yva.reshape(-1, 1), proba.reshape(-1, 1))[0]
    thr_margin = float(np.log(thr / (1.0 - thr)))
    assert abs(thr_margin - p1["positive_control"]["tuned_threshold_margin"]) < 1e-12
    pp = (plain >= thr_margin).astype(int)
    n_pos_plain = int(pp.sum())
    f1_plain = float(f1_score(yva, pp, zero_division=0))

    struct_orig = cc.extract_ensemble_leaf_circuit(booster, set(feats), tree_depth=6)
    struct_shift = cc.TreeCircuitStructure(
        n_leaves=struct_orig.n_leaves, leaf_value=struct_orig.leaf_value,
        levels=[[dataclasses.replace(r, threshold=shifted_threshold(r.feature, r.threshold)) if r.kind != "pad" else r
                 for r in lvl] for lvl in struct_orig.levels])
    Xva_mat = Xva.to_numpy(dtype=np.float64)
    codes, uniques = pd.factorize(va["subject_id"])
    n_subj = len(uniques)
    rng = np.random.default_rng(BOOT_SEED)
    w = rng.multinomial(n_subj, np.full(n_subj, 1.0 / n_subj), size=N_BOOT).astype(np.int64)
    n_boot_stays = w @ np.bincount(codes, minlength=n_subj).astype(np.int64)
    srng = np.random.default_rng(SEED)
    keep_subj = srng.choice(n_subj, size=int(round(SCORE_FRACTION * n_subj)), replace=False)
    sub = np.isin(codes, keep_subj)
    Xsub, plain_sub, pp_sub = Xva_mat[sub], plain[sub], pp[sub]
    span_plain = float(plain_sub.max() - plain_sub.min()) or 1.0

    norm_B = {c: {"min": LIMITS[c][0], "max": LIMITS[c][1]} for c in feats}
    yes_masks = [np.array([r.kind != "pad" and r.is_yes_edge for r in lvl]) for lvl in struct_orig.levels]
    pad_masks = [np.array([r.kind == "pad" for r in lvl]) for lvl in struct_orig.levels]

    # equivalence: the generalized pooled computation reproduces m1c_pilot2.pooled_abs_z when nothing is shifted
    a0, pf0 = pooled_abs_z(clf, Xtr, norm_B)
    a1, pf1 = pooled_abs_z_thr(clf, Xtr, norm_B, shift=False)
    assert np.allclose(np.sort(a0), np.sort(a1)) and all(abs(pf0[c] - pf1[c]) < 1e-12 for c in pf0)

    variants = {}
    for name, shift, struct in (("original", False, struct_orig), ("shifted", True, struct_shift)):
        _, perfeat = pooled_abs_z_thr(clf, Xtr, norm_B, shift)
        norm_F = {c: {"min": LIMITS[c][0], "max": LIMITS[c][0] + (LIMITS[c][1] - LIMITS[c][0]) * perfeat[c]} for c in perfeat}
        for c in feats:
            norm_F.setdefault(c, norm_B[c])
        abs_z, _ = pooled_abs_z_thr(clf, Xtr, norm_F, shift)
        variants[name] = {"struct": struct, "bl": cc.prepare_batch_levels(struct, feats, norm_F, 1.0), "abs_z": abs_z}

    def margin(variant: dict, hfun, X: np.ndarray) -> np.ndarray:
        out = np.empty(X.shape[0])
        st, bl = variant["struct"], variant["bl"]
        for s in range(0, X.shape[0], CHUNK):
            Xc = X[s : s + CHUNK]
            prod = np.ones((Xc.shape[0], st.n_leaves))
            for lvl, yes, pad in zip(bl, yes_masks, pad_masks):
                z = np.clip((Xc[:, lvl.feature_col_idx] - lvl.threshold[None, :]) / lvl.span[None, :], -1.0, 1.0)
                h = hfun(z)
                f = np.where(yes[None, :], h, 1.0 - h)
                prod *= np.where(pad[None, :], 1.0, f)
            out[s : s + CHUNK] = prod @ st.leaf_value
        return out

    # ---- losslessness (hard assertions): exact step, original vs shifted thresholds, vs the plaintext booster
    hard = lambda z: (z < 0).astype(float)  # noqa: E731
    m_o = margin(variants["original"], hard, Xva_mat)
    m_s = margin(variants["shifted"], hard, Xva_mat)
    assert np.max(np.abs(m_o - m_s)) <= 1e-9, "shifted thresholds change the exact-step circuit"
    assert np.max(np.abs(m_s - plain)) <= 1e-4, "exact-step circuit differs from the plaintext booster"
    assert np.array_equal((m_s >= thr_margin), (plain >= thr_margin)), "decisions differ from the plaintext booster"
    print(f"lossless: max |exact-step margin, shifted - original| = {np.max(np.abs(m_o - m_s)):.1e}; "
          f"vs booster {np.max(np.abs(m_s - plain)):.1e}", flush=True)

    # ---- tie census (aggregate)
    n_cmp = n_tie = n_tie_shift = 0
    per_feat = {c: [0, 0] for c in feats}
    for s in range(0, Xva_mat.shape[0], CHUNK):
        Xc = Xva_mat[s : s + CHUNK]
        for d, lvl in enumerate(variants["original"]["bl"]):
            valid = ~pad_masks[d]
            tie = (np.abs(Xc[:, lvl.feature_col_idx] - lvl.threshold[None, :]) <= 1e-9)[:, valid]
            lv2 = variants["shifted"]["bl"][d]
            tie2 = (np.abs(Xc[:, lv2.feature_col_idx] - lv2.threshold[None, :]) <= 1e-9)[:, valid]
            n_cmp += tie.size
            n_tie += int(tie.sum())
            n_tie_shift += int(tie2.sum())
            fn = np.array([r.feature if r.kind != "pad" else "" for r in struct_orig.levels[d]])[valid]
            for c in feats:
                m = fn == c
                if m.any():
                    per_feat[c][0] += int(tie[:, m].size)
                    per_feat[c][1] += int(tie[:, m].sum())
    pairs = {(r.feature, r.threshold) for lvl in struct_orig.levels for r in lvl if r.kind != "pad"}
    on_lattice = sum(1 for f_, t_ in pairs if abs(t_ / LATTICE[f_] - round(t_ / LATTICE[f_])) <= 1e-6)
    census = {"n_comparisons": n_cmp, "exact_tie_share_original": n_tie / n_cmp, "exact_tie_share_shifted": n_tie_shift / n_cmp,
              "distinct_feature_threshold_pairs": len(pairs), "pairs_on_lattice": on_lattice,
              "per_feature_tie_share": {c: (v[1] / v[0]) if v[0] else None for c, v in per_feat.items()}}
    print(f"ties: {census['exact_tie_share_original']:.4f} of comparisons (shifted: {census['exact_tie_share_shifted']:.4f}); "
          f"{on_lattice} of {len(pairs)} distinct thresholds on the lattice", flush=True)

    cache: dict = {}

    def coef_for(vname: str, P_: float, k: int):
        z_ref = float(np.percentile(variants[vname]["abs_z"], P_))
        with np.errstate(divide="ignore"):
            gamma = float(np.log(SIGMOID_EDGE_LEVEL / (1 - SIGMOID_EDGE_LEVEL)) / z_ref) if z_ref > 0 else float("inf")
        return C.Chebyshev.interpolate(lambda z: soft_step(z, gamma), deg=k, domain=[-1.0, 1.0]).coef, gamma

    def full_cell(vname: str, P_: float, k: int) -> dict:
        if (vname, P_, k) in cache:
            return cache[(vname, P_, k)]
        coef, gamma = coef_for(vname, P_, k)
        sim = margin(variants[vname], lambda z: C.chebval(z, coef), Xva_mat)
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
            "auroc_delta_plain_minus_ckks": float(auc_full - roc_auc_score(yva, sim)),
            "f1_plain_val": f1_plain,
            "f1_ckks_at_plain_threshold_val": float(f1_score(yva, pc, zero_division=0)),
            "f1_ckks_at_own_threshold_val": float(f1_score(yva, (pr >= own).astype(int), zero_division=0)),
        }
        cache[(vname, P_, k)] = cell
        return cell

    results: dict = {"original": {}, "shifted": {}}
    for k in DEGREES:
        for vname in ("original", "shifted"):
            t0 = time.perf_counter()
            cands = []
            for P_ in GAMMA_GRID:
                coef, gamma = coef_for(vname, P_, k)
                sim = margin(variants[vname], lambda z: C.chebval(z, coef), Xsub)
                cands.append({"percentile": P_, "gamma": gamma,
                              "decision_agreement": float(np.mean(pp_sub == (sim >= thr_margin).astype(int))),
                              "nmae": float(np.mean(np.abs(sim - plain_sub)) / span_plain)})
            top = max(r["decision_agreement"] for r in cands)
            tied = [r for r in cands if top - r["decision_agreement"] <= TIE_TOLERANCE]
            win = sorted(tied, key=lambda r: (r["nmae"], -r["gamma"]))[0]
            cell = dict(full_cell(vname, win["percentile"], k))
            cell["scoring_candidates"] = cands
            results[vname][f"k={k}"] = cell
            print(f"k={k} {vname}: p{win['percentile']} {cell['disagree_per_1000']:.1f}/1000 "
                  f"{[round(x, 1) for x in cell['disagree_boot95_per_1000']]} lost {cell['lost_share_of_plain_positives']:.3f} "
                  f"AUROCd {cell['auroc_delta_plain_minus_ckks']:.4f} [{time.perf_counter() - t0:.0f}s]", flush=True)
            if vname == "original" and k == 15:
                ref = p4["results"]["k=15"]["disagree_per_1000"]
                assert abs(cell["disagree_per_1000"] - ref) <= 0.5, f"pilot 4 k=15 not reproduced: {cell['disagree_per_1000']} vs {ref}"
        with open(OUT.replace(".json", "_partial.json"), "w", encoding="utf-8") as f:
            json.dump({"census": census, "results_so_far": results}, f)

    meets = [{"variant": v, "k": k, "meets_fidelity": bool(c["disagree_per_1000"] <= FID_PER_1000
                                                            and c["lost_share_of_plain_positives"] <= FID_LOST_POS)}
             for v, d in results.items() for k, c in d.items()]
    out = {
        "protocol": "docs/m1c_pilot5_prespec.md", "backend": "SIMULATED (circuit-faithful NumPy, Chebyshev-basis comparison)",
        "scope": "seed 42, validation split only, test split not read; gamma scored on the fixed 25% subsample",
        "model_auroc_val": auc_full, "n_val_stays": int(len(va)), "n_plain_positive_decisions": n_pos_plain,
        "lossless_max_abs_margin_difference_exact_step": float(np.max(np.abs(m_o - m_s))),
        "tie_census": census, "results": results,
        "fidelity_criteria": {"max_disagree_per_1000": FID_PER_1000, "max_lost_share_of_plain_positives": FID_LOST_POS},
        "meets_fidelity": meets, "runtime_seconds": time.perf_counter() - t_start,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"written -> {OUT} ({out['runtime_seconds']:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
