#!/usr/bin/env python3
"""
Fourth pilot for the M1 continuous-feature task (added analysis): deeper-chain simulation up to k = 63 and a
threshold-adjacency check. Executes docs/m1c_pilot4_prespec.md (committed before this file existed). Validation
split only; the test split is never selected. Simulated circuit. Aggregate output only: results/m1c_pilot4.json.
"""
from __future__ import annotations

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
from select_gamma_anchor import fit_power_coefficients  # noqa: E402
from splits import patient_level_split  # noqa: E402
from train_plaintext import tune_thresholds  # noqa: E402

OUT = os.path.join(_HERE, "..", "results", "m1c_pilot4.json")
PILOT1 = os.path.join(_HERE, "..", "results", "m1c_pilot_validity.json")
PILOT3 = os.path.join(_HERE, "..", "results", "m1c_pilot3.json")
SEED = 42
DEGREES = (15, 23, 31, 47, 63)
GAMMA_GRID = [1, 5, 25]
SCORE_FRACTION = 0.25
TIE_TOLERANCE = 0.002
N_BOOT, BOOT_SEED = 1000, 42
FID_PER_1000, FID_LOST_POS = 50.0, 0.10
CHUNK = 4000
RUNTIME_LIMIT_S = 55 * 60
LATTICE = {"triage_temperature": 0.1 * 5.0 / 9.0, "triage_heartrate": 1.0, "triage_resprate": 1.0, "triage_o2sat": 1.0,
           "triage_sbp": 1.0, "triage_dbp": 1.0, "triage_pain": 1.0, "age": 1.0}


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
    with open(PILOT3, "r", encoding="utf-8") as f:
        p3 = json.load(f)
    assert abs(auc_full - p1["positive_control"]["auroc_full_val"]) < 1e-9, "model differs from earlier pilots"
    proba = 1.0 / (1.0 + np.exp(-plain))
    thr = tune_thresholds(yva.reshape(-1, 1), proba.reshape(-1, 1))[0]
    thr_margin = float(np.log(thr / (1.0 - thr)))
    assert abs(thr_margin - p1["positive_control"]["tuned_threshold_margin"]) < 1e-12
    pp = (plain >= thr_margin).astype(int)
    n_pos_plain = int(pp.sum())
    f1_plain = float(f1_score(yva, pp, zero_division=0))

    structure = cc.extract_ensemble_leaf_circuit(booster, set(feats), tree_depth=6)
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
    _, perfeat = pooled_abs_z(clf, Xtr, norm_B)
    norm_F = {c: {"min": LIMITS[c][0], "max": LIMITS[c][0] + (LIMITS[c][1] - LIMITS[c][0]) * perfeat[c]} for c in perfeat}
    for c in feats:
        norm_F.setdefault(c, norm_B[c])
    abs_z, _ = pooled_abs_z(clf, Xtr, norm_F)
    bl = cc.prepare_batch_levels(structure, feats, norm_F, 1.0)
    yes_masks = [np.array([r.kind != "pad" and r.is_yes_edge for r in lvl]) for lvl in structure.levels]
    pad_masks = [np.array([r.kind == "pad" for r in lvl]) for lvl in structure.levels]

    def margin_cheb(coef: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Circuit margin with the scalar comparison polynomial in the Chebyshev basis (Clenshaw)."""
        out = np.empty(X.shape[0])
        for s in range(0, X.shape[0], CHUNK):
            Xc = X[s : s + CHUNK]
            prod = np.ones((Xc.shape[0], structure.n_leaves))
            for lvl, yes, pad in zip(bl, yes_masks, pad_masks):
                z = np.clip((Xc[:, lvl.feature_col_idx] - lvl.threshold[None, :]) / lvl.span[None, :], -1.0, 1.0)
                h = C.chebval(z, coef)
                f = np.where(yes[None, :], h, 1.0 - h)
                prod *= np.where(pad[None, :], 1.0, f)
            out[s : s + CHUNK] = prod @ structure.leaf_value
        return out

    # ---- equivalence test (i): Chebyshev-basis evaluator vs the existing power-basis evaluator at k = 9
    g_ref = p3["results"]["soft_step"]["k=9"]["gamma"]
    cheb9 = C.Chebyshev.interpolate(lambda z: soft_step(z, g_ref), deg=9, domain=[-1.0, 1.0]).coef
    Xchk = Xva_mat[:2000]
    old = cc.evaluate_circuit_batch(bl, cc.assemble_coefficient_vectors(structure, fit_power_coefficients(g_ref, 1.0, 9), 9),
                                    structure.leaf_value, Xchk, 1.0)
    new = margin_cheb(cheb9, Xchk)
    maxdiff = float(np.max(np.abs(old - new)))
    assert maxdiff < 1e-8, f"Chebyshev evaluator differs from the power-basis evaluator by {maxdiff}"
    print(f"equivalence (k=9, 2000 stays): max |margin difference| = {maxdiff:.2e}", flush=True)

    def coef_for(P_: float, k: int) -> tuple[np.ndarray, float]:
        z_ref = float(np.percentile(abs_z, P_))
        gamma = float(np.log(SIGMOID_EDGE_LEVEL / (1 - SIGMOID_EDGE_LEVEL)) / z_ref)
        return C.Chebyshev.interpolate(lambda z: soft_step(z, gamma), deg=k, domain=[-1.0, 1.0]).coef, gamma

    # ---- runtime guard
    t0 = time.perf_counter()
    margin_cheb(coef_for(5, 63)[0], Xsub)
    dt63 = time.perf_counter() - t0
    projected = 7.0 * dt63 * sum(DEGREES) / 63.0
    print(f"one k=63 scoring evaluation: {dt63:.0f}s; projected total {projected / 60:.0f} min", flush=True)
    if projected > RUNTIME_LIMIT_S:
        print(f"STOP: projected {projected / 60:.0f} min exceeds the {RUNTIME_LIMIT_S // 60} min limit; ask the author", flush=True)
        return 2

    cache: dict = {}

    def full_cell(P_: float, k: int) -> dict:
        if (P_, k) in cache:
            return cache[(P_, k)]
        coef, gamma = coef_for(P_, k)
        sim = margin_cheb(coef, Xva_mat)
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
        cache[(P_, k)] = cell
        return cell

    # ---- equivalence test (ii): pilot 3 soft-step k = 15
    ref15 = p3["results"]["soft_step"]["k=15"]
    chk = full_cell(ref15["selected_percentile"], 15)["disagree_per_1000"]
    assert abs(chk - ref15["disagree_per_1000"]) <= 0.5, f"pilot 3 k=15 not reproduced: {chk} vs {ref15['disagree_per_1000']}"
    print(f"reproduced pilot 3 at k=15: {chk:.2f} vs {ref15['disagree_per_1000']:.2f} per 1000", flush=True)

    # ---- threshold adjacency (aggregate counts of comparisons)
    n_tot = 0
    near1 = near2 = near_z = 0
    per_feat = {c: {"n": 0, "within_1_step": 0} for c in feats}
    for s in range(0, Xva_mat.shape[0], CHUNK):
        Xc = Xva_mat[s : s + CHUNK]
        for d, lvl in enumerate(bl):
            refs = structure.levels[d]
            step = np.array([LATTICE[r.feature] if r.kind != "pad" else np.nan for r in refs])
            fnames = np.array([r.feature if r.kind != "pad" else "" for r in refs])
            raw = np.abs(Xc[:, lvl.feature_col_idx] - lvl.threshold[None, :])
            valid = ~pad_masks[d]
            a1 = (raw <= step[None, :] + 1e-9)[:, valid]
            a2 = (raw <= 2 * step[None, :] + 1e-9)[:, valid]
            zs = np.abs(np.clip((Xc[:, lvl.feature_col_idx] - lvl.threshold[None, :]) / lvl.span[None, :], -1, 1))[:, valid]
            n_tot += a1.size
            near1 += int(a1.sum())
            near2 += int(a2.sum())
            near_z += int((zs < 0.05).sum())
            for c in feats:
                m = (fnames == c)[valid]
                if m.any():
                    per_feat[c]["n"] += int(a1[:, m].size)
                    per_feat[c]["within_1_step"] += int(a1[:, m].sum())
    adjacency = {"n_comparisons": n_tot, "share_within_1_step": near1 / n_tot, "share_within_2_steps": near2 / n_tot,
                 "share_scaled_abs_z_below_0.05": near_z / n_tot,
                 "per_feature": {c: {**v, "share": (v["within_1_step"] / v["n"]) if v["n"] else None} for c, v in per_feat.items()}}
    print(f"adjacency: {adjacency['share_within_1_step']:.4f} of comparisons within 1 lattice step, "
          f"{adjacency['share_within_2_steps']:.4f} within 2", flush=True)

    results = {}
    for k in DEGREES:
        t0 = time.perf_counter()
        cands = []
        for P_ in GAMMA_GRID:
            coef, gamma = coef_for(P_, k)
            sim = margin_cheb(coef, Xsub)
            cands.append({"percentile": P_, "gamma": gamma,
                          "decision_agreement": float(np.mean(pp_sub == (sim >= thr_margin).astype(int))),
                          "nmae": float(np.mean(np.abs(sim - plain_sub)) / span_plain)})
        top = max(r["decision_agreement"] for r in cands)
        tied = [r for r in cands if top - r["decision_agreement"] <= TIE_TOLERANCE]
        win = sorted(tied, key=lambda r: (r["nmae"], -r["gamma"]))[0]
        cell = dict(full_cell(win["percentile"], k))
        cell["scoring_candidates"] = cands
        results[f"k={k}"] = cell
        print(f"k={k}: p{win['percentile']} {cell['disagree_per_1000']:.1f}/1000 "
              f"{[round(x, 1) for x in cell['disagree_boot95_per_1000']]} lost {cell['lost_share_of_plain_positives']:.3f} "
              f"AUROCd {cell['auroc_delta_plain_minus_ckks']:.4f} [{time.perf_counter() - t0:.0f}s]", flush=True)
        with open(OUT.replace(".json", "_partial.json"), "w", encoding="utf-8") as f:
            json.dump({"results_so_far": results, "adjacency": adjacency}, f)

    d63 = results["k=63"]["disagree_per_1000"] / 1000.0
    print(f"ratio of adjacency share to the k=63 disagreement rate: {adjacency['share_within_1_step'] / d63:.2f}", flush=True)
    adjacency["k63_disagreement_rate"] = d63
    adjacency["ratio_adjacency_to_k63_disagreement"] = adjacency["share_within_1_step"] / d63
    print(f"adjacency: {adjacency['share_within_1_step']:.4f} of comparisons within 1 lattice step "
          f"(k=63 disagreement rate {d63:.4f}; ratio {adjacency['ratio_adjacency_to_k63_disagreement']:.2f})", flush=True)

    meets = [{"k": k, "meets_fidelity": bool(c["disagree_per_1000"] <= FID_PER_1000
                                              and c["lost_share_of_plain_positives"] <= FID_LOST_POS)}
             for k, c in results.items()]
    out = {
        "protocol": "docs/m1c_pilot4_prespec.md", "backend": "SIMULATED (circuit-faithful NumPy, Chebyshev-basis comparison)",
        "scope": "seed 42, validation split only, test split not read; gamma scored on the fixed 25% subsample",
        "model_auroc_val": auc_full, "n_val_stays": int(len(va)), "n_plain_positive_decisions": n_pos_plain,
        "equivalence_k9_max_abs_margin_difference": maxdiff, "pilot3_k15_reproduced_per_1000": chk,
        "results": results, "adjacency": adjacency,
        "fidelity_criteria": {"max_disagree_per_1000": FID_PER_1000, "max_lost_share_of_plain_positives": FID_LOST_POS},
        "meets_fidelity": meets, "runtime_seconds": time.perf_counter() - t_start,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"written -> {OUT} ({out['runtime_seconds']:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
