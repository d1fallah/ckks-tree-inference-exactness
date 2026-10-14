#!/usr/bin/env python3
"""
Third validity pilot for the M1 continuous-feature task (added analysis). Executes docs/m1c_pilot3_prespec.md
(committed before this file existed). Validation split only; the test split is never selected. Simulated
circuit. Aggregate output only: results/m1c_pilot3.json.
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
from numpy.polynomial import polynomial as P
from scipy.optimize import linprog
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, roc_auc_score

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import ckks_circuit as cc  # noqa: E402
import m1c_cohort as mc  # noqa: E402
from generate_poly_coefficients import SIGMOID_EDGE_LEVEL  # noqa: E402
from m1c_pilot2 import pooled_abs_z  # noqa: E402
from m1c_pilot_validity import LIMITS, fit_model, simulate  # noqa: E402
from select_gamma_anchor import fit_power_coefficients  # noqa: E402
from splits import patient_level_split  # noqa: E402
from train_plaintext import tune_thresholds  # noqa: E402

OUT = os.path.join(_HERE, "..", "results", "m1c_pilot3.json")
PILOT1 = os.path.join(_HERE, "..", "results", "m1c_pilot_validity.json")
PILOT2 = os.path.join(_HERE, "..", "results", "m1c_pilot2.json")
SEED = 42
DEGREES = (5, 7, 9, 11, 13, 15)
GAMMA_GRID = [5, 10, 25, 50, 75, 90, 95, 99.9]
EPS_GRID = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
SCORE_FRACTION = 0.25
TIE_TOLERANCE = 0.002
RESOLVE_TOLS = (0.05, 0.10, 0.20)
N_BOOT, BOOT_SEED = 1000, 42
FID_PER_1000, FID_LOST_POS = 50.0, 0.10
HIST_BINS = 20000


def minimax_odd_coeffs(k: int, eps: float) -> dict:
    """Minimax odd polynomial p of degree k for sign on [eps, 1] (linear program on a dense grid; Chebyshev
    basis, odd terms only), returned as the power-basis step h(z) = 0.5 - 0.5 p(z) the circuit evaluates."""
    xs = np.cos(np.linspace(0.0, np.pi, 4001))  # Chebyshev-distributed on [-1, 1]
    xs = np.unique(np.concatenate([xs[xs >= eps], np.linspace(eps, 1.0, 2001)]))
    odd = list(range(1, k + 1, 2))
    T = np.column_stack([np.cos(j * np.arccos(np.clip(xs, -1, 1))) for j in odd])
    n = len(odd)
    c_obj = np.zeros(n + 1)
    c_obj[-1] = 1.0
    A = np.vstack([np.hstack([T, -np.ones((len(xs), 1))]), np.hstack([-T, -np.ones((len(xs), 1))])])
    b = np.concatenate([np.ones(len(xs)), -np.ones(len(xs))])
    res = linprog(c_obj, A_ub=A, b_ub=b, bounds=[(None, None)] * n + [(0, None)], method="highs")
    assert res.status == 0, f"minimax LP failed for k={k}, eps={eps}: {res.message}"
    cheb = np.zeros(k + 1)
    for j, cj in zip(odd, res.x[:n]):
        cheb[j] = cj
    p_pow = C.Chebyshev(cheb, domain=[-1, 1]).convert(kind=np.polynomial.Polynomial).coef
    h = -0.5 * np.asarray(p_pow, dtype=float)
    h[0] += 0.5
    return {f"k={k}": {"coefficients_power_basis": [float(v) for v in h]}, "_minimax_error": float(res.x[-1])}


def band_edge(power_coef: list[float], tol: float) -> float:
    """Smallest e with |h(z) - step(z)| <= tol for every |z| >= e (grid of 20,001 points on [-1, 1])."""
    z = np.linspace(-1.0, 1.0, 20001)
    h = P.polyval(z, power_coef)
    step = (z < 0).astype(float)
    bad = np.abs(h - step) > tol
    bad &= np.abs(z) > 0
    return float(np.max(np.abs(z[bad]))) if bad.any() else 0.0


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
    with open(PILOT2, "r", encoding="utf-8") as f:
        p2 = json.load(f)
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

    # per-feature interval, exactly as pilot 2
    norm_B = {c: {"min": LIMITS[c][0], "max": LIMITS[c][1]} for c in feats}
    _, perfeat = pooled_abs_z(clf, Xtr, norm_B)
    norm_F = {c: {"min": LIMITS[c][0], "max": LIMITS[c][0] + (LIMITS[c][1] - LIMITS[c][0]) * perfeat[c]} for c in perfeat}
    for c in feats:
        norm_F.setdefault(c, norm_B[c])
    abs_z, _ = pooled_abs_z(clf, Xtr, norm_F)
    bl = cc.prepare_batch_levels(structure, feats, norm_F, 1.0)

    # histogram of |z'| over every comparison the circuit evaluates on the validation split (aggregate)
    hist = np.zeros(HIST_BINS, dtype=np.int64)
    n_cmp = 0
    for s in range(0, Xva_mat.shape[0], 4000):
        Xc = Xva_mat[s : s + 4000]
        for lvl in bl:
            z = (Xc[:, lvl.feature_col_idx] - lvl.threshold[None, :]) / lvl.span[None, :]
            z = np.clip(z, -1.0, 1.0)[:, lvl.should_clamp]
            h, _ = np.histogram(np.abs(z), bins=HIST_BINS, range=(0.0, 1.0))
            hist += h
            n_cmp += z.size
    cum = np.cumsum(hist)

    def unresolved_share(edge: float) -> float:
        i = int(np.floor(edge * HIST_BINS))
        return float(cum[i - 1] / n_cmp) if i >= 1 else 0.0

    def score(coef_dict, k, Xm, plain_m, pp_m):
        ck = cc.assemble_coefficient_vectors(structure, coef_dict, k)
        sim = simulate(bl, ck, structure.leaf_value, Xm, 1.0)
        pc = (sim >= thr_margin).astype(int)
        return sim, pc

    def full_cell(coef_dict, k):
        sim, pc = score(coef_dict, k, Xva_mat, plain, pp)
        diff = pp != pc
        per_subj = np.bincount(codes[diff], minlength=n_subj).astype(np.int64)
        boot = (w @ per_subj) / n_boot_stays
        lost = int(((pp == 1) & (pc == 0)).sum())
        gained = int(((pp == 0) & (pc == 1)).sum())
        own = tune_thresholds(yva.reshape(-1, 1), (1.0 / (1.0 + np.exp(-sim))).reshape(-1, 1))[0]
        pc_own = ((1.0 / (1.0 + np.exp(-sim))) >= own).astype(int)
        dis = float(1000 * diff.mean())
        return {
            "disagree_per_1000": dis,
            "disagree_boot95_per_1000": [float(1000 * np.percentile(boot, 2.5)), float(1000 * np.percentile(boot, 97.5))],
            "n_plain_pos_to_ckks_neg": lost, "n_plain_neg_to_ckks_pos": gained,
            "lost_share_of_plain_positives": lost / n_pos_plain,
            "auroc_ckks_val": float(roc_auc_score(yva, sim)),
            "auroc_delta_plain_minus_ckks": float(auc_full - roc_auc_score(yva, sim)),
            "f1_plain_val": f1_plain,
            "f1_ckks_at_plain_threshold_val": float(f1_score(yva, pc, zero_division=0)),
            "circuit_own_threshold": float(own),
            "f1_ckks_at_own_threshold_val": float(f1_score(yva, pc_own, zero_division=0)),
        }

    # consistency with pilot 2 (soft step, pilot-2 gamma, k = 9)
    g2 = p2["configs"]["per_feature|threshold_decision"]["gamma"]
    chk = full_cell(fit_power_coefficients(g2, 1.0, 9), 9)
    ref9 = p2["configs"]["per_feature|threshold_decision"]["cells"]["k=9"]["disagree_per_1000"]
    assert abs(chk["disagree_per_1000"] - ref9) < 1e-9, f"pilot-2 k=9 not reproduced: {chk['disagree_per_1000']} vs {ref9}"
    print(f"reproduced pilot 2 at k=9: {ref9:.2f} per 1000", flush=True)

    results: dict = {"soft_step": {}, "minimax_gap": {}}
    for k in DEGREES:
        t0 = time.perf_counter()
        # ---- soft-step interpolation: gamma chosen at this k on the scoring subsample
        cands = []
        for Pc in GAMMA_GRID:
            z_ref = float(np.percentile(abs_z, Pc))
            gamma = float(np.log(SIGMOID_EDGE_LEVEL / (1 - SIGMOID_EDGE_LEVEL)) / z_ref)
            cd = fit_power_coefficients(gamma, 1.0, k)
            sim, pc = score(cd, k, Xsub, plain_sub, pp_sub)
            cands.append({"param": Pc, "gamma": gamma, "coef": cd,
                          "decision_agreement": float(np.mean(pp_sub == pc)),
                          "nmae": float(np.mean(np.abs(sim - plain_sub)) / span_plain)})
        top = max(r["decision_agreement"] for r in cands)
        tied = [r for r in cands if top - r["decision_agreement"] <= TIE_TOLERANCE]
        win = sorted(tied, key=lambda r: (r["nmae"], -r["gamma"]))[0]
        cell = full_cell(win["coef"], k)
        bands = {str(tol): band_edge(win["coef"][f"k={k}"]["coefficients_power_basis"], tol) for tol in RESOLVE_TOLS}
        cell.update({"selected_percentile": win["param"], "gamma": win["gamma"], "band_edge": bands,
                     "unresolved_share_of_comparisons": {t_: unresolved_share(e) for t_, e in bands.items()},
                     "scoring_candidates": [{kk: v for kk, v in r.items() if kk != "coef"} for r in cands]})
        results["soft_step"][f"k={k}"] = cell
        # ---- minimax odd polynomial with gap: eps chosen at this k on the same subsample
        cands = []
        for eps in EPS_GRID:
            mm = minimax_odd_coeffs(k, eps)
            cd = {f"k={k}": mm[f"k={k}"]}
            sim, pc = score(cd, k, Xsub, plain_sub, pp_sub)
            cands.append({"param": eps, "minimax_error_on_gap_complement": mm["_minimax_error"], "coef": cd,
                          "decision_agreement": float(np.mean(pp_sub == pc)),
                          "nmae": float(np.mean(np.abs(sim - plain_sub)) / span_plain)})
        top = max(r["decision_agreement"] for r in cands)
        tied = [r for r in cands if top - r["decision_agreement"] <= TIE_TOLERANCE]
        win = sorted(tied, key=lambda r: (r["nmae"], -r["param"]))[0]
        cell = full_cell(win["coef"], k)
        bands = {str(tol): band_edge(win["coef"][f"k={k}"]["coefficients_power_basis"], tol) for tol in RESOLVE_TOLS}
        cell.update({"selected_eps": win["param"], "minimax_error_on_gap_complement": win["minimax_error_on_gap_complement"],
                     "band_edge": bands, "unresolved_share_of_comparisons": {t_: unresolved_share(e) for t_, e in bands.items()},
                     "scoring_candidates": [{kk: v for kk, v in r.items() if kk != "coef"} for r in cands]})
        results["minimax_gap"][f"k={k}"] = cell
        s, m = results["soft_step"][f"k={k}"], results["minimax_gap"][f"k={k}"]
        print(f"k={k}: soft {s['disagree_per_1000']:.1f}/1000 (lost {s['lost_share_of_plain_positives']:.3f}, band@0.10 {s['band_edge']['0.1']:.3f}) | "
              f"minimax eps={m['selected_eps']} {m['disagree_per_1000']:.1f}/1000 (lost {m['lost_share_of_plain_positives']:.3f}, "
              f"band@0.10 {m['band_edge']['0.1']:.3f}) [{time.perf_counter() - t0:.0f}s]", flush=True)

    cells = [(fam, k, v) for fam, d in results.items() for k, v in d.items()]
    rhos = {str(tol): float(spearmanr([v["unresolved_share_of_comparisons"][str(tol)] for _, _, v in cells],
                                      [v["disagree_per_1000"] for _, _, v in cells]).statistic) for tol in RESOLVE_TOLS}
    meets = [{"family": fam, "k": k, "meets_fidelity": bool(v["disagree_per_1000"] <= FID_PER_1000
                                                             and v["lost_share_of_plain_positives"] <= FID_LOST_POS)}
             for fam, k, v in cells]
    out = {
        "protocol": "docs/m1c_pilot3_prespec.md", "backend": "SIMULATED (circuit-faithful NumPy)",
        "scope": "seed 42, validation split only, test split not read; parameters scored on the fixed 25% subsample",
        "model_auroc_val": auc_full, "n_val_stays": int(len(va)), "n_plain_positive_decisions": n_pos_plain,
        "n_comparisons_counted": int(n_cmp), "resolve_tolerances": list(RESOLVE_TOLS),
        "pilot2_k9_reproduced_disagree_per_1000": chk["disagree_per_1000"],
        "results": results,
        "spearman_unresolved_share_vs_disagreement": {"rho_by_tolerance": rhos, "n_cells": len(cells)},
        "fidelity_criteria": {"max_disagree_per_1000": FID_PER_1000, "max_lost_share_of_plain_positives": FID_LOST_POS},
        "meets_fidelity": meets, "runtime_seconds": time.perf_counter() - t_start,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"spearman rho (unresolved share vs disagreement) by tolerance = {rhos} over {len(cells)} cells; "
          f"written -> {OUT} ({out['runtime_seconds']:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
