#!/usr/bin/env python3
"""
Validity-gate pilot for the M1 continuous-feature task. Executes docs/m1c_pilot_prespec.md (committed before
this file ran). One seed (42), one outcome, TRAIN and VALIDATION splits only; the test split is never
selected. Simulated circuit (circuit-faithful NumPy), not real ciphertexts.

Aggregate output only: results/m1c_pilot_validity.json. No row is printed or written (CLAUDE.md 1.2).
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
from derive_circuit_params import derive_normalization, pooled_interval  # noqa: E402
from generate_poly_coefficients import SIGMOID_EDGE_LEVEL  # noqa: E402
from select_gamma_anchor import fit_power_coefficients  # noqa: E402
from splits import patient_level_split  # noqa: E402
from train_plaintext import MAX_DEPTH, MAX_TREES, tune_thresholds  # noqa: E402

OUT = os.path.join(_HERE, "..", "results", "m1c_pilot_validity.json")
SEED = 42
DEGREES = (1, 3, 5, 7, 9)
ANCHORS = [50, 75, 90, 95, 99.9]
TIE_TOLERANCE = 0.002
N_BOOT, BOOT_SEED = 1000, 42
CHUNK = 8000
# Variant B: public clinical limits, fixed in docs/m1c_pilot_prespec.md.
LIMITS = {
    "triage_temperature": (30.0, 43.0), "triage_heartrate": (20.0, 250.0), "triage_resprate": (4.0, 80.0),
    "triage_o2sat": (50.0, 100.0), "triage_sbp": (40.0, 300.0), "triage_dbp": (20.0, 200.0),
    "triage_pain": (0.0, 10.0), "age": (18.0, 105.0),
}
PREVALENCE_REFERENCE = 0.0515  # M2 outcome gate, horizon-labelled 12 h rate
VITAL_FEATURES = mc.VITAL_FEATURES


def fit_model(X: pd.DataFrame, y: np.ndarray) -> xgb.XGBClassifier:
    # Same configuration as train_plaintext.train_and_eval (base_score 0.5 is locked there).
    clf = xgb.XGBClassifier(
        n_estimators=MAX_TREES, max_depth=MAX_DEPTH, tree_method="hist", subsample=0.8, colsample_bytree=0.8,
        base_score=0.5, random_state=SEED, eval_metric="logloss",
    )
    clf.fit(X, y)
    return clf


def simulate(batch_levels, coeffs, leaf_value, X_mat, a) -> np.ndarray:
    out = np.empty(X_mat.shape[0])
    for s in range(0, X_mat.shape[0], CHUNK):
        out[s : s + CHUNK] = cc.evaluate_circuit_batch(batch_levels, coeffs, leaf_value, X_mat[s : s + CHUNK], a)
    return out


def main() -> int:
    t_start = time.perf_counter()
    # D4 by source inspection: the cohort loader never reads ed/vitalsign.
    with open(os.path.join(_HERE, "m1c_cohort.py"), "r", encoding="utf-8") as f:
        assert "vitalsign.csv" not in f.read(), "D4 FAILED: cohort loader reads vitalsign"

    df, flow = mc.build_cohort()
    train_ids, val_ids, test_ids = patient_level_split(df)
    assert not (train_ids & val_ids) and not (train_ids & test_ids) and not (val_ids & test_ids)
    tr = df[df["subject_id"].isin(train_ids)].reset_index(drop=True)
    va = df[df["subject_id"].isin(val_ids)].reset_index(drop=True)
    assert set(va["subject_id"]).isdisjoint(test_ids) and set(tr["subject_id"]).isdisjoint(test_ids)
    del df  # the test split is not used from here on
    feats = mc.FEATURE_COLS

    # ---- data realism (D1-D3), on the training split before imputation
    d_checks = {"per_feature": {}}
    d1 = d2 = True
    for c in feats:
        lo, hi = LIMITS[c]
        v = tr[c].dropna().to_numpy(dtype=float)
        q001, q999 = float(np.quantile(v, 0.001)), float(np.quantile(v, 0.999))
        outside = float(((v < lo) | (v > hi)).mean())
        ok1 = bool(lo <= q001 and q999 <= hi)
        ok2 = bool(outside <= 0.005)
        d1 &= ok1
        d2 &= ok2
        d_checks["per_feature"][c] = {"q0.1": q001, "q99.9": q999, "limits": [lo, hi],
                                      "share_outside_limits": outside, "D1_ok": ok1, "D2_ok": ok2}
    prev = {"train": float(tr[mc.OUTCOME_COL].mean()), "val": float(va[mc.OUTCOME_COL].mean())}
    d3 = all(abs(p - PREVALENCE_REFERENCE) <= 0.01 for p in prev.values())
    d_checks.update({"D1": bool(d1), "D2": bool(d2), "D3": bool(d3), "D4": True, "prevalence": prev,
                     "prevalence_reference": PREVALENCE_REFERENCE})

    # ---- imputation (training medians), no indicators
    med = {c: float(tr[c].median()) for c in VITAL_FEATURES}
    Xtr = tr[feats].copy()
    Xva = va[feats].copy()
    for c in VITAL_FEATURES:
        Xtr[c] = Xtr[c].fillna(med[c])
        Xva[c] = Xva[c].fillna(med[c])
    ytr = tr[mc.OUTCOME_COL].to_numpy(dtype=int)
    yva = va[mc.OUTCOME_COL].to_numpy(dtype=int)
    codes, uniques = pd.factorize(va["subject_id"])
    n_subj = len(uniques)

    # ---- positive control, plaintext (P1, P2)
    clf = fit_model(Xtr, ytr)
    booster = clf.get_booster()
    plain = booster.predict(xgb.DMatrix(Xva, feature_names=feats), output_margin=True)
    auc_full = float(roc_auc_score(yva, plain))
    clf_age = fit_model(Xtr[["age"]], ytr)
    auc_age = float(roc_auc_score(yva, clf_age.predict_proba(Xva[["age"]])[:, 1]))
    p1, p2 = bool(auc_full >= 0.60), bool(auc_full - auc_age >= 0.03)
    proba_val = 1.0 / (1.0 + np.exp(-plain))
    thr = tune_thresholds(yva.reshape(-1, 1), proba_val.reshape(-1, 1))[0]
    thr_margin = float(np.log(thr / (1.0 - thr)))
    pp = (plain >= thr_margin).astype(int)
    census = booster.trees_to_dataframe()
    census = census[census["Feature"] != "Leaf"]["Feature"].value_counts().to_dict()
    positive = {"auroc_full_val": auc_full, "auroc_age_only_val": auc_age, "auroc_gain_over_age_only": auc_full - auc_age,
                "P1": p1, "P2": p2, "tuned_threshold": thr, "tuned_threshold_margin": thr_margin,
                "predicted_positive_rate": float(pp.mean()), "f1_plain_val": float(f1_score(yva, pp, zero_division=0)),
                "n_internal_splits": int(sum(census.values())),
                "splits_per_feature": {k: int(v) for k, v in census.items()},
                "binary_splits": 0}

    # ---- circuit, two normalization variants
    structure = cc.extract_ensemble_leaf_circuit(booster, set(feats), tree_depth=6)
    Xva_mat = Xva.to_numpy(dtype=np.float64)
    rng = np.random.default_rng(BOOT_SEED)
    w = rng.multinomial(n_subj, np.full(n_subj, 1.0 / n_subj), size=N_BOOT).astype(np.int64)
    n_boot_stays = w @ np.bincount(codes, minlength=n_subj).astype(np.int64)
    variants = {"A_data_bounds": derive_normalization(Xtr),
                "B_clinical_limits": {c: {"min": LIMITS[c][0], "max": LIMITS[c][1]} for c in feats}}
    variant_out = {}
    for vname, norm in variants.items():
        t0 = time.perf_counter()
        interval = pooled_interval({"pilot": clf}, Xtr, norm, set(), seed=SEED)
        a = interval["a_half_width"]
        quant = interval["abs_diff_quantiles"]
        bl = cc.prepare_batch_levels(structure, feats, norm, a)
        span = float(plain.max() - plain.min()) or 1.0
        cand = []
        for P in ANCHORS:
            z_ref = quant[P]
            gamma = float(np.log(SIGMOID_EDGE_LEVEL / (1 - SIGMOID_EDGE_LEVEL)) / z_ref)
            c5 = cc.assemble_coefficient_vectors(structure, fit_power_coefficients(gamma, a, 5), 5)
            sim5 = simulate(bl, c5, structure.leaf_value, Xva_mat, a)
            cand.append({"percentile": P, "z_ref": z_ref, "gamma": gamma,
                         "agreement": float(np.mean(np.sign(plain) == np.sign(sim5))),
                         "nmae": float(np.mean(np.abs(sim5 - plain)) / span)})
        top = max(r["agreement"] for r in cand)
        tied = [r for r in cand if top - r["agreement"] <= TIE_TOLERANCE]
        win = sorted(tied, key=lambda r: (r["nmae"], -r["gamma"]))[0]
        gamma = win["gamma"]
        cells = {}
        rates = {}
        for k in DEGREES:
            ck = cc.assemble_coefficient_vectors(structure, fit_power_coefficients(gamma, a, k), k)
            sim = simulate(bl, ck, structure.leaf_value, Xva_mat, a)
            pc = (sim >= thr_margin).astype(int)
            diff = pp != pc
            per_subj = np.bincount(codes[diff], minlength=n_subj).astype(np.int64)
            boot = (w @ per_subj) / n_boot_stays
            rates[k] = (diff.astype(float), per_subj)
            dp = np.abs(proba_val - 1.0 / (1.0 + np.exp(-sim)))
            cells[f"k={k}"] = {
                "disagree_per_1000": float(1000 * diff.mean()),
                "disagree_boot95_per_1000": [float(1000 * np.percentile(boot, 2.5)), float(1000 * np.percentile(boot, 97.5))],
                "n_plain_pos_to_ckks_neg": int(((pp == 1) & (pc == 0)).sum()),
                "n_plain_neg_to_ckks_pos": int(((pp == 0) & (pc == 1)).sum()),
                "f1_ckks_val": float(f1_score(yva, pc, zero_division=0)),
                "f1_delta_plain_minus_ckks": float(f1_score(yva, pp, zero_division=0) - f1_score(yva, pc, zero_division=0)),
                "auroc_ckks_val": float(roc_auc_score(yva, sim)),
                "auroc_delta_plain_minus_ckks": float(auc_full - roc_auc_score(yva, sim)),
                "prob_abs_error_mean": float(dp.mean()), "prob_abs_error_max": float(dp.max()),
            }
        d1k = (w @ rates[1][1]) / n_boot_stays - (w @ rates[9][1]) / n_boot_stays
        diff_point = float(1000 * (rates[1][0].mean() - rates[9][0].mean()))
        ci = [float(1000 * np.percentile(d1k, 2.5)), float(1000 * np.percentile(d1k, 97.5))]
        p3 = bool(diff_point >= 1.0 and ci[0] > 0.0)
        variant_out[vname] = {
            "a_half_width": a, "interval_quantiles": {str(k): v for k, v in quant.items()},
            "anchor_candidates_k5_val": cand, "selected_anchor_percentile": win["percentile"], "gamma": gamma,
            "tie_break_used": len(tied) > 1, "cells": cells,
            "k1_minus_k9_disagree_per_1000": diff_point, "k1_minus_k9_boot95_per_1000": ci, "P3": p3,
            "runtime_seconds": time.perf_counter() - t0,
        }
        print(f"{vname}: a={a:.4f} anchor=p{win['percentile']} P3={p3} "
              + " ".join(f"k{k}={cells[f'k={k}']['disagree_per_1000']:.1f}" for k in DEGREES), flush=True)

    out = {
        "protocol": "docs/m1c_pilot_prespec.md", "backend": "SIMULATED (circuit-faithful NumPy)",
        "scope": "seed 42, one outcome (M2 critical outcome), train and validation splits only",
        "flow": flow, "n_train_stays": int(len(tr)), "n_val_stays": int(len(va)), "n_val_subjects": int(n_subj),
        "median_imputation_values_used": "training medians (not printed per patient; per-feature scalars)",
        "data_realism": d_checks, "positive_control": positive, "variants": variant_out,
        "missingness_by_outcome_train": {
            c: {"outcome0": float(tr.loc[tr[mc.OUTCOME_COL] == 0, c].isna().mean()),
                "outcome1": float(tr.loc[tr[mc.OUTCOME_COL] == 1, c].isna().mean())} for c in VITAL_FEATURES},
        "runtime_seconds": time.perf_counter() - t_start,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"P1={p1} P2={p2} (AUROC full {auc_full:.4f}, age-only {auc_age:.4f}); D1={d1} D2={d2} D3={d3}")
    print(f"written -> {OUT} ({out['runtime_seconds']:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
