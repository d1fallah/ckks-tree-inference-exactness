#!/usr/bin/env python3
"""
Seventh pilot for the M1 continuous-feature task (added analysis): exact comparison on quantized inputs. Executes
docs/m1c_pilot7_prespec.md (committed before this file existed). Train and validation splits only; the test split is never
selected. Simulation for B in {5, 6, 8}; real TenSEAL ciphertexts for B = 5 at k = 9 (folding evaluator) on 300 validation stays;
a depth probe of TenSEAL polyval on synthetic input. Aggregate output only: results/m1c_pilot7.json.
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import tenseal as ts
import xgboost as xgb
from sklearn.metrics import f1_score, roc_auc_score

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import ckks_circuit as cc  # noqa: E402
import m1c_cohort as mc  # noqa: E402
from m1c_pilot_validity import fit_model, simulate  # noqa: E402
from splits import patient_level_split  # noqa: E402
from train_plaintext import tune_thresholds  # noqa: E402

OUT = os.path.join(_HERE, "..", "results", "m1c_pilot7.json")
PILOT1 = os.path.join(_HERE, "..", "results", "m1c_pilot_validity.json")
LEVELS = (5, 6, 8)
REAL_B, REAL_PER_CLASS, REAL_GATE = 5, 150, 5e-4
N_VALUE, COEFF_MOD_BIT_SIZES, SCALE = 16384, [60, 40, 40, 40, 40, 40, 40, 40, 60], 2**40
SEED = 42
CHUNK = 4000


def shifted_threshold(t: float) -> float:
    return math.ceil(t - 1e-6) - 0.5


def exact_odd_step(B: int) -> np.ndarray:
    """Power-basis coefficients (degree 2B-1) of h = 0.5 - 0.5 p, with p odd and p = 1 at (j + 0.5)/(B - 0.5), j = 0..B-1."""
    u = (np.arange(B) + 0.5) / (B - 0.5)
    powers = np.arange(1, 2 * B, 2)
    c = np.linalg.solve(u[:, None] ** powers[None, :], np.ones(B))
    p = np.zeros(2 * B)
    p[powers] = c
    h = -0.5 * p
    h[0] += 0.5
    hv = np.polynomial.polynomial.polyval
    assert np.max(np.abs(hv(u, h))) < 1e-9 and np.max(np.abs(hv(-u, h) - 1.0)) < 1e-9, "interpolant is not exact"
    return h


def make_ctx():
    ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=N_VALUE, coeff_mod_bit_sizes=COEFF_MOD_BIT_SIZES)
    ctx.global_scale = SCALE
    ctx.generate_galois_keys()
    return ctx


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
    auc_fp, f1_fp = p1["positive_control"]["auroc_full_val"], p1["positive_control"]["f1_plain_val"]

    out_levels, real_out = {}, None
    for B in LEVELS:
        t_b = time.perf_counter()
        edges = {c: np.unique(np.quantile(Xtr_raw[c].to_numpy(dtype=float), np.arange(1, B) / B)) for c in feats}

        def binned(X):
            return pd.DataFrame({c: np.searchsorted(edges[c], X[c].to_numpy(dtype=float), side="right").astype(float)
                                 for c in feats})
        Xtr, Xva = binned(Xtr_raw), binned(Xva_raw)
        clf = fit_model(Xtr, ytr)
        booster = clf.get_booster()
        plain = booster.predict(xgb.DMatrix(Xva, feature_names=feats), output_margin=True)
        auc = float(roc_auc_score(yva, plain))
        auc_age = float(roc_auc_score(yva, fit_model(Xtr[["age"]], ytr).predict_proba(Xva[["age"]])[:, 1]))
        proba = 1.0 / (1.0 + np.exp(-plain))
        thr = tune_thresholds(yva.reshape(-1, 1), proba.reshape(-1, 1))[0]
        thr_margin = float(np.log(thr / (1.0 - thr)))
        pp = (plain >= thr_margin).astype(int)
        f1_plain = float(f1_score(yva, pp, zero_division=0))

        so = cc.extract_ensemble_leaf_circuit(booster, set(feats), tree_depth=6)
        ss = cc.TreeCircuitStructure(
            n_leaves=so.n_leaves, leaf_value=so.leaf_value,
            levels=[[dataclasses.replace(r, threshold=shifted_threshold(r.threshold)) if r.kind != "pad" else r
                     for r in lvl] for lvl in so.levels])
        norm = {c: {"min": 0.0, "max": B - 0.5} for c in feats}
        bl_o = cc.prepare_batch_levels(so, feats, norm, 1.0)
        bl_s = cc.prepare_batch_levels(ss, feats, norm, 1.0)
        yes = [np.array([r.kind != "pad" and r.is_yes_edge for r in lvl]) for lvl in so.levels]
        pad = [np.array([r.kind == "pad" for r in lvl]) for lvl in so.levels]
        Xva_mat = Xva.to_numpy(dtype=np.float64)

        def hard_margin(st, blv, X):
            o = np.empty(X.shape[0])
            for s in range(0, X.shape[0], CHUNK):
                Xc = X[s : s + CHUNK]
                prod = np.ones((Xc.shape[0], st.n_leaves))
                for lvl, y_, p_ in zip(blv, yes, pad):
                    z = (Xc[:, lvl.feature_col_idx] - lvl.threshold[None, :]) / lvl.span[None, :]
                    h = (z < 0).astype(float)
                    prod *= np.where(p_[None, :], 1.0, np.where(y_[None, :], h, 1.0 - h))
                o[s : s + CHUNK] = prod @ st.leaf_value
            return o

        m_o, m_s = hard_margin(so, bl_o, Xva_mat), hard_margin(ss, bl_s, Xva_mat)
        assert np.max(np.abs(m_o - m_s)) <= 1e-9, f"B={B}: threshold shift is not lossless"
        assert np.max(np.abs(m_s - plain)) <= 1e-4, f"B={B}: exact-step circuit differs from the booster"
        assert np.array_equal(m_s >= thr_margin, plain >= thr_margin), f"B={B}: exact-step decisions differ"

        k = 2 * B - 1
        h = exact_odd_step(B)
        coeffs = cc.assemble_coefficient_vectors(ss, {f"k={k}": {"coefficients_power_basis": [float(v) for v in h]}}, k)
        sim = simulate(bl_s, coeffs, ss.leaf_value, Xva_mat, 1.0)
        diff = (sim >= thr_margin) != (plain >= thr_margin)
        out_levels[f"B={B}"] = {
            "degree": k, "max_abs_power_coefficient": float(np.max(np.abs(h))),
            "auroc_val": auc, "auroc_age_only_val": auc_age, "P1": bool(auc >= 0.60), "P2": bool(auc - auc_age >= 0.03),
            "auroc_cost_vs_full_precision": auc_fp - auc, "f1_plain_val": f1_plain, "f1_cost_vs_full_precision": f1_fp - f1_plain,
            "bins_per_feature": {c: int(Xtr[c].max()) + 1 for c in feats},
            "sim_max_abs_margin_minus_booster": float(np.max(np.abs(sim - plain))),
            "sim_n_decisions_differing": int(diff.sum()), "n_val_stays": int(len(va)),
            "runtime_seconds": time.perf_counter() - t_b,
        }
        r = out_levels[f"B={B}"]
        print(f"B={B} k={k}: AUROC {auc:.4f} (cost {r['auroc_cost_vs_full_precision']:.4f}, age-only {auc_age:.4f}); "
              f"max|coef| {r['max_abs_power_coefficient']:.1f}; sim max|margin-booster| {r['sim_max_abs_margin_minus_booster']:.2e}; "
              f"differing decisions {r['sim_n_decisions_differing']}", flush=True)

        if B == REAL_B:
            rng = np.random.default_rng(SEED)
            pos, neg = np.flatnonzero(pp == 1), np.flatnonzero(pp == 0)
            idx = np.concatenate([rng.choice(pos, REAL_PER_CLASS, replace=False), rng.choice(neg, REAL_PER_CLASS, replace=False)])
            ctx = make_ctx()
            errs, agree, secs = [], 0, []
            for n_done, i in enumerate(idx, 1):
                x_row = Xva.iloc[i]
                t0 = time.perf_counter()
                real, _ = cc.evaluate_circuit_tenseal(ctx, ss, coeffs, x_row, norm, 1.0, poly_evaluator=cc.eval_poly_tenseal_folded)
                secs.append(time.perf_counter() - t0)
                errs.append(abs(real - sim[i]))
                agree += int((real >= thr_margin) == (plain[i] >= thr_margin))
                if n_done % 50 == 0:
                    print(f"  real B={B}: {n_done}/{len(idx)} done, max err so far {max(errs):.2e}, "
                          f"agree {agree}/{n_done}, {np.mean(secs):.2f}s/stay", flush=True)
            real_out = {"B": B, "degree": k, "evaluator": "eval_poly_tenseal_folded", "n_stays": int(len(idx)),
                        "sample": f"{REAL_PER_CLASS} plaintext-positive + {REAL_PER_CLASS} plaintext-negative validation stays, generator seed {SEED}",
                        "decisions_agreeing_with_booster": agree,
                        "max_abs_real_minus_sim": float(max(errs)), "mean_abs_real_minus_sim": float(np.mean(errs)),
                        "gate_tolerance": REAL_GATE, "gate_passed": bool(max(errs) <= REAL_GATE),
                        "mean_seconds_per_stay": float(np.mean(secs)), "sd_seconds_per_stay": float(np.std(secs, ddof=1))}
            print(f"real B={B}: gate {'PASSED' if real_out['gate_passed'] else 'FAILED'} (max err {real_out['max_abs_real_minus_sim']:.2e}); "
                  f"agree {agree}/{len(idx)}; {real_out['mean_seconds_per_stay']:.2f}s/stay", flush=True)
        with open(OUT.replace(".json", "_partial.json"), "w", encoding="utf-8") as f:
            json.dump({"levels": out_levels, "real": real_out}, f)

    # ---- depth probe of TenSEAL polyval (synthetic input, no patient data)
    probe = {}
    ctx = make_ctx()
    xs = np.random.default_rng(SEED).uniform(-1.0, 1.0, 4096)
    for B in (6, 8):
        h = exact_odd_step(B)
        entry = {"degree": 2 * B - 1}
        try:
            y = ts.ckks_vector(ctx, xs.tolist()).polyval([float(v) for v in h])
            entry["max_abs_error_vs_numpy"] = float(np.max(np.abs(np.asarray(y.decrypt()) - np.polynomial.polynomial.polyval(xs, h))))
            n_sq = 0
            while n_sq < 10:
                try:
                    y = y * y
                    y.decrypt()
                    n_sq += 1
                except Exception:
                    break
            entry["squarings_after_polyval"] = n_sq
            entry["levels_used_by_polyval"] = 7 - n_sq
        except Exception as exc:
            entry["error"] = str(exc)[:300]
        probe[f"k={2 * B - 1}"] = entry
        print(f"polyval probe degree {2 * B - 1}: {entry}", flush=True)

    out = {"protocol": "docs/m1c_pilot7_prespec.md", "scope": "seed 42, train and validation splits only, test split not read",
           "full_precision_reference": {"auroc_val": auc_fp, "f1_val": f1_fp},
           "levels": out_levels, "real_ciphertext_check": real_out, "tenseal_polyval_depth_probe": probe,
           "runtime_seconds": time.perf_counter() - t_start}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"written -> {OUT} ({out['runtime_seconds']:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
