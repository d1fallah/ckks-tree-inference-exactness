#!/usr/bin/env python3
"""
Final run (Analysis-plan gate in GATES.md): five seeds, test split read here for the first time. Simulated circuits.
  (i)   primary: 5 quantile bins per input, exact comparison of degree 9 (existing circuit, folding evaluator in the
        real-ciphertext script), simulated on the whole test split;
  (ii)  floor reference: full-precision model with the approximate circuit at k in {1, 3, 5, 7, 9};
  (iii) sensitivity: 8 bins, exact degree 15 (simulation only).
Aggregate output only: results/m1c_final_results.json. Per-seed model artifacts (booster JSON, public bin table, thresholds) go to
data/final/ (git-ignored) for m1c_final_real.py.
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
from sklearn.metrics import f1_score, roc_auc_score

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import ckks_circuit as cc  # noqa: E402
import m1c_final_common as fc  # noqa: E402
from generate_poly_coefficients import SIGMOID_EDGE_LEVEL  # noqa: E402
from m1c_pilot2 import pooled_abs_z  # noqa: E402
from m1c_pilot_validity import simulate  # noqa: E402
from select_gamma_anchor import fit_power_coefficients  # noqa: E402

OUT = os.path.join(fc.RESULTS, "m1c_final_results.json")
DEGREES_FLOOR = (1, 3, 5, 7, 9)
GAMMA_GRID = [1, 5, 25]
TIE_TOLERANCE = 0.002
CHUNK = 4000


def build_exact_circuit(booster, feats, B):
    so = cc.extract_ensemble_leaf_circuit(booster, set(feats), tree_depth=6)
    ss = cc.TreeCircuitStructure(
        n_leaves=so.n_leaves, leaf_value=so.leaf_value,
        levels=[[dataclasses.replace(r, threshold=fc.shifted_threshold(r.threshold)) if r.kind != "pad" else r
                 for r in lvl] for lvl in so.levels])
    norm = {c: {"min": 0.0, "max": B - 0.5} for c in feats}
    k = 2 * B - 1
    h = fc.exact_odd_step(B)
    coeffs = cc.assemble_coefficient_vectors(ss, {f"k={k}": {"coefficients_power_basis": [float(v) for v in h]}}, k)
    return so, ss, norm, coeffs, k


def hard_margin(st, blv, X, yes, pad):
    out = np.empty(X.shape[0])
    for s in range(0, X.shape[0], CHUNK):
        Xc = X[s : s + CHUNK]
        prod = np.ones((Xc.shape[0], st.n_leaves))
        for lvl, y_, p_ in zip(blv, yes, pad):
            z = (Xc[:, lvl.feature_col_idx] - lvl.threshold[None, :]) / lvl.span[None, :]
            h = (z < 0).astype(float)
            prod *= np.where(p_[None, :], 1.0, np.where(y_[None, :], h, 1.0 - h))
        out[s : s + CHUNK] = prod @ st.leaf_value
    return out


def exact_arm(label, B, feats, fr, seed, w, codes_te, n_boot_stays, n_te_subj, save=False):
    """One quantized model and its exact circuit. Returns (summary dict, plain test margin, ops)."""
    edges, binner = fc.make_binner(fr["Xtr"], feats, B)
    Xtr_b, Xva_b, Xte_b = binner(fr["Xtr"]), binner(fr["Xva"]), binner(fr["Xte"])
    clf = fc.fit_xgb(Xtr_b, fr["ytr"], seed)
    bst = clf.get_booster()
    m_va = bst.predict(xgb.DMatrix(Xva_b, feature_names=feats), output_margin=True)
    m_te = bst.predict(xgb.DMatrix(Xte_b, feature_names=feats), output_margin=True)
    ops = fc.operating_points(fr["yva"], m_va)
    so, ss, norm, coeffs, k = build_exact_circuit(bst, feats, B)
    bl_o = cc.prepare_batch_levels(so, feats, norm, 1.0)
    bl_s = cc.prepare_batch_levels(ss, feats, norm, 1.0)
    yes = [np.array([r.kind != "pad" and r.is_yes_edge for r in lvl]) for lvl in so.levels]
    pad = [np.array([r.kind == "pad" for r in lvl]) for lvl in so.levels]
    Xva_mat = Xva_b.to_numpy(dtype=np.float64)
    m_o, m_s = hard_margin(so, bl_o, Xva_mat, yes, pad), hard_margin(ss, bl_s, Xva_mat, yes, pad)
    assert np.max(np.abs(m_o - m_s)) <= 1e-9, f"{label}: threshold shift is not lossless"
    assert np.max(np.abs(m_s - m_va)) <= 1e-4, f"{label}: exact-step circuit differs from the booster"
    sim = simulate(bl_s, coeffs, ss.leaf_value, Xte_b.to_numpy(dtype=np.float64), 1.0)
    yte = fr["yte"]
    out = {"B": B, "degree": k, "bins_per_feature": {c: int(Xtr_b[c].max()) + 1 for c in feats},
           "operating_points_margin": ops, "utility_plain_test": fc.utility(yte, m_te, ops),
           "circuit_vs_plain": {}}
    for name in ("f1", "sens80"):
        ag = fc.agreement(m_te, sim, ops[name])
        d = (m_te >= ops[name]) != (sim >= ops[name])
        per_subj = np.bincount(codes_te[d], minlength=n_te_subj).astype(np.int64)
        boot = 1000 * (w @ per_subj) / n_boot_stays
        ag["differing_per_1000_boot95"] = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
        out["circuit_vs_plain"][name] = ag
    out["circuit_auroc_test"] = float(roc_auc_score(yte, sim))
    if save:
        os.makedirs(fc.ARTIFACT_DIR, exist_ok=True)
        clf.save_model(os.path.join(fc.ARTIFACT_DIR, f"seed{seed}_{label}.json"))
        with open(os.path.join(fc.ARTIFACT_DIR, f"seed{seed}_{label}_params.json"), "w", encoding="utf-8") as f:
            json.dump({"B": B, "feats": feats, "edges": {c: [float(x) for x in e] for c, e in edges.items()},
                       "medians": fr["medians"], "ops": ops, "use_acuity": fr["use_acuity"]}, f)
    return out, m_te, ops, (Xte_b, m_va)


def main() -> int:
    t_start = time.perf_counter()
    fc.require_signed_gates()
    with open(fc.DECISION_PATH, "r", encoding="utf-8") as f:
        decision = json.load(f)
    use_acuity = bool(decision["use_acuity"])
    fr = fc.load_frames(use_acuity, need_test=True)
    feats, yte = fr["feats"], fr["yte"]
    print(f"cohort {fr['flow']}; inputs {feats}; use_acuity={use_acuity}", flush=True)
    w, codes_te, n_te_subj = fc.subject_weights(fr["te"]["subject_id"])
    n_boot_stays = w @ np.bincount(codes_te, minlength=n_te_subj).astype(np.int64)
    codes_va, uniq_va = pd.factorize(fr["va"]["subject_id"])
    keep = np.random.default_rng(42).choice(len(uniq_va), size=int(round(fc.SCORE_FRACTION * len(uniq_va))), replace=False)
    sub = np.isin(codes_va, keep)
    Xva_mat_raw = fr["Xva"].to_numpy(dtype=np.float64)
    Xte_mat_raw = fr["Xte"].to_numpy(dtype=np.float64)
    norm_B = {c: {"min": fc.LIMITS[c][0], "max": fc.LIMITS[c][1]} for c in feats}

    seeds_out = {}
    for seed in fc.SEEDS:
        t_s = time.perf_counter()
        S = {}
        # ---- full-precision model: utility and floor reference
        clf_fp = fc.fit_xgb(fr["Xtr"], fr["ytr"], seed)
        bst_fp = clf_fp.get_booster()
        mva_fp = bst_fp.predict(xgb.DMatrix(fr["Xva"], feature_names=feats), output_margin=True)
        mte_fp = bst_fp.predict(xgb.DMatrix(fr["Xte"], feature_names=feats), output_margin=True)
        ops_fp = fc.operating_points(fr["yva"], mva_fp)
        S["full_precision"] = {"operating_points_margin": ops_fp, "utility_test": fc.utility(yte, mte_fp, ops_fp)}
        # ---- primary: exact comparison on 5 bins
        prim, mte_q, ops_q, _ = exact_arm("q5", fc.B_PRIMARY, feats, fr, seed, w, codes_te, n_boot_stays, n_te_subj, save=True)
        S["primary_exact_B5"] = prim
        # utility bootstrap (plain models) and quantization cost, separate from circuit error
        auc_q, auc_fp, f1_q, f1_fp = [], [], [], []
        pred_q = (mte_q >= ops_q["f1"]).astype(int)
        pred_fp = (mte_fp >= ops_fp["f1"]).astype(int)
        for b in range(fc.N_BOOT):
            wt = w[b][codes_te].astype(float)
            auc_q.append(roc_auc_score(yte, mte_q, sample_weight=wt))
            auc_fp.append(roc_auc_score(yte, mte_fp, sample_weight=wt))
            f1_q.append(f1_score(yte, pred_q, sample_weight=wt, zero_division=0))
            f1_fp.append(f1_score(yte, pred_fp, sample_weight=wt, zero_division=0))
        auc_q, auc_fp, f1_q, f1_fp = map(np.asarray, (auc_q, auc_fp, f1_q, f1_fp))
        ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]  # noqa: E731
        S["quantization_cost"] = {
            "auroc_full_precision": float(roc_auc_score(yte, mte_fp)), "auroc_quantized": float(roc_auc_score(yte, mte_q)),
            "auroc_cost": float(roc_auc_score(yte, mte_fp) - roc_auc_score(yte, mte_q)),
            "auroc_cost_boot95": ci(auc_fp - auc_q), "auroc_quantized_boot95": ci(auc_q), "auroc_full_precision_boot95": ci(auc_fp),
            "f1_full_precision": S["full_precision"]["utility_test"]["at_f1_threshold"]["f1"],
            "f1_quantized": prim["utility_plain_test"]["at_f1_threshold"]["f1"],
            "f1_cost_boot95": ci(f1_fp - f1_q), "f1_quantized_boot95": ci(f1_q), "f1_full_precision_boot95": ci(f1_fp)}
        print(f"seed {seed}: exact B5 differing {prim['circuit_vs_plain']['f1']['n_differing']} (F1 op), "
              f"{prim['circuit_vs_plain']['sens80']['n_differing']} (80% sens op); AUROC full {S['quantization_cost']['auroc_full_precision']:.4f} "
              f"quantized {S['quantization_cost']['auroc_quantized']:.4f} [{time.perf_counter() - t_s:.0f}s]", flush=True)

        # ---- floor reference: approximate circuit on the full-precision model
        so_fp = cc.extract_ensemble_leaf_circuit(bst_fp, set(feats), tree_depth=6)
        _, perfeat = pooled_abs_z(clf_fp, fr["Xtr"], norm_B)
        norm_F = {c: {"min": fc.LIMITS[c][0], "max": fc.LIMITS[c][0] + (fc.LIMITS[c][1] - fc.LIMITS[c][0]) * perfeat[c]}
                  for c in perfeat}
        for c in feats:
            norm_F.setdefault(c, norm_B[c])
        abs_z, _ = pooled_abs_z(clf_fp, fr["Xtr"], norm_F)
        bl_fp = cc.prepare_batch_levels(so_fp, feats, norm_F, 1.0)
        mva_sub, pp_sub = mva_fp[sub], (mva_fp[sub] >= ops_fp["f1"]).astype(int)
        span = float(mva_sub.max() - mva_sub.min()) or 1.0
        floor = {}
        for k in DEGREES_FLOOR:
            cands = []
            for P in GAMMA_GRID:
                z_ref = float(np.percentile(abs_z, P))
                gamma = float(np.log(SIGMOID_EDGE_LEVEL / (1 - SIGMOID_EDGE_LEVEL)) / z_ref) if z_ref > 0 else 1e9
                with np.errstate(all="ignore"):
                    cd = fit_power_coefficients(gamma, 1.0, k)
                co = cc.assemble_coefficient_vectors(so_fp, cd, k)
                sim_sub = simulate(bl_fp, co, so_fp.leaf_value, Xva_mat_raw[sub], 1.0)
                cands.append({"percentile": P, "gamma": gamma, "coef": co,
                              "agree": float(np.mean(pp_sub == (sim_sub >= ops_fp["f1"]).astype(int))),
                              "nmae": float(np.mean(np.abs(sim_sub - mva_sub)) / span)})
            top = max(r["agree"] for r in cands)
            tied = [r for r in cands if top - r["agree"] <= TIE_TOLERANCE]
            win = sorted(tied, key=lambda r: (r["nmae"], -r["gamma"]))[0]
            sim_te = simulate(bl_fp, win["coef"], so_fp.leaf_value, Xte_mat_raw, 1.0)
            cell = {"selected_percentile": win["percentile"], "gamma": win["gamma"], "circuit_auroc_test": float(roc_auc_score(yte, sim_te)),
                    "auroc_delta_plain_minus_circuit": float(roc_auc_score(yte, mte_fp) - roc_auc_score(yte, sim_te)),
                    "vs_plain": {}}
            for name in ("f1", "sens80"):
                ag = fc.agreement(mte_fp, sim_te, ops_fp[name])
                d = (mte_fp >= ops_fp[name]) != (sim_te >= ops_fp[name])
                per_subj = np.bincount(codes_te[d], minlength=n_te_subj).astype(np.int64)
                boot = 1000 * (w @ per_subj) / n_boot_stays
                ag["differing_per_1000_boot95"] = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
                cell["vs_plain"][name] = ag
            floor[f"k={k}"] = cell
            print(f"seed {seed} floor k={k}: {cell['vs_plain']['f1']['differing_per_1000']:.1f}/1000, lost "
                  f"{cell['vs_plain']['f1']['lost_share_of_plain_positives']:.3f} [{time.perf_counter() - t_s:.0f}s]", flush=True)
        S["floor_reference"] = floor

        # ---- sensitivity: 8 bins, exact degree 15 (simulation only)
        sens, _, _, _ = exact_arm("q8", fc.B_SENS, feats, fr, seed, w, codes_te, n_boot_stays, n_te_subj, save=False)
        S["sensitivity_exact_B8"] = sens
        print(f"seed {seed}: B8 exact differing {sens['circuit_vs_plain']['f1']['n_differing']} (F1 op); done [{time.perf_counter() - t_s:.0f}s]", flush=True)
        seeds_out[str(seed)] = S
        with open(OUT.replace(".json", "_partial.json"), "w", encoding="utf-8") as f:
            json.dump(seeds_out, f)

    # ---- aggregates over the five seeds
    def over(path):
        vals = []
        for s in seeds_out.values():
            v = s
            for p in path:
                v = v[p]
            vals.append(v)
        return fc.ms(vals)
    agg = {
        "primary_exact_B5": {
            "differing_per_1000_f1_op": over(["primary_exact_B5", "circuit_vs_plain", "f1", "differing_per_1000"]),
            "n_differing_f1_op": over(["primary_exact_B5", "circuit_vs_plain", "f1", "n_differing"]),
            "n_differing_sens80_op": over(["primary_exact_B5", "circuit_vs_plain", "sens80", "n_differing"]),
            "max_abs_margin_difference_f1_op": over(["primary_exact_B5", "circuit_vs_plain", "f1", "max_abs_margin_difference"]),
            "auroc_test": over(["primary_exact_B5", "utility_plain_test", "auroc"]),
            "auprc_test": over(["primary_exact_B5", "utility_plain_test", "auprc"]),
            "f1_f1_op": over(["primary_exact_B5", "utility_plain_test", "at_f1_threshold", "f1"]),
            "sensitivity_sens80_op": over(["primary_exact_B5", "utility_plain_test", "at_sens80_threshold", "sensitivity"]),
            "specificity_sens80_op": over(["primary_exact_B5", "utility_plain_test", "at_sens80_threshold", "specificity"]),
            "ppv_sens80_op": over(["primary_exact_B5", "utility_plain_test", "at_sens80_threshold", "ppv"]),
        },
        "full_precision": {"auroc_test": over(["full_precision", "utility_test", "auroc"]),
                           "auprc_test": over(["full_precision", "utility_test", "auprc"]),
                           "f1_f1_op": over(["full_precision", "utility_test", "at_f1_threshold", "f1"])},
        "quantization_cost": {"auroc_cost": over(["quantization_cost", "auroc_cost"]), "f1_cost_pairs": None},
        "floor_reference": {k: {"differing_per_1000_f1_op": over(["floor_reference", k, "vs_plain", "f1", "differing_per_1000"]),
                                "lost_share_f1_op": over(["floor_reference", k, "vs_plain", "f1", "lost_share_of_plain_positives"]),
                                "auroc_delta": over(["floor_reference", k, "auroc_delta_plain_minus_circuit"])}
                            for k in (f"k={d}" for d in DEGREES_FLOOR)},
        "sensitivity_exact_B8": {"n_differing_f1_op": over(["sensitivity_exact_B8", "circuit_vs_plain", "f1", "n_differing"]),
                                 "auroc_test": over(["sensitivity_exact_B8", "utility_plain_test", "auroc"])},
    }
    exact_all = all(s["primary_exact_B5"]["circuit_vs_plain"][n]["n_differing"] == 0 for s in seeds_out.values()
                    for n in ("f1", "sens80"))
    out = {"protocol": "Analysis-plan gate in GATES.md (M1 continuous-feature task)", "backend": "SIMULATED (circuit-faithful NumPy)",
           "cohort_flow": fr["flow"], "inputs": feats, "acuity_decision": decision,
           "n_test_stays": int(len(fr["te"])), "n_test_subjects": int(n_te_subj), "seeds": list(fc.SEEDS),
           "bootstrap": f"patient-cluster, {fc.N_BOOT} resamples of test subjects, fixed generator (seed {fc.BOOT_SEED}); reported separately from seed SD",
           "exact_in_simulation_all_seeds_both_operating_points": bool(exact_all),
           "per_seed": seeds_out, "aggregate": agg, "runtime_seconds": time.perf_counter() - t_start}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"exact in simulation for all seeds and both operating points: {exact_all}; written -> {OUT} ({out['runtime_seconds']:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
