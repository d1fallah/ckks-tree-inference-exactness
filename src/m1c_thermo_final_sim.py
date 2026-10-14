#!/usr/bin/env python3
"""
Thermometer final run, stage 1 (docs/m1c_thermo_final_prespec.md): quantized models, midpoint operating points, thermometer circuit simulated on the whole
test split, utility and quantization cost with a patient-cluster bootstrap, exact interpolation (B = 5) simulated with the same thresholds. One seed per call.
Aggregate output only: results/m1c_thermo_final_sim_seed{S}.json. Boosters and operating points go to data/final/ (git-ignored).
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import ckks_circuit as cc  # noqa: E402
import m1c_final_common as fc  # noqa: E402
import m1c_thermo as th  # noqa: E402
import m1c_thermo_final_common as tc  # noqa: E402
from m1c_final_run import build_exact_circuit  # noqa: E402
from m1c_pilot_validity import simulate  # noqa: E402
from test_m1c_thermo import check_every_split  # noqa: E402


def boot_differing(m_ref, m_circ, thr, w, codes, n_subj, n_stays_boot):
    d = (m_ref >= thr) != (m_circ >= thr)
    per_subj = np.bincount(codes[d], minlength=n_subj).astype(np.int64)
    boot = 1000 * (w @ per_subj) / n_stays_boot
    return [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]


def main() -> int:
    a = tc.parse_args()
    seed, smoke = a.seed, a.smoke
    t_start = time.perf_counter()
    tc.require_gates()
    fr = tc.load_frames(smoke)
    feats, yva, yte = fr["feats"], fr["yva"], fr["yte"]
    w, codes, n_subj = fc.subject_weights(fr["te"]["subject_id"])
    n_stays_boot = w @ np.bincount(codes, minlength=n_subj).astype(np.int64)
    os.makedirs(fc.ARTIFACT_DIR, exist_ok=True)

    # full-precision reference
    clf_fp = fc.fit_xgb(fr["Xtr"], fr["ytr"], seed)
    b_fp = clf_fp.get_booster()
    mva_fp = b_fp.predict(xgb.DMatrix(fr["Xva"], feature_names=feats), output_margin=True)
    mte_fp = b_fp.predict(xgb.DMatrix(fr["Xte"], feature_names=feats), output_margin=True)
    ops_fp = fc.operating_points_midpoint(yva, mva_fp)
    S = {"seed": seed, "smoke": smoke, "n_test_stays": int(len(yte)), "n_test_subjects": int(n_subj),
         "full_precision": {"operating_points_margin": ops_fp, "utility_test": fc.utility(yte, mte_fp, ops_fp)}, "by_B": {}}
    auc_fp_boot = np.array([roc_auc_score(yte, mte_fp, sample_weight=w[b][codes].astype(float)) for b in range(fc.N_BOOT)])
    f1_pred_fp = (mte_fp >= ops_fp["f1"]).astype(int)
    print(f"seed {seed}: full precision test AUROC {S['full_precision']['utility_test']['auroc']:.4f}", flush=True)

    for B in tc.LEVELS_B:
        t_b = time.perf_counter()
        edges, binner = fc.make_binner(fr["Xtr"], feats, B)
        Xtr_b, Xva_b, Xte_b = binner(fr["Xtr"]), binner(fr["Xva"]), binner(fr["Xte"])
        clf = fc.fit_xgb(Xtr_b, fr["ytr"], seed)
        bst = clf.get_booster()
        m_va = bst.predict(xgb.DMatrix(Xva_b, feature_names=feats), output_margin=True)
        m_te = bst.predict(xgb.DMatrix(Xte_b, feature_names=feats), output_margin=True)
        ops = fc.operating_points_midpoint(yva, m_va)
        clf.save_model(tc.booster_path(seed, B, smoke))
        with open(tc.params_path(seed, B, smoke), "w", encoding="utf-8") as f:
            json.dump({"B": B, "feats": feats, "ops": ops, "use_acuity": fr["use_acuity"]}, f)
        R = {"B": B, "operating_points_margin": ops, "utility_plain_test": fc.utility(yte, m_te, ops),
             "n_splits_enumerated_ok": check_every_split(bst, B)}
        # quantization cost, separate from circuit error
        auc_q = np.array([roc_auc_score(yte, m_te, sample_weight=w[b][codes].astype(float)) for b in range(fc.N_BOOT)])
        pred_q = (m_te >= ops["f1"]).astype(int)
        from sklearn.metrics import f1_score
        f1_boot = np.array([f1_score(yte, pred_q, sample_weight=w[b][codes].astype(float), zero_division=0)
                            - f1_score(yte, f1_pred_fp, sample_weight=w[b][codes].astype(float), zero_division=0) for b in range(fc.N_BOOT)])
        ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]  # noqa: E731
        R["quantization_cost"] = {"auroc_full_precision": S["full_precision"]["utility_test"]["auroc"], "auroc_quantized": R["utility_plain_test"]["auroc"],
                                  "auroc_cost": S["full_precision"]["utility_test"]["auroc"] - R["utility_plain_test"]["auroc"],
                                  "auroc_cost_boot95": ci(auc_fp_boot - auc_q), "f1_quantized_minus_full_precision_boot95": ci(f1_boot)}
        # thermometer circuit, simulated on the whole test split
        structure = cc.extract_ensemble_leaf_circuit(bst, set(feats), tree_depth=6)
        levels = th.level_arrays(structure, feats, B)
        sim = th.simulate_margin(levels, structure.leaf_value, th.thermometer_bits(Xte_b.to_numpy(dtype=np.float64), B), B)
        R["thermometer_sim_vs_booster"] = {}
        for name in ("f1", "sens80"):
            ag = fc.agreement(m_te, sim, ops[name])
            ag["differing_per_1000_boot95"] = boot_differing(m_te, sim, ops[name], w, codes, n_subj, n_stays_boot)
            R["thermometer_sim_vs_booster"][name] = ag
        R["thermometer_sim_auroc_test"] = float(roc_auc_score(yte, sim))
        R["stays_near_threshold"] = {name: {"min_abs_gap": float(np.min(np.abs(m_te - ops[name]))),
                                            "n_within_gate": int((np.abs(m_te - ops[name]) <= tc.GATE).sum())} for name in ("f1", "sens80")}
        R["stays_near_either_threshold"] = int(((np.abs(m_te - ops["f1"]) <= tc.GATE) | (np.abs(m_te - ops["sens80"]) <= tc.GATE)).sum())
        if B == 5:
            old = os.path.join(fc.ARTIFACT_DIR, f"seed{seed}_q5.json")
            if os.path.exists(old) and not smoke:
                oc = xgb.XGBClassifier()
                oc.load_model(old)
                m_old = oc.get_booster().predict(xgb.DMatrix(Xte_b, feature_names=feats), output_margin=True)
                R["determinism_vs_first_final_run_booster_max_abs_margin_diff"] = float(np.max(np.abs(m_old - m_te)))
            so, ss, norm, coeffs, k = build_exact_circuit(bst, feats, B)
            bl_s = cc.prepare_batch_levels(ss, feats, norm, 1.0)
            sim_i = simulate(bl_s, coeffs, ss.leaf_value, Xte_b.to_numpy(dtype=np.float64), 1.0)
            R["interpolation_sim_vs_booster"] = {"degree": k}
            for name in ("f1", "sens80"):
                ag = fc.agreement(m_te, sim_i, ops[name])
                ag["differing_per_1000_boot95"] = boot_differing(m_te, sim_i, ops[name], w, codes, n_subj, n_stays_boot)
                R["interpolation_sim_vs_booster"][name] = ag
        R["runtime_seconds"] = time.perf_counter() - t_b
        S["by_B"][f"B={B}"] = R
        d = R["thermometer_sim_vs_booster"]
        print(f"seed {seed} B={B}: AUROC {R['utility_plain_test']['auroc']:.4f} (cost {R['quantization_cost']['auroc_cost']:.4f}); thermometer sim differing "
              f"{d['f1']['n_differing']} (F1), {d['sens80']['n_differing']} (80% sens), max margin diff {d['f1']['max_abs_margin_difference']:.1e}; "
              f"stays near a threshold {R['stays_near_either_threshold']} [{R['runtime_seconds']:.0f}s]", flush=True)
    S["runtime_seconds"] = time.perf_counter() - t_start
    with open(tc.out_path("sim", seed, smoke), "w", encoding="utf-8") as f:
        json.dump(S, f, indent=2)
    print(f"written -> {tc.out_path('sim', seed, smoke)} ({S['runtime_seconds']:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
