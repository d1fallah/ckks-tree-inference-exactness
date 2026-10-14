#!/usr/bin/env python3
"""
Added analysis (docs/m1c_final_ties_note.md, committed before this file existed): are the stays that differ between the exact 5-bin circuit and the
plaintext model at the 80%-sensitivity operating point boundary ties? Per seed, on the test split, with the saved models and public bin tables of the
final run. Aggregate output only: results/m1c_final_ties.json.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import xgboost as xgb

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import ckks_circuit as cc  # noqa: E402
import m1c_final_common as fc  # noqa: E402
from m1c_final_run import build_exact_circuit  # noqa: E402
from m1c_pilot_validity import simulate  # noqa: E402

OUT = os.path.join(fc.RESULTS, "m1c_final_ties.json")
TIE_TOL = 1e-9


def main() -> int:
    fc.require_signed_gates()
    with open(fc.DECISION_PATH, "r", encoding="utf-8") as f:
        use_acuity = bool(json.load(f)["use_acuity"])
    fr = fc.load_frames(use_acuity, need_test=True)
    feats, yva, yte = fr["feats"], fr["yva"], fr["yte"]
    per_seed = {}
    for seed in fc.SEEDS:
        with open(os.path.join(fc.ARTIFACT_DIR, f"seed{seed}_q5_params.json"), "r", encoding="utf-8") as f:
            par = json.load(f)
        edges = {c: np.asarray(e) for c, e in par["edges"].items()}

        def binned(X):
            cols = {}
            for c in feats:
                v = X[c].to_numpy(dtype=float)
                cols[c] = (np.clip(np.round(v) - 1, 0, 4) if c == fc.ACUITY else np.searchsorted(edges[c], v, side="right").astype(float))
            return pd.DataFrame(cols)
        Xva_b, Xte_b = binned(fr["Xva"]), binned(fr["Xte"])
        clf = xgb.XGBClassifier()
        clf.load_model(os.path.join(fc.ARTIFACT_DIR, f"seed{seed}_q5.json"))
        bst = clf.get_booster()
        m_va = bst.predict(xgb.DMatrix(Xva_b, feature_names=feats), output_margin=True)
        m_te = bst.predict(xgb.DMatrix(Xte_b, feature_names=feats), output_margin=True)
        so, ss, norm, coeffs, k = build_exact_circuit(bst, feats, par["B"])
        bl = cc.prepare_batch_levels(ss, feats, norm, 1.0)
        sim = simulate(bl, coeffs, ss.leaf_value, Xte_b.to_numpy(dtype=np.float64), 1.0)
        res = {}
        for name in ("f1", "sens80"):
            thr = par["ops"][name]
            diff = (m_te >= thr) != (sim >= thr)
            at_thr = np.abs(m_te - thr) <= TIE_TOL
            res[name] = {"threshold": float(thr), "n_test_stays_with_margin_equal_to_threshold": int(at_thr.sum()),
                         "n_differing": int(diff.sum()), "n_differing_that_are_ties": int((diff & at_thr).sum()),
                         "n_differing_not_ties": int((diff & ~at_thr).sum())}
        thr = par["ops"]["sens80"]
        below = np.unique(m_va[m_va < thr - TIE_TOL])
        thr2 = float((thr + below.max()) / 2.0) if len(below) else float(thr - 1e-3)
        diff2 = (m_te >= thr2) != (sim >= thr2)
        pos = yte == 1
        res["sens80_tie_free"] = {
            "threshold": thr2, "n_differing": int(diff2.sum()),
            "n_test_stays_with_margin_within_1e-6_of_threshold": int((np.abs(m_te - thr2) <= 1e-6).sum()),
            "sensitivity_test_at_run_threshold": float(((m_te >= thr) & pos).sum() / pos.sum()),
            "sensitivity_test_at_tie_free_threshold": float(((m_te >= thr2) & pos).sum() / pos.sum()),
            "sensitivity_validation_at_tie_free_threshold": float(((m_va >= thr2) & (yva == 1)).sum() / (yva == 1).sum())}
        per_seed[str(seed)] = res
        print(f"seed {seed}: sens80 differing {res['sens80']['n_differing']} (ties {res['sens80']['n_differing_that_are_ties']}, "
              f"stays at threshold {res['sens80']['n_test_stays_with_margin_equal_to_threshold']}); tie-free threshold differing "
              f"{res['sens80_tie_free']['n_differing']}; F1 op differing {res['f1']['n_differing']}", flush=True)
    all_ties = all(r["sens80"]["n_differing_not_ties"] == 0 and r["f1"]["n_differing_not_ties"] == 0 for r in per_seed.values())
    tie_free_zero = all(r["sens80_tie_free"]["n_differing"] == 0 for r in per_seed.values())
    out = {"note": "docs/m1c_final_ties_note.md (added analysis)", "tolerance": TIE_TOL, "per_seed": per_seed,
           "all_differing_decisions_are_ties": bool(all_ties), "tie_free_threshold_zero_differing_all_seeds": bool(tie_free_zero)}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"all differing are ties: {all_ties}; tie-free zero differing in every seed: {tie_free_zero}; written {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
