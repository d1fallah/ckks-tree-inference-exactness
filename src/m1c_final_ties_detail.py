#!/usr/bin/env python3
"""
Added analysis (docs/m1c_final_ties_detail_note.md, committed before this file existed): numerical characterization of the first final run's tie decisions at the
80%-sensitivity threshold. Reads the saved 5-bin models and the test split; aggregate output only (no per-stay value): results/m1c_final_ties_detail.json.
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

OUT = os.path.join(fc.RESULTS, "m1c_final_ties_detail.json")
TIE_TOL = 1e-9


def main() -> int:
    fc.require_signed_gates()
    with open(fc.DECISION_PATH, "r", encoding="utf-8") as f:
        use_acuity = bool(json.load(f)["use_acuity"])
    fr = fc.load_frames(use_acuity, need_test=True)
    feats = fr["feats"]
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
        Xte_b = binned(fr["Xte"])
        clf = xgb.XGBClassifier()
        clf.load_model(os.path.join(fc.ARTIFACT_DIR, f"seed{seed}_q5.json"))
        bst = clf.get_booster()
        m_te = bst.predict(xgb.DMatrix(Xte_b, feature_names=feats), output_margin=True)
        so, ss, norm, coeffs, k = build_exact_circuit(bst, feats, par["B"])
        bl = cc.prepare_batch_levels(ss, feats, norm, 1.0)
        sim = simulate(bl, coeffs, ss.leaf_value, Xte_b.to_numpy(dtype=np.float64), 1.0)
        thr = par["ops"]["sens80"]
        d = sim - m_te
        diff = (m_te >= thr) != (sim >= thr)
        at_thr = np.abs(m_te - thr) <= TIE_TOL
        sd = d[diff]
        per_seed[str(seed)] = {
            "decision_convention": "positive when margin >= threshold",
            "threshold": float(thr),
            "max_abs_margin_difference_whole_test_split": float(np.max(np.abs(d))),
            "mean_abs_margin_difference_whole_test_split": float(np.mean(np.abs(d))),
            "n_differing": int(diff.sum()),
            "n_differing_plain_positive_circuit_negative": int((diff & (m_te >= thr)).sum()),
            "n_differing_plain_negative_circuit_positive": int((diff & (m_te < thr)).sum()),
            "min_signed_difference_among_differing": float(sd.min()) if len(sd) else None,
            "max_signed_difference_among_differing": float(sd.max()) if len(sd) else None,
            "n_stays_with_margin_equal_to_threshold": int(at_thr.sum()),
            "of_those_circuit_positive": int((at_thr & (sim >= thr)).sum()),
            "of_those_circuit_negative": int((at_thr & (sim < thr)).sum()),
        }
        r = per_seed[str(seed)]
        print(f"seed {seed}: max |sim - plain| {r['max_abs_margin_difference_whole_test_split']:.2e}; differing {r['n_differing']}; at threshold "
              f"{r['n_stays_with_margin_equal_to_threshold']} (circuit positive {r['of_those_circuit_positive']}, negative {r['of_those_circuit_negative']})", flush=True)
    out = {"note": "docs/m1c_final_ties_detail_note.md (added analysis)", "tolerance": TIE_TOL, "per_seed": per_seed,
           "max_abs_margin_difference_all_seeds": float(max(v["max_abs_margin_difference_whole_test_split"] for v in per_seed.values()))}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"largest |sim - plain| over seeds {out['max_abs_margin_difference_all_seeds']:.2e}; written {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
