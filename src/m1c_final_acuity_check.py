#!/usr/bin/env python3
"""
Final run, step 1 (Analysis-plan gate in GATES.md): decide whether triage acuity is added to the inputs. VALIDATION ONLY;
the test split is not loaded. Rule fixed in the gate: add acuity if it raises the 5-bin model's validation AUROC (seed 42)
by at least 0.01. Writes results/m1c_final_acuity_decision.json, which the final scripts require before they read any test row.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import m1c_final_common as fc  # noqa: E402

THRESHOLD_GAIN = 0.01


def main() -> int:
    fc.require_signed_gates()
    fr = fc.load_frames(use_acuity=True, need_test=False)
    assert fr["te"] is None
    base_feats = [c for c in fr["feats"] if c != fc.ACUITY]
    res = {}
    for name, feats in (("without_acuity", base_feats), ("with_acuity", fr["feats"])):
        _, binner = fc.make_binner(fr["Xtr"], feats, fc.B_PRIMARY)
        Xtr, Xva = binner(fr["Xtr"][feats]), binner(fr["Xva"][feats])
        clf = fc.fit_xgb(Xtr, fr["ytr"], seed=42)
        margin = clf.get_booster().predict(xgb.DMatrix(Xva, feature_names=feats), output_margin=True)
        ops = fc.operating_points(fr["yva"], margin)
        res[name] = {"auroc_val": float(roc_auc_score(fr["yva"], margin)), "f1_at_f1_threshold_val":
                     fc.decision_metrics(fr["yva"], margin, ops["f1"])["f1"]}
        print(f"{name}: validation AUROC {res[name]['auroc_val']:.4f}, F1 {res[name]['f1_at_f1_threshold_val']:.3f}", flush=True)
    gain = res["with_acuity"]["auroc_val"] - res["without_acuity"]["auroc_val"]
    out = {"rule": f"add triage acuity if the 5-bin validation AUROC (seed 42) rises by at least {THRESHOLD_GAIN}",
           "bins": fc.B_PRIMARY, "seed": 42, "split_used": "train and validation only; test not loaded",
           "results": res, "auroc_gain": float(gain), "use_acuity": bool(gain >= THRESHOLD_GAIN),
           "acuity_missing_rate_train": float(fr["tr"][fc.ACUITY].isna().mean())}
    os.makedirs(fc.RESULTS, exist_ok=True)
    with open(fc.DECISION_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"acuity gain {gain:.4f} -> use_acuity = {out['use_acuity']}; written {fc.DECISION_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
