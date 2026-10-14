"""
Shared pieces of the thermometer final run (Analysis-plan gate "M1 model-private thermometer protocol" in GATES.md; docs/m1c_thermo_final_prespec.md).
Nothing here prints or writes patient-level rows (CLAUDE.md 1.2).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import xgboost as xgb

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import m1c_final_common as fc  # noqa: E402
from m1c_thermo_pilot import CHAIN4, CHAIN7, GATE, key_bytes, make_ctx, summarize  # noqa: E402,F401

LEVELS_B = (5, 8, 16)
PER_CLASS = 50
TITLES = ("Question gate, M1 model-private thermometer protocol",
          "Validity gate, M1 model-private thermometer protocol",
          "Analysis-plan gate, M1 model-private thermometer protocol")


def require_gates() -> None:
    for t in TITLES:
        fc.require_signed_entry(t)


def parse_args(seed_required: bool = True):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=seed_required, default=42)
    ap.add_argument("--smoke", action="store_true", help="engineering check on the validation split, 4 stays, R = 2; not a result")
    a = ap.parse_args()
    assert a.seed in fc.SEEDS, f"seed must be one of {fc.SEEDS}"
    return a


def out_path(stage: str, seed: int | None, smoke: bool) -> str:
    name = f"m1c_thermo_final_{stage}" + (f"_seed{seed}" if seed is not None else "") + ("_smoke" if smoke else "") + ".json"
    return os.path.join(fc.RESULTS, name)


def load_frames(smoke: bool):
    with open(fc.DECISION_PATH, "r", encoding="utf-8") as f:
        use_acuity = bool(json.load(f)["use_acuity"])
    fr = fc.load_frames(use_acuity, need_test=not smoke)
    if smoke:  # evaluation split = validation (no test row is read)
        fr["Xte"], fr["yte"], fr["te"] = fr["Xva"], fr["yva"], fr["va"]
    return fr


def booster_path(seed: int, B: int, smoke: bool = False) -> str:
    return os.path.join(fc.ARTIFACT_DIR, f"thermo_s{seed}_B{B}" + ("_smoke" if smoke else "") + ".json")


def params_path(seed: int, B: int, smoke: bool = False) -> str:
    return booster_path(seed, B, smoke).replace(".json", "_params.json")


def load_model(fr, seed: int, B: int, smoke: bool = False):
    """Booster saved by the simulation stage, the public binner rebuilt from the training split, and the saved operating points."""
    feats = fr["feats"]
    edges, binner = fc.make_binner(fr["Xtr"], feats, B)
    clf = xgb.XGBClassifier()
    clf.load_model(booster_path(seed, B, smoke))
    with open(params_path(seed, B, smoke), "r", encoding="utf-8") as f:
        par = json.load(f)
    assert par["feats"] == feats and par["B"] == B
    Xte_b = binner(fr["Xte"])
    m_te = clf.get_booster().predict(xgb.DMatrix(Xte_b, feature_names=feats), output_margin=True)
    return clf.get_booster(), Xte_b, m_te, par["ops"]


def stratified_sample(m_te: np.ndarray, thr: float, per_class: int = PER_CLASS) -> np.ndarray:
    """Frozen stratified sample: per_class plaintext-positive then per_class plaintext-negative stays at the F1 threshold (generator seed 42)."""
    rng = np.random.default_rng(42)
    pp = (m_te >= thr).astype(int)
    return np.concatenate([rng.choice(np.flatnonzero(pp == 1), per_class, replace=False),
                           rng.choice(np.flatnonzero(pp == 0), per_class, replace=False)])


def circuit_summary(errs, reals, m_sel, secs, upload, ops: dict) -> dict:
    """summarize() of the pilot (gate 5e-4, F1-point agreement, gaps to the F1 threshold) plus agreement and gaps at both thresholds."""
    reals, m_sel = np.asarray(reals), np.asarray(m_sel)
    agree = {n: int(((reals >= ops[n]) == (m_sel >= ops[n])).sum()) for n in ("f1", "sens80")}
    gaps = {n: np.abs(m_sel - ops[n]) for n in ("f1", "sens80")}
    r = summarize(errs, agree["f1"], secs, upload, len(errs), gaps=gaps["f1"])
    r["decisions_agreeing_sens80"] = agree["sens80"]
    r["min_abs_gap_to_sens80_threshold_sampled"] = float(gaps["sens80"].min())
    r["error_below_min_gap_sens80"] = bool(max(errs) < gaps["sens80"].min())
    return r
