"""
Shared pieces for the final M1 continuous-feature run (Analysis-plan gate, GATES.md). Nothing here prints or writes
patient-level rows (CLAUDE.md 1.2). `require_signed_gates()` refuses to let any final script run unless the Validity gate and
the Analysis-plan gate entries in the root GATES.md each carry a non-empty "approved by:" line, and the Validity gate's
"Author's decision:" line is filled in. It cannot tell who wrote the line; the signatures are the author's.
"""
from __future__ import annotations

import json
import math
import os
import re
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import m1c_cohort as mc  # noqa: E402
from m1c_pilot_validity import LIMITS as _LIMITS_BASE  # noqa: E402
from splits import MODEL_SEEDS, patient_level_split  # noqa: E402
from train_plaintext import MAX_DEPTH, MAX_TREES, tune_thresholds  # noqa: E402

SEEDS = tuple(MODEL_SEEDS)
ACUITY = "triage_acuity"
B_PRIMARY, DEGREE_PRIMARY = 5, 9
B_SENS, DEGREE_SENS = 8, 15
N_BOOT, BOOT_SEED = 1000, 42
SCORE_FRACTION = 0.25
LIMITS = {**_LIMITS_BASE, ACUITY: (1.0, 5.0)}
RESULTS = os.path.join(_HERE, "..", "results")
DECISION_PATH = os.path.join(RESULTS, "m1c_final_acuity_decision.json")
ARTIFACT_DIR = os.path.join(_HERE, "..", "data", "final")
GATES_PATH = os.path.join(_HERE, "..", "..", "GATES.md")


def require_signed_gates() -> None:
    with open(GATES_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    for title in ("Validity gate, M1 continuous-feature task", "Analysis-plan gate, M1 continuous-feature task"):
        i = text.find(title)
        if i < 0:
            raise SystemExit(f"STOP: GATES.md has no entry '{title}'")
        j = text.find("\n### ", i + 1)
        j = j if j > 0 else text.find("Consequences stated in CLAUDE.md", i)
        body = text[i:j]
        if not re.search(r"^\s*approved by:[ \t]*\S", body, flags=re.M):
            raise SystemExit(f"STOP: '{title}' has no 'approved by' line written in GATES.md (CLAUDE.md 2.1, 7.3)")
        if title.startswith("Validity") and not re.search(r"Author's decision:[ \t]*\S", body):
            raise SystemExit("STOP: the Validity gate's 'Author's decision' line is empty")
    print("GATES.md: Validity and Analysis-plan entries both carry an 'approved by' line", flush=True)


def require_signed_entry(title: str) -> None:
    """Refuse to run unless the GATES.md entry whose heading contains `title` carries a non-empty 'approved by' line."""
    with open(GATES_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    i = text.find(title)
    if i < 0:
        raise SystemExit(f"STOP: GATES.md has no entry '{title}'")
    j = text.find("\n### ", i + 1)
    j = j if j > 0 else text.find("Consequences stated in CLAUDE.md", i)
    if not re.search(r"^\s*approved by:[ \t]*\S", text[i:j], flags=re.M):
        raise SystemExit(f"STOP: '{title}' has no 'approved by' line written in GATES.md (CLAUDE.md 2.1, 7.3)")
    print(f"GATES.md: '{title}' carries an 'approved by' line", flush=True)


def operating_points_midpoint(y_val: np.ndarray, margin_val: np.ndarray) -> dict:
    """Operating points as in operating_points, with every threshold that coincides with an attained validation margin moved to the midpoint between
    that margin and the next lower distinct attained margin. Decisions on the validation split do not change; no decision sits on an attained margin.
    This is the fix for the ties found in the first final run (docs/m1c_final_ties_note.md)."""
    ops = operating_points(y_val, margin_val)
    attained = np.unique(margin_val)
    out = {}
    for name, thr in ops.items():
        hit = np.flatnonzero(np.abs(attained - thr) <= 1e-9)
        if len(hit) and hit[0] > 0:
            k = hit[0]
            out[name] = float((attained[k] + attained[k - 1]) / 2.0)
        else:
            out[name] = float(thr)
    return out


def exact_odd_step(B: int) -> np.ndarray:
    """Power-basis coefficients (degree 2B-1) of h = 0.5 - 0.5 p, p odd with p = 1 at (j + 0.5)/(B - 0.5), j = 0..B-1."""
    u = (np.arange(B) + 0.5) / (B - 0.5)
    powers = np.arange(1, 2 * B, 2)
    c = np.linalg.solve(u[:, None] ** powers[None, :], np.ones(B))
    p = np.zeros(2 * B)
    p[powers] = c
    h = -0.5 * p
    h[0] += 0.5
    hv = np.polynomial.polynomial.polyval
    assert np.max(np.abs(hv(u, h))) < 1e-8 and np.max(np.abs(hv(-u, h) - 1.0)) < 1e-8, "interpolant is not exact"
    return h


def shifted_threshold(t: float) -> float:
    return math.ceil(t - 1e-6) - 0.5


def fit_xgb(X: pd.DataFrame, y: np.ndarray, seed: int) -> xgb.XGBClassifier:
    """Same configuration as train_plaintext.train_and_eval (base_score 0.5 is locked there)."""
    clf = xgb.XGBClassifier(n_estimators=MAX_TREES, max_depth=MAX_DEPTH, tree_method="hist", subsample=0.8,
                            colsample_bytree=0.8, base_score=0.5, random_state=seed, eval_metric="logloss")
    clf.fit(X, y)
    return clf


def load_frames(use_acuity: bool, need_test: bool) -> dict:
    """Train and validation frames (and the test frame only when need_test). Reading the test split requires the acuity
    decision file to exist (Analysis-plan gate: decided before any test row is read)."""
    if need_test and not os.path.exists(DECISION_PATH):
        raise SystemExit("STOP: results/m1c_final_acuity_decision.json is missing; the test split is not read before it exists")
    extra = (ACUITY,) if use_acuity else ()
    df, flow = mc.build_cohort(extra=extra)
    train_ids, val_ids, test_ids = patient_level_split(df)
    tr = df[df["subject_id"].isin(train_ids)].reset_index(drop=True)
    va = df[df["subject_id"].isin(val_ids)].reset_index(drop=True)
    te = df[df["subject_id"].isin(test_ids)].reset_index(drop=True) if need_test else None
    assert set(tr["subject_id"]).isdisjoint(val_ids) and set(va["subject_id"]).isdisjoint(test_ids)
    feats = list(mc.FEATURE_COLS) + ([ACUITY] if use_acuity else [])
    impute_cols = list(mc.VITAL_FEATURES) + ([ACUITY] if use_acuity else [])
    med = {c: float(tr[c].median()) for c in impute_cols}
    if use_acuity:
        assert float(med[ACUITY]).is_integer(), "median acuity is not an integer level"

    def imp(d):
        if d is None:
            return None
        X = d[feats].copy()
        for c in impute_cols:
            X[c] = X[c].fillna(med[c])
        return X
    return {"flow": flow, "feats": feats, "medians": med, "use_acuity": use_acuity,
            "tr": tr, "va": va, "te": te, "Xtr": imp(tr), "Xva": imp(va), "Xte": imp(te),
            "ytr": tr[mc.OUTCOME_COL].to_numpy(dtype=int), "yva": va[mc.OUTCOME_COL].to_numpy(dtype=int),
            "yte": None if te is None else te[mc.OUTCOME_COL].to_numpy(dtype=int)}


def make_binner(Xtr_raw: pd.DataFrame, feats: list[str], B: int):
    """Public bin table: training quantile edges for every input except acuity, which maps v -> v - 1."""
    edges = {c: np.unique(np.quantile(Xtr_raw[c].to_numpy(dtype=float), np.arange(1, B) / B))
             for c in feats if c != ACUITY}

    def binned(X: pd.DataFrame) -> pd.DataFrame:
        cols = {}
        for c in feats:
            v = X[c].to_numpy(dtype=float)
            cols[c] = (np.clip(np.round(v) - 1, 0, 4) if c == ACUITY
                       else np.searchsorted(edges[c], v, side="right").astype(float))
        return pd.DataFrame(cols)
    return edges, binned


def operating_points(y_val: np.ndarray, margin_val: np.ndarray) -> dict:
    """Two decision thresholds in margin space, set on validation only: F1-maximizing and 80% sensitivity."""
    proba = 1.0 / (1.0 + np.exp(-margin_val))
    thr = tune_thresholds(y_val.reshape(-1, 1), proba.reshape(-1, 1))[0]
    f1_thr = float(np.log(thr / (1.0 - thr)))
    s80 = float(np.quantile(margin_val[y_val == 1], 0.20, method="lower"))
    return {"f1": f1_thr, "sens80": s80}


def decision_metrics(y: np.ndarray, margin: np.ndarray, thr_margin: float) -> dict:
    pred = (margin >= thr_margin).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    return {"f1": float(f1_score(y, pred, zero_division=0)), "sensitivity": tp / max(tp + fn, 1),
            "specificity": tn / max(tn + fp, 1), "ppv": tp / max(tp + fp, 1), "predicted_positive_rate": float(pred.mean())}


def utility(y: np.ndarray, margin: np.ndarray, ops: dict) -> dict:
    return {"auroc": float(roc_auc_score(y, margin)), "auprc": float(average_precision_score(y, margin)),
            "at_f1_threshold": decision_metrics(y, margin, ops["f1"]),
            "at_sens80_threshold": decision_metrics(y, margin, ops["sens80"])}


def agreement(y_ref_margin: np.ndarray, circuit_margin: np.ndarray, thr: float, n_pos_ref: int | None = None) -> dict:
    pr = (y_ref_margin >= thr).astype(int)
    pc = (circuit_margin >= thr).astype(int)
    lost = int(((pr == 1) & (pc == 0)).sum())
    gained = int(((pr == 0) & (pc == 1)).sum())
    npos = int(pr.sum())
    return {"n_differing": int((pr != pc).sum()), "differing_per_1000": float(1000 * (pr != pc).mean()),
            "plain_positive_decisions": npos, "plain_pos_to_circuit_neg": lost, "plain_neg_to_circuit_pos": gained,
            "lost_share_of_plain_positives": lost / max(npos, 1),
            "max_abs_margin_difference": float(np.max(np.abs(y_ref_margin - circuit_margin)))}


def subject_weights(subject_ids: pd.Series) -> tuple[np.ndarray, np.ndarray, int]:
    """Patient-cluster bootstrap weights: (per-resample subject counts, per-row subject codes, n_subjects)."""
    codes, uniques = pd.factorize(subject_ids)
    n = len(uniques)
    rng = np.random.default_rng(BOOT_SEED)
    w = rng.multinomial(n, np.full(n, 1.0 / n), size=N_BOOT).astype(np.int64)
    return w, codes, n


def ms(values) -> dict:
    v = np.asarray(values, dtype=float)
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "n": int(len(v)),
            "per_seed": [float(x) for x in v]}
