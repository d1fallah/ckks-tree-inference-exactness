#!/usr/bin/env python3
"""
Track A mandatory core, plaintext half: XGBoost (<=50 trees, depth <=6) on the
top-K chronic-condition label space, patient-level split, 5-seed mean +/- std.

Multi-label design: the label space is 5 (to 10) independent binary chronic-
condition flags, not mutually exclusive -- a stay can have zero, one, or
several. This is fit as a One-vs-Rest ensemble of independent XGBoost binary
classifiers (`sklearn.multioutput.MultiOutputClassifier`), one tree ensemble
per label. This is the same OvR structure the label-space feasibility note's
"OvR circuit multiplier" already assumes for the encrypted-inference cost
estimate -- the plaintext and encrypted pipelines share one label-space design.

DECISION THRESHOLD (2026-09-11 audit fix). Every label in this cohort is a
minority class (21-38% prevalence), and at XGBoost's default 0.5 threshold the
OvR ensemble badly under-predicts positives -- measured predicted-positive rate
of 1.0% against 22.9% actual prevalence for fluid/electrolyte disorders,
collapsing its F1 to 0.037 despite an AUROC of 0.634 (i.e. real discriminative
signal that simply never crosses 0.5). Reporting that F1 would have been
doubly wrong: it understates the model, and it would have made this module's
primary utility criterion, |F1_plaintext - F1_encrypted| < 0.03, trivially
satisfiable, since an F1 near zero has no headroom to lose 0.03. So:

  * one decision threshold per label is tuned on a dedicated VALIDATION split
    (never on test, which would be optimistically biased, and never on train,
    where the model is overfit);
  * AUROC and AUPRC are reported alongside F1 as threshold-independent
    metrics, so the utility comparison never rests on threshold choice alone;
  * F1 at the fixed 0.5 threshold is still reported, so the effect of the
    tuning is visible rather than hidden.

Split design: ONE fixed patient-level train/val/test split from `splits.py`
(60/20/20), not re-drawn per seed. CLAUDE.md's "mean +/- std across >=5 fixed
random seeds" is about model-fitting stochasticity, not about re-drawing the
evaluation set, which would confound split variance with model variance.

Usage:
    python train_plaintext.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.multioutput import MultiOutputClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from splits import MODEL_SEEDS, SPLIT_SEED, patient_level_split  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FEATURES_PATH = os.path.join(_HERE, "..", "data", "m1_features.parquet")
DEFAULT_COHORT_PATH = os.path.join(_HERE, "..", "data", "m1_cohort.parquet")
DEFAULT_RESULTS_PATH = os.path.join(_HERE, "..", "results", "m1_plaintext_baseline.json")
DEFAULT_MODEL_DIR = os.path.join(_HERE, "..", "data", "models")

MAX_TREES = 50
MAX_DEPTH = 6
# Threshold grid for the per-label tuning pass on the validation split.
THRESHOLD_GRID = np.arange(0.05, 0.96, 0.01)


def tune_thresholds(y_val: np.ndarray, proba_val: np.ndarray) -> list[float]:
    """One decision threshold per label, maximizing that label's F1 on the
    validation split. Returns thresholds in label-column order."""
    thresholds = []
    for i in range(y_val.shape[1]):
        best_f1, best_thr = -1.0, 0.5
        for thr in THRESHOLD_GRID:
            f = f1_score(y_val[:, i], (proba_val[:, i] >= thr).astype(int), zero_division=0)
            if f > best_f1:
                best_f1, best_thr = f, float(thr)
        thresholds.append(round(best_thr, 4))
    return thresholds


def train_and_eval(
    X_train, y_train, X_val, y_val, X_test, y_test, seed: int, n_estimators: int, max_depth: int,
    model_dir: str | None = None, label_cols: list[str] | None = None,
) -> dict:
    # subsample/colsample_bytree < 1.0 (standard regularization values, not
    # roadmap-mandated) are set deliberately: at XGBoost's defaults (1.0/1.0)
    # every tree sees the full training set and every feature at every split,
    # so tree construction under tree_method="hist" is entirely deterministic
    # and random_state has nothing left to randomize. A first run at the
    # defaults produced an identical macro-F1 across all 5 seeds (std = 0.0000)
    # -- not a bug, but a vacuous "5-seed" report. Row/column subsampling gives
    # random_state a real stochastic component to vary, so the reported std
    # reflects genuine training variance rather than a fixed point restated.
    # base_score=0.5 is LOCKED, not a default (M1-0 Implementation Lock).
    # XGBoost >=2.0 estimates base_score from the training label mean and writes
    # it into the saved model (here it would be ~0.5367). Concrete-ML's pinned
    # xgboost 1.6.2 -- the version that must load this exact frozen model to
    # produce XGB_quant for the guide's float->quant->CKKS decomposition (§3.4)
    # -- silently ignores that stored value and falls back to 0.5. Measured
    # 2026-09-16 on a 300x12 synthetic probe: loading a 3.2.0-trained model into
    # 1.6.2 shifted probabilities by a mean of -0.019 and flipped 6/300 labels
    # (agreement 0.98), every flip at a near-zero margin. That is a silent
    # cross-version contamination that would be misattributed to quantization.
    # Training with base_score=0.5 explicitly makes the round trip exact:
    # re-verified agreement 1.0, max |proba diff| 0.000e+00.
    base = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        tree_method="hist",
        subsample=0.8,
        colsample_bytree=0.8,
        base_score=0.5,
        random_state=seed,
        eval_metric="logloss",
    )
    clf = MultiOutputClassifier(base)
    clf.fit(X_train, y_train)

    # Persist every per-label booster. These frozen models are the anchor for
    # three downstream M1-0 requirements that cannot be satisfied without them:
    # (a) the pooled, seed-independent approximation interval of §3.2, derived
    # from the (x_j - t) distribution over ALL 5 seeds' split thresholds;
    # (b) XGB_quant (§3.4), which must be a quantization of this exact frozen
    # model, never a separately trained one; (c) eval_ckks.py, which reads the
    # tree structure and leaf values directly. Saved as XGBoost's JSON format
    # (not pickle) so the Concrete-ML venv's xgboost 1.6.2 can load them.
    if model_dir is not None:
        os.makedirs(model_dir, exist_ok=True)
        names = label_cols or [f"label{i}" for i in range(len(clf.estimators_))]
        for est, name in zip(clf.estimators_, names):
            est.save_model(os.path.join(model_dir, f"xgb_float_seed{seed}_{name}.json"))

    def probas(X):
        return np.column_stack([p[:, 1] for p in clf.predict_proba(X)])

    y_val_arr = np.asarray(y_val)
    y_test_arr = np.asarray(y_test)
    proba_val = probas(X_val)
    proba_test = probas(X_test)

    thresholds = tune_thresholds(y_val_arr, proba_val)

    pred_tuned = np.column_stack(
        [(proba_test[:, i] >= thresholds[i]).astype(int) for i in range(len(thresholds))]
    )
    pred_default = (proba_test >= 0.5).astype(int)

    per_label = []
    for i in range(y_test_arr.shape[1]):
        per_label.append(
            {
                "test_prevalence": float(y_test_arr[:, i].mean()),
                "tuned_threshold": thresholds[i],
                # Margin-space equivalent of the probability threshold above.
                # The guide (§3.3) removes the output-probability sigmoid from
                # the CKKS circuit entirely, because sigmoid is monotonic and so
                # preserves both threshold decisions and rank order -- meaning
                # F1, AUROC and AUPRC are all obtainable from the raw XGBoost
                # margin. The encrypted circuit therefore emits a margin, and
                # the frozen decision rule it is compared against must live in
                # margin space too: margin_threshold = logit(p_threshold).
                # Recorded here so eval_ckks.py consumes a frozen number rather
                # than re-deriving it.
                "tuned_threshold_margin": float(np.log(thresholds[i] / (1.0 - thresholds[i]))),
                "predicted_positive_rate_tuned": float(pred_tuned[:, i].mean()),
                "predicted_positive_rate_at_0.5": float(pred_default[:, i].mean()),
                "f1_tuned": float(f1_score(y_test_arr[:, i], pred_tuned[:, i], zero_division=0)),
                "f1_at_0.5": float(f1_score(y_test_arr[:, i], pred_default[:, i], zero_division=0)),
                "auroc": float(roc_auc_score(y_test_arr[:, i], proba_test[:, i])),
                "auprc": float(average_precision_score(y_test_arr[:, i], proba_test[:, i])),
            }
        )

    return {
        "seed": seed,
        "macro_f1_tuned": float(f1_score(y_test_arr, pred_tuned, average="macro", zero_division=0)),
        "macro_f1_at_0.5": float(f1_score(y_test_arr, pred_default, average="macro", zero_division=0)),
        "mean_auroc": float(np.mean([p["auroc"] for p in per_label])),
        "mean_auprc": float(np.mean([p["auprc"] for p in per_label])),
        "per_label": per_label,
    }


def _mean_std(values: list[float]) -> dict:
    return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1))}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default=DEFAULT_FEATURES_PATH)
    parser.add_argument("--cohort", default=DEFAULT_COHORT_PATH)
    parser.add_argument("--results", default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="where the frozen per-seed, per-label XGB_float boosters are written (JSON)")
    parser.add_argument("--n-estimators", type=int, default=MAX_TREES)
    parser.add_argument("--max-depth", type=int, default=MAX_DEPTH)
    args = parser.parse_args()

    if not (1 <= args.n_estimators <= MAX_TREES):
        raise ValueError(f"n_estimators must be in [1, {MAX_TREES}] per the roadmap's Track A core spec, got {args.n_estimators}")
    if not (1 <= args.max_depth <= MAX_DEPTH):
        raise ValueError(f"max_depth must be in [1, {MAX_DEPTH}] per the roadmap's Track A core spec, got {args.max_depth}")

    cohort = pd.read_parquet(args.cohort)
    features = pd.read_parquet(args.features)
    label_cols = [c for c in cohort.columns if c.startswith("label_")]

    merged = features.merge(cohort[["stay_id", "subject_id"] + label_cols], on="stay_id", how="inner")
    feature_cols = [c for c in features.columns if c != "stay_id"]

    train_ids, val_ids, test_ids = patient_level_split(merged)
    tr = merged["subject_id"].isin(train_ids)
    va = merged["subject_id"].isin(val_ids)
    te = merged["subject_id"].isin(test_ids)

    X_train, y_train = merged.loc[tr, feature_cols], merged.loc[tr, label_cols]
    X_val, y_val = merged.loc[va, feature_cols], merged.loc[va, label_cols]
    X_test, y_test = merged.loc[te, feature_cols], merged.loc[te, label_cols]

    print(f"train: {len(X_train)} stays / {merged.loc[tr, 'subject_id'].nunique()} patients")
    print(f"val:   {len(X_val)} stays / {merged.loc[va, 'subject_id'].nunique()} patients")
    print(f"test:  {len(X_test)} stays / {merged.loc[te, 'subject_id'].nunique()} patients")
    print(f"feature dim: {len(feature_cols)}  labels: {label_cols}")

    per_seed = []
    for seed in MODEL_SEEDS:
        r = train_and_eval(
            X_train, y_train, X_val, y_val, X_test, y_test, seed, args.n_estimators, args.max_depth,
            model_dir=args.model_dir, label_cols=label_cols,
        )
        per_seed.append(r)
        print(
            f"  seed={seed}: macro_f1_tuned={r['macro_f1_tuned']:.4f} "
            f"(at 0.5: {r['macro_f1_at_0.5']:.4f})  mean_auroc={r['mean_auroc']:.4f}"
        )

    n_labels = len(label_cols)
    summary = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "seeds": list(MODEL_SEEDS),
        "split_seed": SPLIT_SEED,
        "split": {"train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum())},
        "feature_dim": len(feature_cols),
        "label_cols": label_cols,
        "threshold_policy": "per-label, tuned to maximize F1 on the validation split only",
        "macro_f1_tuned": _mean_std([r["macro_f1_tuned"] for r in per_seed]),
        "macro_f1_at_0.5": _mean_std([r["macro_f1_at_0.5"] for r in per_seed]),
        "mean_auroc": _mean_std([r["mean_auroc"] for r in per_seed]),
        "mean_auprc": _mean_std([r["mean_auprc"] for r in per_seed]),
        "per_label_across_seeds": {
            label_cols[i]: {
                "test_prevalence": per_seed[0]["per_label"][i]["test_prevalence"],
                "f1_tuned": _mean_std([r["per_label"][i]["f1_tuned"] for r in per_seed]),
                "f1_at_0.5": _mean_std([r["per_label"][i]["f1_at_0.5"] for r in per_seed]),
                "auroc": _mean_std([r["per_label"][i]["auroc"] for r in per_seed]),
                "auprc": _mean_std([r["per_label"][i]["auprc"] for r in per_seed]),
                "tuned_threshold_mean": float(np.mean([r["per_label"][i]["tuned_threshold"] for r in per_seed])),
            }
            for i in range(n_labels)
        },
        "per_seed_results": per_seed,
    }

    os.makedirs(os.path.dirname(args.results), exist_ok=True)
    with open(args.results, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(
        f"\nmacro_f1 (tuned thresholds) = {summary['macro_f1_tuned']['mean']:.4f} "
        f"+/- {summary['macro_f1_tuned']['std']:.4f}"
    )
    print(
        f"macro_f1 (fixed 0.5)        = {summary['macro_f1_at_0.5']['mean']:.4f} "
        f"+/- {summary['macro_f1_at_0.5']['std']:.4f}"
    )
    print(f"mean AUROC                  = {summary['mean_auroc']['mean']:.4f} +/- {summary['mean_auroc']['std']:.4f}")
    print(f"mean AUPRC                  = {summary['mean_auprc']['mean']:.4f} +/- {summary['mean_auprc']['std']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
