#!/usr/bin/env python3
"""
Frozen evaluation protocol, per guide v4.1 Sec 4 and m1-0-implementation-
lock.md Sec 9. Every function here implements one specific, pre-specified
computation -- none of this is tunable after seeing results.

Sec 4.1: per-seed order. For EACH seed independently, compute F1_plain and
F1_CKKS on the test set with thresholds frozen at training time, THEN take
the difference. NEVER average predictions or per-label F1 across seeds
before differencing.

Sec 4.2: paired bootstrap CI. Train/freeze once per seed (already done);
resample the PAIRED test observations (stays) with replacement using
ALREADY-GENERATED predictions -- models are never retrained per resample.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """y_true, y_pred: (n_rows, n_labels) binary indicator matrices."""
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def per_seed_delta_f1(f1_plain: float, f1_ckks: float) -> float:
    """Sec 4.1: the ONLY sanctioned way to combine plain/CKKS F1 for one
    seed -- take the absolute difference of the two already-computed
    macro-F1 values. Never average F1 or predictions across seeds first."""
    return abs(f1_plain - f1_ckks)


def paired_bootstrap_delta_f1_ci(
    y_true: np.ndarray,
    y_pred_plain: np.ndarray,
    y_pred_ckks: np.ndarray,
    n_resamples: int = 2000,
    seed: int = 42,
) -> dict:
    """Sec 4.2: resample PAIRED test stays (rows) with replacement, using
    the already-generated plain/CKKS predictions for this seed -- no
    retraining. Returns the point estimate and the 95% percentile interval
    over resamples of |F1_plain - F1_CKKS|."""
    rng = np.random.default_rng(seed)
    n = y_true.shape[0]
    point = abs(macro_f1(y_true, y_pred_plain) - macro_f1(y_true, y_pred_ckks))
    deltas = np.empty(n_resamples)
    for b in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        f1p = macro_f1(y_true[idx], y_pred_plain[idx])
        f1c = macro_f1(y_true[idx], y_pred_ckks[idx])
        deltas[b] = abs(f1p - f1c)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {"point_estimate": float(point), "ci_95_lower": float(lo), "ci_95_upper": float(hi)}


def multi_seed_paired_bootstrap_delta_f1(
    y_true: np.ndarray,
    pred_plain_per_seed: list[np.ndarray],
    pred_ckks_per_seed: list[np.ndarray],
    n_resamples: int = 2000,
    seed: int = 42,
) -> dict:
    """Extends Sec 4.2's single-seed paired bootstrap to the Sec 11.3
    headline number, which is a SINGLE 'DeltaF1 = [value] [95% CI]' across
    the 5 training seeds. Since all seeds share the identical physical test
    set (patient-level split is fixed at SPLIT_SEED, only the model's
    random_state varies -- splits.py), one resampled row-index set is drawn
    per bootstrap iteration and applied identically across all 5 seeds
    ('paired' extended across seeds, not just across plain/CKKS arms), each
    seed's Delta-F1 on that resample is computed, and the 5 are averaged to
    produce one bootstrap replicate of the headline Delta-F1(k). The 95%
    percentile interval over replicates is the reported CI. Per-seed point
    estimates (Sec 4.1 order) are also returned, unaveraged, for transparency."""
    rng = np.random.default_rng(seed)
    n = y_true.shape[0]
    n_seeds = len(pred_plain_per_seed)
    point_per_seed = [
        per_seed_delta_f1(macro_f1(y_true, pred_plain_per_seed[s]), macro_f1(y_true, pred_ckks_per_seed[s]))
        for s in range(n_seeds)
    ]
    point = float(np.mean(point_per_seed))
    draws = np.empty(n_resamples)
    for b in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        per_seed_vals = [
            per_seed_delta_f1(macro_f1(yt, pred_plain_per_seed[s][idx]), macro_f1(yt, pred_ckks_per_seed[s][idx]))
            for s in range(n_seeds)
        ]
        draws[b] = float(np.mean(per_seed_vals))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {
        "point_estimate": point,
        "ci_95_lower": float(lo),
        "ci_95_upper": float(hi),
        "per_seed_delta_f1": point_per_seed,
    }


def e_score(margin_plain: np.ndarray, margin_ckks: np.ndarray) -> dict:
    """Normalized numerical score error between plaintext and encrypted
    (or simulated) margins, per Sec 4.4."""
    err = np.abs(margin_ckks - margin_plain)
    rng_span = float(margin_plain.max() - margin_plain.min()) if margin_plain.max() > margin_plain.min() else 1.0
    return {
        "mae_raw": float(err.mean()),
        "rmse_raw": float(np.sqrt((err**2).mean())),
        "mae_normalized": float(err.mean() / rng_span),
        "rmse_normalized": float(np.sqrt((err**2).mean()) / rng_span),
    }


def prediction_agreement(margin_plain: np.ndarray, margin_ckks: np.ndarray) -> float:
    """Sign-of-margin agreement rate between plaintext and CKKS/simulated
    label outputs, per Sec 4.4."""
    return float(np.mean(np.sign(margin_plain) == np.sign(margin_ckks)))


def threshold_at_margin(margin: np.ndarray, threshold_margin: float) -> np.ndarray:
    """Apply a frozen margin-space threshold (logit(p), from
    results/m1_plaintext_baseline.json's tuned_threshold_margin) directly to
    a margin score -- avoids re-introducing a sigmoid into the circuit, per
    M1-0 Sec 3."""
    return (margin >= threshold_margin).astype(int)


def per_class_metrics(y_true_col: np.ndarray, margin_plain: np.ndarray, margin_ckks: np.ndarray, threshold_margin: float) -> dict:
    pred_plain = threshold_at_margin(margin_plain, threshold_margin)
    pred_ckks = threshold_at_margin(margin_ckks, threshold_margin)
    out = {
        "f1_plain": float(f1_score(y_true_col, pred_plain, zero_division=0)),
        "f1_ckks": float(f1_score(y_true_col, pred_ckks, zero_division=0)),
        "agreement": prediction_agreement(margin_plain, margin_ckks),
    }
    out["delta_f1"] = abs(out["f1_plain"] - out["f1_ckks"])
    try:
        out["auroc_plain"] = float(roc_auc_score(y_true_col, margin_plain))
        out["auroc_ckks"] = float(roc_auc_score(y_true_col, margin_ckks))
        out["auprc_plain"] = float(average_precision_score(y_true_col, margin_plain))
        out["auprc_ckks"] = float(average_precision_score(y_true_col, margin_ckks))
    except ValueError:
        pass  # degenerate single-class fold, e.g. on a tiny subsample
    out.update(e_score(margin_plain, margin_ckks))
    return out
