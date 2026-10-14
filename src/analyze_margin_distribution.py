#!/usr/bin/env python3
"""
Added analysis S2.2 (docs/step2_prespec_m1.md, section A): the distribution of comparison margins on the
current task, separately for splits on continuous features (the only ones that use the Chebyshev
polynomial) and splits on binary features (evaluated exactly), pooled over the 25 frozen models.

Reuses derive_circuit_params.py's loaders, normalization and 4,000-row training sample (seed 42), so the
pool is the same one that defined the interval `a`; it only adds the binary splits and the near-zero shares.
Reads the TRAINING split only. Output is aggregate (counts, shares, quantiles); no patient-level content.

Output: results/m1_margin_distribution.json
Usage: python analyze_margin_distribution.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from derive_circuit_params import (  # noqa: E402
    DEFAULT_COHORT, DEFAULT_FEATURES, DEFAULT_MODEL_DIR, derive_normalization, load_frozen_models,
)
from splits import patient_level_split  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(_HERE, "..", "results", "m1_margin_distribution.json")
SAMPLE_ROWS, SAMPLE_SEED = 4000, 42
CUTOFFS = (0.05, 0.10, 0.20)
QUANTILES = (10, 25, 50, 75, 90)


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cum = np.cumsum(w)
    return float(v[np.searchsorted(cum, q / 100.0 * cum[-1])])


def summarize(values: np.ndarray, weights: np.ndarray) -> dict:
    """Shares and quantiles of |value| under integer weights."""
    a = np.abs(values)
    total = float(weights.sum())
    out = {"n_values": int(total), "n_distinct_values": int(len(values))}
    for c in CUTOFFS:
        out[f"share_abs_lt_{c:.2f}"] = float(weights[a < c].sum() / total)
    out["abs_quantiles"] = {str(q): weighted_quantile(a, weights, q) for q in QUANTILES}
    return out


def main() -> int:
    cohort = pd.read_parquet(DEFAULT_COHORT)
    features = pd.read_parquet(DEFAULT_FEATURES)
    label_cols = [c for c in cohort.columns if c.startswith("label_")]
    merged = features.merge(cohort[["stay_id", "subject_id"] + label_cols], on="stay_id", how="inner")
    feature_cols = [c for c in features.columns if c != "stay_id"]
    train_ids, _, _ = patient_level_split(merged)
    X_train = merged.loc[merged["subject_id"].isin(train_ids), feature_cols]
    nunique = X_train.nunique()
    binary_features = set(nunique[nunique <= 2].index)
    norm = derive_normalization(X_train)

    rng = np.random.default_rng(SAMPLE_SEED)
    idx = rng.choice(len(X_train), size=min(SAMPLE_ROWS, len(X_train)), replace=False)
    Xs = X_train.iloc[idx]

    thresholds: dict[str, list[float]] = {}
    for clf in load_frozen_models(DEFAULT_MODEL_DIR).values():
        df = clf.get_booster().trees_to_dataframe()
        internal = df[df["Feature"] != "Leaf"]
        for feat, thr in zip(internal["Feature"], internal["Split"]):
            thresholds.setdefault(feat, []).append(float(thr))

    cont_vals, cont_w = [], []
    bin_vals, bin_w = [], []
    for feat, thrs in thresholds.items():
        t = np.asarray(thrs, dtype=float)
        if feat in binary_features:
            x = Xs[feat].to_numpy(dtype=float)
            assert set(np.unique(x)).issubset({0.0, 1.0}), f"{feat}: binary feature not in {{0,1}}"
            n0, n1 = float((x == 0).sum()), float((x == 1).sum())
            bin_vals += [0.0 - t, 1.0 - t]
            bin_w += [np.full(len(t), n0), np.full(len(t), n1)]
        else:
            lo, hi = norm[feat]["min"], norm[feat]["max"]
            span = (hi - lo) if hi > lo else 1.0
            z = ((Xs[feat].to_numpy(dtype=float) - lo) / span)[:, None] - ((t - lo) / span)[None, :]
            vals, counts = np.unique(np.round(z.ravel(), 12), return_counts=True)
            cont_vals.append(vals)
            cont_w.append(counts.astype(float))

    # pool continuous values (merge duplicates across features)
    cv = np.concatenate(cont_vals)
    cw = np.concatenate(cont_w)
    uniq, inv = np.unique(cv, return_inverse=True)
    cw = np.bincount(inv, weights=cw)
    bv = np.concatenate(bin_vals)
    bw = np.concatenate(bin_w)

    n_split_cont = sum(len(v) for f, v in thresholds.items() if f not in binary_features)
    n_split_bin = sum(len(v) for f, v in thresholds.items() if f in binary_features)
    cont = summarize(uniq, cw)
    binary = summarize(bv, bw)
    both = summarize(np.concatenate([uniq, bv]), np.concatenate([cw, bw]))
    out = {
        "protocol": "docs/step2_prespec_m1.md section A (added analysis S2.2)",
        "pool": f"all split nodes of the 25 frozen models x {len(Xs)} training rows (seed {SAMPLE_SEED}); margin z = x - tau "
                "(continuous features normalized with training-split min-max, binary features on the raw 0/1 scale)",
        "split_test_read": False,
        "n_splits": {"continuous": n_split_cont, "binary": n_split_bin, "all": n_split_cont + n_split_bin,
                     "share_continuous": n_split_cont / (n_split_cont + n_split_bin)},
        "continuous_feature_splits": cont,
        "binary_feature_splits": binary,
        "all_splits": both,
        "cutoffs": list(CUTOFFS),
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    for name, d in (("continuous", cont), ("binary", binary), ("all", both)):
        print(f"{name:11s} n={d['n_values']:,}  " + "  ".join(f"|z|<{c:.2f}: {d[f'share_abs_lt_{c:.2f}']:.4f}" for c in CUTOFFS)
              + f"  median|z|={d['abs_quantiles']['50']:.4f}")
    print(f"splits: continuous {n_split_cont:,}, binary {n_split_bin:,}\nwritten -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
