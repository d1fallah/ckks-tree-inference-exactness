#!/usr/bin/env python3
"""
Derive the M1-0 Implementation Lock quantities from the frozen XGB_float models.

Four of the ten lock items in the supervisor's v4.1 guide cannot be written
from a desk -- they are properties of the trained ensembles and must be
measured once, seed-independently, and then frozen:

  (1) §3.1/§3.2  the input normalization rule and the global approximation
                 interval half-width `a`, pooled across ALL FIVE seeds'
                 models so that `a` is not a second varying approximation
                 source alongside k;
  (2) §3.1.1     the multiplicative depth of the COMPLETE tree and ensemble
                 circuit, not merely of one comparison polynomial;
  (3) §3.6       the packing layout and slot budget;
  (4) §3.10      the inputs needed to pre-register M1-A's numerical-fidelity
                 pass thresholds before any result is observed.

This script reads only frozen artifacts (data/models/*.json, written by
train_plaintext.py) and the TRAINING split. It never touches the test split:
the interval is a development-time quantity, and deriving it from test data
would leak.

Output: results/m1_circuit_params.json

Usage:
    python derive_circuit_params.py
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from notation import K_APPROX_DEGREE_VALUES  # noqa: E402
from splits import patient_level_split  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_DIR = os.path.join(_HERE, "..", "data", "models")
DEFAULT_COHORT = os.path.join(_HERE, "..", "data", "m1_cohort.parquet")
DEFAULT_FEATURES = os.path.join(_HERE, "..", "data", "m1_features.parquet")
DEFAULT_OUT = os.path.join(_HERE, "..", "results", "m1_circuit_params.json")

# The interval percentile is a PRE-REGISTERED choice, fixed before the sweep:
# the smallest round percentile that covers essentially all observed
# (x_j - t) mass while keeping the interval tight enough for low-degree
# approximation. Declared here rather than selected after seeing k-sweep
# results, per §3.2's "specified in M1-0" requirement.
INTERVAL_PERCENTILE = 99.9
# Tree depth of the locked Track A configuration.
TREE_DEPTH = 6


def load_frozen_models(model_dir: str) -> dict[str, xgb.XGBClassifier]:
    models = {}
    for path in sorted(glob.glob(os.path.join(model_dir, "xgb_float_seed*_label_*.json"))):
        clf = xgb.XGBClassifier()
        clf.load_model(path)
        models[os.path.basename(path)] = clf
    return models


def split_census(models: dict, feature_cols: list[str], binary_features: set[str]) -> dict:
    """Every internal split across every frozen model, classified by whether
    its feature is binary or continuous.

    This matters more than it looks. For a binary feature x in {0,1} and any
    threshold t in (0,1], the branch decision `x < t` is EXACT and linear:
    it equals (1 - x). No polynomial approximation is needed, no
    multiplicative depth is consumed, and k has no effect on it whatsoever.
    Only splits on genuinely continuous features are approximated, so the
    fraction of continuous splits bounds how much the k-sweep can move any
    metric at all.
    """
    per_model = []
    feature_split_counts: Counter = Counter()
    n_binary = n_continuous = 0
    for name, clf in models.items():
        df = clf.get_booster().trees_to_dataframe()
        internal = df[df["Feature"] != "Leaf"]
        b = int(internal["Feature"].isin(binary_features).sum())
        c = int(len(internal) - b)
        n_binary += b
        n_continuous += c
        feature_split_counts.update(internal["Feature"].tolist())
        per_model.append(
            {
                "model": name,
                "n_trees": int(df["Tree"].nunique()),
                "n_internal_splits": int(len(internal)),
                "n_leaves": int((df["Feature"] == "Leaf").sum()),
                "n_binary_splits": b,
                "n_continuous_splits": c,
            }
        )
    total = n_binary + n_continuous
    return {
        "per_model": per_model,
        "total_internal_splits": total,
        "total_binary_splits": n_binary,
        "total_continuous_splits": n_continuous,
        "fraction_continuous_splits": round(n_continuous / total, 6) if total else 0.0,
        "max_internal_splits_in_one_model": max(m["n_internal_splits"] for m in per_model),
        "top_20_features_by_split_count": feature_split_counts.most_common(20),
    }


def derive_normalization(X_train: pd.DataFrame) -> dict:
    """Min-max bounds from the TRAINING split only.

    Normalizing is not cosmetic here. Raw `age` spans 18-103, so a raw
    (x_j - t) for an age split can reach +/-85, and approximating a step
    function over an interval that wide needs a far higher polynomial degree
    than over [-1, 1]. Mapping every feature to [0, 1] bounds (x_j - t) to
    [-1, 1] by construction.
    """
    return {
        col: {"min": float(X_train[col].min()), "max": float(X_train[col].max())}
        for col in X_train.columns
    }


def pooled_interval(
    models: dict,
    X_train: pd.DataFrame,
    norm: dict,
    binary_features: set[str],
    sample_rows: int = 4000,
    seed: int = 42,
) -> dict:
    """Pooled |x_j - t| distribution over CONTINUOUS-feature splits only,
    in normalized space, across all five seeds' models at once.

    Binary splits are excluded because they are evaluated exactly (see
    `split_census`) and would otherwise dominate the pooled distribution with
    a trivial +/-0.5 spike, biasing `a` downward.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_train), size=min(sample_rows, len(X_train)), replace=False)
    Xs = X_train.iloc[idx]

    # Collect continuous-feature thresholds, grouped by feature.
    thresholds_by_feature: dict[str, list[float]] = {}
    for clf in models.values():
        df = clf.get_booster().trees_to_dataframe()
        internal = df[(df["Feature"] != "Leaf") & (~df["Feature"].isin(binary_features))]
        for feat, thr in zip(internal["Feature"], internal["Split"]):
            thresholds_by_feature.setdefault(feat, []).append(float(thr))

    diffs = []
    per_feature = {}
    for feat, thrs in thresholds_by_feature.items():
        lo, hi = norm[feat]["min"], norm[feat]["max"]
        rng_span = (hi - lo) if hi > lo else 1.0
        x_norm = ((Xs[feat].to_numpy(dtype=float) - lo) / rng_span)[:, None]
        t_norm = ((np.asarray(thrs, dtype=float) - lo) / rng_span)[None, :]
        d = (x_norm - t_norm).ravel()
        diffs.append(d)
        per_feature[feat] = {
            "n_distinct_thresholds": len(set(thrs)),
            "n_splits": len(thrs),
            "abs_diff_p99_9": float(np.percentile(np.abs(d), 99.9)),
            "abs_diff_max": float(np.abs(d).max()),
        }

    if not diffs:
        return {"note": "no continuous-feature splits found", "per_feature": {}}

    all_d = np.concatenate(diffs)
    abs_d = np.abs(all_d)
    a = float(np.percentile(abs_d, INTERVAL_PERCENTILE))
    return {
        "interval_percentile_preregistered": INTERVAL_PERCENTILE,
        "a_half_width": round(a, 6),
        "interval": [-round(a, 6), round(a, 6)],
        "n_sampled_rows": int(len(Xs)),
        "n_pooled_difference_values": int(all_d.size),
        "abs_diff_quantiles": {
            q: round(float(np.percentile(abs_d, q)), 6)
            for q in (50, 75, 90, 95, 99, 99.9, 100)
        },
        "fraction_outside_interval_clamped": round(float((abs_d > a).mean()), 8),
        "per_feature": per_feature,
    }


def depth_budget(max_splits_per_model: int) -> dict:
    """Multiplicative depth of the COMPLETE circuit, per §3.1.1.

    Circuit, per the guide's own definition:
      b_i(x)  = branch indicator for split i
                -- exact and depth-0 for a binary feature (1 - x);
                -- degree-k polynomial for a continuous feature.
      P_l(x)  = product of b_i over the splits on the path to leaf l
                -- a balanced multiplication tree over TREE_DEPTH factors
                   costs ceil(log2(TREE_DEPTH)) levels.
      T(x)    = sum_l P_l(x) * v_l
                -- v_l is a PLAINTEXT leaf value, so this is a
                   plaintext-ciphertext multiply. It is FOLDED into the last
                   level of the path product (scale one factor by v_l before
                   multiplying) rather than charged as its own level.
      ensemble = sum over trees, and OvR per-class sum -- both additions,
                 which cost zero multiplicative depth.

    Polynomial depth is DERIVED FROM CODE (amendment v1.3, 2026-09-21): depth_accounting.walk_circuit(k)
    runs the committed evaluator ckks_circuit.eval_poly_tenseal on a level-tracking stand-in and reads the
    level off the result. That evaluator is a balanced power ladder with a plaintext coefficient multiply per
    power -- it is NOT Paterson-Stockmeyer -- so the earlier closed form ceil(log2(k+1)) omitted the
    coefficient-multiply level and undercounted EVERY k by one (k=9: 7 tabled, 8 actual). The count is
    checked against real ciphertexts by src/probe_multiplicative_depth.py.
    """
    from depth_accounting import walk_circuit  # local import: pulls in ckks_circuit (TenSEAL)

    out = {}
    for k in K_APPROX_DEGREE_VALUES:
        walked = walk_circuit(k, TREE_DEPTH)
        poly_depth = walked["polynomial_depth"]
        path_depth = walked["path_product_depth"]
        total = walked["total_multiplicative_depth"]
        # SEAL/TenSEAL coefficient-modulus pattern: one 60-bit special prime at
        # each end plus one 40-bit prime per rescaling level.
        coeff_bits = 60 + 40 * total + 60
        # Max total coefficient-modulus bits at 128-bit security, per the
        # Homomorphic Encryption Security Standard tables.
        if coeff_bits <= 218:
            n_required = 8192
        elif coeff_bits <= 438:
            n_required = 16384
        elif coeff_bits <= 881:
            n_required = 32768
        else:
            n_required = None
        out[f"k={k}"] = {
            "comparison_polynomial_depth": poly_depth,
            "path_product_depth": path_depth,
            "leaf_scaling_depth": 0,
            "ensemble_aggregation_depth": 0,
            "total_multiplicative_depth": total,
            "coeff_modulus_bits_required": coeff_bits,
            "min_poly_modulus_degree_at_128bit": n_required,
            "slots_at_that_degree": (n_required // 2) if n_required else None,
            "all_ensemble_splits_fit_in_one_ciphertext": (
                bool(max_splits_per_model <= n_required // 2) if n_required else None
            ),
        }
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
    continuous_features = [c for c in feature_cols if c not in binary_features]

    models = load_frozen_models(DEFAULT_MODEL_DIR)
    if not models:
        raise SystemExit(f"no frozen models in {DEFAULT_MODEL_DIR} -- run train_plaintext.py first")

    census = split_census(models, feature_cols, binary_features)
    norm = derive_normalization(X_train)
    interval = pooled_interval(models, X_train, norm, binary_features)
    depths = depth_budget(census["max_internal_splits_in_one_model"])

    out = {
        "n_frozen_models": len(models),
        "n_binary_features": len(binary_features),
        "n_continuous_features": len(continuous_features),
        "continuous_features": continuous_features,
        "split_census": census,
        "normalization_rule": "per-feature min-max to [0,1], bounds from the TRAINING split only",
        "normalization_bounds_continuous_only": {c: norm[c] for c in continuous_features},
        "approximation_interval": interval,
        "depth_budget": depths,
        "tree_depth": TREE_DEPTH,
    }

    os.makedirs(os.path.dirname(DEFAULT_OUT), exist_ok=True)
    with open(DEFAULT_OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"frozen models loaded: {len(models)}")
    print(f"binary features: {len(binary_features)}  continuous: {len(continuous_features)} {continuous_features}")
    print()
    print("--- split census ---")
    print(f"  total internal splits across all models: {census['total_internal_splits']}")
    print(f"  on binary features    : {census['total_binary_splits']}")
    print(f"  on continuous features: {census['total_continuous_splits']}  "
          f"({100*census['fraction_continuous_splits']:.2f}% -- only these are approximated)")
    print(f"  max splits in one model: {census['max_internal_splits_in_one_model']}")
    print("  top features by split count:")
    for feat, n in census["top_20_features_by_split_count"][:8]:
        print(f"    {feat:<34} {n}")
    print()
    print("--- approximation interval (continuous splits, normalized, pooled over 5 seeds) ---")
    if "a_half_width" in interval:
        print(f"  a (p{INTERVAL_PERCENTILE}) = {interval['a_half_width']}")
        print(f"  quantiles of |x-t|: {interval['abs_diff_quantiles']}")
        print(f"  fraction clamped  : {interval['fraction_outside_interval_clamped']}")
    print()
    print("--- depth budget (complete circuit) ---")
    for kk, v in depths.items():
        print(f"  {kk}: depth={v['total_multiplicative_depth']} "
              f"(poly {v['comparison_polynomial_depth']} + path {v['path_product_depth']}), "
              f"coeff={v['coeff_modulus_bits_required']}b, "
              f"N>={v['min_poly_modulus_degree_at_128bit']}, "
              f"splits fit in 1 ct: {v['all_ensemble_splits_fit_in_one_ciphertext']}")
    print(f"\nwritten to {DEFAULT_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
