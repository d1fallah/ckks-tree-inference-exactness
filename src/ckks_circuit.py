#!/usr/bin/env python3
"""
Shared CKKS tree-inference circuit: leaf extraction, coefficient assembly,
and two backends (numpy simulation, real TenSEAL encryption) that evaluate
the IDENTICAL structure, per Gate M1-0 Amendment v1.1 (m1-0-implementation-
lock.md Sec 15).

Packing (Sec 15.2): slot = leaf, across the WHOLE 50-tree ensemble for one
(seed, label) model -- not per tree. One ciphertext per tree level d in
1..TREE_DEPTH; slot l of level-d ciphertext holds the (already normalized,
already clamped) difference x_j - t for the depth-d split on global leaf l's
root-to-leaf path, or 0 if leaf l's path is shorter than d (padded, Sec
15.3). The path product over the 6 level-ciphertexts, followed by one sum(),
gives the ensemble margin directly: sum over ALL leaves of ALL trees of
P_l(x) * v_l = sum over trees of (that tree's leaf sum) = ensemble margin,
at zero extra depth for tree/ensemble aggregation.

Branch polynomial, per slot, per Sec 15.3:
  - CONTINUOUS feature (age, n_meds, n_distinct_meds): the frozen degree-k
    soft-step Chebyshev polynomial from data/poly_coeffs_v1.json, evaluated
    at z = normalized(x_j) - normalized(t), clamped to [-a, a] (M1-0 Sec 1.3
    item 6) BEFORE encryption (a client-side plaintext operation, since the
    client computes z itself -- no depth cost).
  - BINARY feature (131 of 134 features): EXACT degree-1 polynomial, fitted
    from the two-point system b(0-t)=1, b(1-t)=0 using the split's ACTUAL
    recorded XGBoost threshold t (not assumed to be 1.0): c0 = 1-t, c1 = -1.
    Zero approximation error at every k (M1-0 Sec 1.1 / Amendment Sec 15.3).
  - PAD (leaf shallower than the current level d): constant-1 polynomial
    (identity for the path product), input value irrelevant.
  - "No"/right-edge complement (x_j >= t): 1 - b(z), i.e. coefficients
    (1-c0, -c1, ..., -ck) -- same mechanism for binary and continuous.

Comparison orientation (M1-0 Sec 1.3 item 7): b(z) approximates 1 for z<0
(x_j < t, XGBoost's Yes/left branch), 0 for z>0; equality goes right,
matching XGBoost's own convention.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import tenseal as ts
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from notation import K_APPROX_DEGREE_VALUES  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
CIRCUIT_PARAMS_PATH = os.path.join(_HERE, "..", "results", "m1_circuit_params.json")
POLY_COEFFS_PATH = os.path.join(_HERE, "..", "data", "poly_coeffs_v1.json")

TREE_DEPTH = 6


def load_circuit_frozen_artifacts() -> dict:
    """The two frozen artifacts every circuit assembly reads: normalization
    bounds + approximation interval, and the versioned polynomial
    coefficients. Never regenerated here -- read-only consumption."""
    with open(CIRCUIT_PARAMS_PATH, "r", encoding="utf-8") as f:
        circuit_params = json.load(f)
    with open(POLY_COEFFS_PATH, "r", encoding="utf-8") as f:
        poly_coeffs = json.load(f)
    return {
        "continuous_features": set(circuit_params["continuous_features"]),
        "norm_bounds": circuit_params["normalization_bounds_continuous_only"],
        "a_half_width": circuit_params["approximation_interval"]["a_half_width"],
        "poly_coeffs": poly_coeffs,
    }


@dataclass
class SplitRef:
    """One split on one leaf's root-to-leaf path, at a fixed level d."""
    kind: str  # "binary" | "continuous" | "pad"
    feature: str | None
    threshold: float | None  # raw (unnormalized) XGBoost split value
    is_yes_edge: bool  # True: leaf follows the Yes/left (x_j < t) branch


@dataclass
class TreeCircuitStructure:
    """The frozen (patient-independent) structure of one (seed, label)
    model's complete ensemble, extracted once from the booster."""
    n_leaves: int
    leaf_value: np.ndarray  # shape (n_leaves,), frozen XGBoost leaf values
    # levels[d] has length n_leaves; levels[d][l] is leaf l's split at depth d+1
    levels: list[list[SplitRef]] = field(default_factory=list)


def extract_ensemble_leaf_circuit(
    booster: xgb.Booster, continuous_features: set[str], tree_depth: int = TREE_DEPTH
) -> TreeCircuitStructure:
    """Parse trees_to_dataframe() into the leaf-aligned structure above.

    Reuses the exact same parsing entry point as derive_circuit_params.py
    (trees_to_dataframe -> Feature/Split/Yes/No/ID columns), extended here to
    walk root-to-leaf paths rather than just counting split types.
    """
    df = booster.trees_to_dataframe()
    levels: list[list[SplitRef]] = [[] for _ in range(tree_depth)]
    leaf_values: list[float] = []

    for tree_id, tree_df in df.groupby("Tree"):
        nodes = {row.ID: row for row in tree_df.itertuples()}
        root_id = f"{tree_id}-0"

        def walk(node_id: str, path: list[SplitRef]) -> None:
            row = nodes[node_id]
            if row.Feature == "Leaf":
                leaf_values.append(float(row.Gain))
                leaf_idx = len(leaf_values) - 1
                for d in range(tree_depth):
                    levels[d].append(
                        path[d] if d < len(path) else SplitRef("pad", None, None, True)
                    )
                return
            feat = row.Feature
            thr = float(row.Split)
            kind = "continuous" if feat in continuous_features else "binary"
            walk(row.Yes, path + [SplitRef(kind, feat, thr, True)])
            walk(row.No, path + [SplitRef(kind, feat, thr, False)])

        walk(root_id, [])

    return TreeCircuitStructure(
        n_leaves=len(leaf_values),
        leaf_value=np.asarray(leaf_values, dtype=np.float64),
        levels=levels,
    )


def _binary_exact_coeffs(threshold: float, degree: int) -> np.ndarray:
    """Exact degree-1 fit for a binary {0,1} feature at recorded threshold t:
    b(z) with z = x - t must satisfy b(0-t)=1, b(1-t)=0 -> c0=1-t, c1=-1.
    Padded with zeros to length degree+1 so it shares the same evaluator as
    the continuous branch."""
    c = np.zeros(degree + 1)
    c[0] = 1.0 - threshold
    c[1] = -1.0
    return c


def _pad_coeffs(degree: int) -> np.ndarray:
    c = np.zeros(degree + 1)
    c[0] = 1.0
    return c


def _complement(c: np.ndarray) -> np.ndarray:
    out = -c.copy()
    out[0] = 1.0 - c[0]
    return out


def assemble_coefficient_vectors(
    structure: TreeCircuitStructure, poly_coeffs_power_basis: dict, degree: int
) -> list[np.ndarray]:
    """Per level d, a (degree+1, n_leaves) array: coeff_vectors[d][p, l] is
    the z^p coefficient for leaf l's branch polynomial at level d. Depends
    on the model structure and degree, NOT on any patient -- computed once
    and reused for every patient evaluated against this (seed, label, degree)."""
    soft_coeffs = np.asarray(poly_coeffs_power_basis[f"k={degree}"]["coefficients_power_basis"])
    assert len(soft_coeffs) == degree + 1

    out = []
    for level_refs in structure.levels:
        mat = np.zeros((degree + 1, structure.n_leaves))
        for l, ref in enumerate(level_refs):
            if ref.kind == "pad":
                c = _pad_coeffs(degree)
            elif ref.kind == "binary":
                c = _binary_exact_coeffs(ref.threshold, degree)
            else:  # continuous
                c = soft_coeffs.copy()
            if not ref.is_yes_edge:
                c = _complement(c)
            mat[:, l] = c
        out.append(mat)
    return out


def compute_level_inputs(
    structure: TreeCircuitStructure,
    x_row: pd.Series,
    norm_bounds: dict,
    a_half_width: float,
) -> list[np.ndarray]:
    """Per level d, the length-n_leaves array of z values to encrypt (or, in
    the numpy backend, to evaluate directly). Binary z is exact and never
    clamped (only two possible values); continuous z is normalized then
    clamped to [-a, a] per M1-0 Sec 1.3 item 6, entirely client-side."""
    out = []
    for level_refs in structure.levels:
        z = np.zeros(structure.n_leaves)
        for l, ref in enumerate(level_refs):
            if ref.kind == "pad":
                z[l] = 0.0
            elif ref.kind == "binary":
                z[l] = float(x_row[ref.feature]) - ref.threshold
            else:
                lo = norm_bounds[ref.feature]["min"]
                hi = norm_bounds[ref.feature]["max"]
                span = (hi - lo) if hi > lo else 1.0
                x_norm = (float(x_row[ref.feature]) - lo) / span
                t_norm = (ref.threshold - lo) / span
                zval = x_norm - t_norm
                z[l] = float(np.clip(zval, -a_half_width, a_half_width))
        out.append(z)
    return out


def eval_poly_numpy(coeff_mat: np.ndarray, z: np.ndarray) -> np.ndarray:
    """coeff_mat: (degree+1, n_leaves); z: (n_leaves,). Horner's method, per slot."""
    degree = coeff_mat.shape[0] - 1
    result = coeff_mat[degree].copy()
    for p in range(degree - 1, -1, -1):
        result = result * z + coeff_mat[p]
    return result


def evaluate_circuit_numpy(
    structure: TreeCircuitStructure,
    coeff_vectors: list[np.ndarray],
    x_row: pd.Series,
    norm_bounds: dict,
    a_half_width: float,
) -> float:
    """Full simulation: identical assembly to the encrypted path, evaluated
    in float64. This is the circuit-faithful simulation backend referenced
    by Amendment Sec 15.5 item 1."""
    z_levels = compute_level_inputs(structure, x_row, norm_bounds, a_half_width)
    path_product = np.ones(structure.n_leaves)
    for coeff_mat, z in zip(coeff_vectors, z_levels):
        path_product = path_product * eval_poly_numpy(coeff_mat, z)
    return float(np.sum(path_product * structure.leaf_value))


@dataclass
class BatchLevelArrays:
    """Per-level, vectorized-over-leaves versions of the same SplitRef data,
    precomputed once per (structure, feature_cols) so evaluate_circuit_batch
    can process an entire test set with no per-row or per-leaf Python loop.

    Key identity making one formula cover binary/continuous/pad uniformly:
    z = (x - lo)/span - (t - lo)/span = (x - t)/span. Setting lo=0, span=1
    for binary and pad slots reduces this to z = x - t (binary's exact
    formula) or an ignored value (pad, multiplied by a constant coefficient
    anyway) -- only continuous slots get real lo/span and get clamped.
    """
    feature_col_idx: np.ndarray  # (n_leaves,) int, index into the X column array
    threshold: np.ndarray  # (n_leaves,)
    lo: np.ndarray  # (n_leaves,)
    span: np.ndarray  # (n_leaves,)
    should_clamp: np.ndarray  # (n_leaves,) bool, True only for continuous


def prepare_batch_levels(
    structure: TreeCircuitStructure, feature_cols: list[str], norm_bounds: dict, a_half_width: float
) -> list[BatchLevelArrays]:
    col_index = {c: i for i, c in enumerate(feature_cols)}
    out = []
    for level_refs in structure.levels:
        n = len(level_refs)
        feat_idx = np.zeros(n, dtype=np.int64)
        threshold = np.zeros(n)
        lo = np.zeros(n)
        span = np.ones(n)
        clamp = np.zeros(n, dtype=bool)
        for l, ref in enumerate(level_refs):
            if ref.kind == "pad":
                feat_idx[l] = 0
                threshold[l] = 0.0
            elif ref.kind == "binary":
                feat_idx[l] = col_index[ref.feature]
                threshold[l] = ref.threshold
            else:
                feat_idx[l] = col_index[ref.feature]
                threshold[l] = ref.threshold
                b = norm_bounds[ref.feature]
                lo[l] = b["min"]
                span[l] = (b["max"] - b["min"]) if b["max"] > b["min"] else 1.0
                clamp[l] = True
        out.append(BatchLevelArrays(feat_idx, threshold, lo, span, clamp))
    return out


def eval_poly_numpy_batch(coeff_mat: np.ndarray, z: np.ndarray) -> np.ndarray:
    """coeff_mat: (degree+1, n_leaves); z: (n_rows, n_leaves). Horner's
    method, vectorized over both rows and leaves at once."""
    degree = coeff_mat.shape[0] - 1
    result = np.broadcast_to(coeff_mat[degree], z.shape).copy()
    for p in range(degree - 1, -1, -1):
        result = result * z + coeff_mat[p][None, :]
    return result


def evaluate_circuit_batch(
    batch_levels: list[BatchLevelArrays],
    coeff_vectors: list[np.ndarray],
    leaf_value: np.ndarray,
    X: np.ndarray,
    a_half_width: float,
) -> np.ndarray:
    """Vectorized simulation over an entire matrix of patients at once.
    X: (n_rows, n_features) in the SAME column order used to build
    batch_levels (see prepare_batch_levels). Returns (n_rows,) margins,
    numerically identical (to float precision) to calling
    evaluate_circuit_numpy row by row -- verified in tests/."""
    n_rows = X.shape[0]
    path_product = np.ones((n_rows, leaf_value.shape[0]))
    for bl, coeff_mat in zip(batch_levels, coeff_vectors):
        x_sel = X[:, bl.feature_col_idx]  # (n_rows, n_leaves)
        z = (x_sel - bl.threshold[None, :]) / bl.span[None, :]
        z = np.where(bl.should_clamp[None, :], np.clip(z, -a_half_width, a_half_width), z)
        path_product *= eval_poly_numpy_batch(coeff_mat, z)
    return path_product @ leaf_value


def _ct_power(z_ct, p: int, cache: dict):
    """Balanced (divide-and-conquer) computation of z_ct^p, giving depth
    O(log p) rather than Horner's O(p) -- required to fit M1-0 Sec 5's
    depth budget (poly depth ceil(log2(k+1))) for real ciphertexts. TenSEAL
    auto-aligns levels on addition (verified 2026-09-18), so no manual
    mod-switch calls are needed once every power is computed."""
    if p in cache:
        return cache[p]
    half = p // 2
    a = _ct_power(z_ct, half, cache) if half >= 1 else z_ct
    b = _ct_power(z_ct, p - half, cache) if (p - half) >= 1 else z_ct
    result = a * b if half >= 1 else z_ct
    cache[p] = result
    return result


def eval_poly_tenseal(coeff_mat: np.ndarray, z_ct):
    """coeff_mat: (degree+1, n_leaves) plaintext coefficients; z_ct: one
    CKKSVector ciphertext (n_leaves slots). Real ciphertext polynomial
    evaluation via the balanced power ladder above -- NOT a numpy
    simulation. Returns one CKKSVector ciphertext."""
    degree = coeff_mat.shape[0] - 1
    cache = {1: z_ct}
    # Start the accumulator from the p=1 term (a real ciphertext) rather
    # than a "multiply by zero" placeholder -- that placeholder would cost
    # one spurious extra depth level for no reason, and was the bug behind
    # an early "scale out of bounds" failure at k=9 (2026-09-18).
    result = z_ct * coeff_mat[1].tolist()
    result = result + coeff_mat[0].tolist()
    for p in range(2, degree + 1):
        pw = _ct_power(z_ct, p, cache)
        result = result + (pw * coeff_mat[p].tolist())
    return result


def eval_poly_tenseal_folded(coeff_mat: np.ndarray, z_ct):
    """Depth-reduced alternative to eval_poly_tenseal (revision experiment E1, opt-in only).

    Each plaintext coefficient is multiplied into the degree-1 factor BEFORE the last
    ciphertext multiplication: c_p * z^p = (c_p * z) * z^(p-1). The coefficient multiply then
    runs in parallel with the power ladder instead of after it, so term p costs
    max(1, depth(z^(p-1))) + 1 levels rather than depth(z^p) + 1. For k=9 the deepest term,
    (c_9 z) * z^8, needs 4 levels instead of 5. Same polynomial, same coefficients, same power
    ladder (_ct_power); only the order of multiplications changes. Not Paterson-Stockmeyer:
    the number of ciphertext multiplications is not reduced."""
    degree = coeff_mat.shape[0] - 1
    cache = {1: z_ct}
    result = z_ct * coeff_mat[1].tolist()
    result = result + coeff_mat[0].tolist()
    for p in range(2, degree + 1):
        scaled = z_ct * coeff_mat[p].tolist()
        lower = _ct_power(z_ct, p - 1, cache)
        result = result + (scaled * lower)
    return result


def evaluate_circuit_tenseal(
    ctx,
    structure: TreeCircuitStructure,
    coeff_vectors: list[np.ndarray],
    x_row: pd.Series,
    norm_bounds: dict,
    a_half_width: float,
    poly_evaluator=None,
) -> tuple[float, dict]:
    """Real CKKS ciphertext evaluation of the SAME circuit as
    evaluate_circuit_numpy -- the encrypted-execution arm of Amendment
    v1.1 Sec 15.5 item 2. Client-side z computation (Sec 15.2) happens in
    the clear (as specified), THEN each level is encrypted, evaluated
    under real homomorphic operations, and only the final sum is
    decrypted. Returns (decrypted margin, timing breakdown in seconds)."""
    import time

    timing = {"encrypt": 0.0, "poly_eval": 0.0, "path_product": 0.0, "sum_decrypt": 0.0}
    z_levels = compute_level_inputs(structure, x_row, norm_bounds, a_half_width)

    level_cts = []
    for z in z_levels:
        t0 = time.perf_counter()
        level_cts.append(ts.ckks_vector(ctx, z.tolist()))
        timing["encrypt"] += time.perf_counter() - t0

    # Fold the leaf value into the LAST level's coefficients (pure plaintext
    # preprocessing, since leaf_value and coeff_vectors are both plaintext
    # at this point) so the ensemble-sum's leaf-scaling step costs ZERO
    # extra ciphertext depth, per Amendment Sec 15.2's "B_6's coefficient
    # vector is pre-multiplied by the frozen leaf value" design. Applying
    # leaf_value as a SEPARATE ct*pt multiply after the path product (the
    # first version of this function did that) silently exceeds the depth
    # budget at k>=5 -- found and fixed 2026-09-18.
    scaled_coeff_vectors = list(coeff_vectors)
    scaled_coeff_vectors[-1] = coeff_vectors[-1] * structure.leaf_value[None, :]

    branch_cts = []
    for coeff_mat, ct in zip(scaled_coeff_vectors, level_cts):
        t0 = time.perf_counter()
        branch_cts.append((poly_evaluator or eval_poly_tenseal)(coeff_mat, ct))
        timing["poly_eval"] += time.perf_counter() - t0

    def _balanced_product(cts):
        """Balanced binary-tree product: depth ceil(log2(len(cts))), not
        the depth-(n-1) a sequential fold would consume. This is the
        'balanced multiplication tree over 6 factors: depth ceil(log2(6))=3'
        M1-0 Sec 5's depth table assumes -- a sequential fold would have
        silently blown the depth budget instead."""
        if len(cts) == 1:
            return cts[0]
        mid = len(cts) // 2
        return _balanced_product(cts[:mid]) * _balanced_product(cts[mid:])

    t0 = time.perf_counter()
    output_ct = _balanced_product(branch_cts)
    timing["path_product"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    total = output_ct.sum()
    margin = total.decrypt()[0]
    timing["sum_decrypt"] = time.perf_counter() - t0

    return float(margin), timing


def evaluate_plaintext_exact(structure: TreeCircuitStructure, x_row: pd.Series) -> float:
    """The SAME leaf-aligned structure evaluated with EXACT branch
    indicators (hard threshold, not any polynomial) -- used only as a unit
    test that extract_ensemble_leaf_circuit correctly reproduces the
    original XGBoost tree, independent of k or any approximation."""
    total = 0.0
    n_leaves = structure.n_leaves
    for l in range(n_leaves):
        prod = 1.0
        for level_refs in structure.levels:
            ref = level_refs[l]
            if ref.kind == "pad":
                continue
            x_val = float(x_row[ref.feature])
            went_left = x_val < ref.threshold
            prod *= 1.0 if went_left == ref.is_yes_edge else 0.0
            if prod == 0.0:
                break
        total += prod * structure.leaf_value[l]
    return total
