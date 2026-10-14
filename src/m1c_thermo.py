"""
Model-private thermometer-bit circuit for the M1 continuous-feature task (GATES.md Question gate, "model-private thermometer protocol").

Protocol. Inputs are integer bin indices x_f in 0..B-1. The client sends only thermometer bits b[f, m] = [x_f >= m] for every feature f and cut
m = 1..B-1: a fixed layout that does not depend on the model, so the client learns neither the tree structure nor the thresholds. The server holds
the masks and the leaf values. For an integer x and a threshold t, x < t holds exactly when x < c with c = ceil(t), and:
    1 <= c <= B-1 :  yes-branch = 1 - b[f, c],   no-branch = b[f, c]
    c <= 0        :  yes-branch = 0,             no-branch = 1          (constants; b[f, 0] is never sent)
    c >= B        :  yes-branch = 1,             no-branch = 0
A padded level is the constant 1. So every leaf's factor at level d is  alpha + beta * b[f, c]  with alpha in {0, 1} and beta in {-1, 0, 1}
(`level_arrays`). The server forms each level as  alpha_d + sum_j (beta_d * mask_j) * ct_j  (one plaintext multiplication level), multiplies the six
level factors with the existing balanced product (3 levels), and sums the slots. The leaf values are folded into the last level's plaintext vectors,
so they cost no level. Total: 4 levels. No polynomial, so the circuit is exact up to CKKS noise.

This module holds the construction, a NumPy simulator, and the real-ciphertext evaluators of the three protocols compared in the gate (broadcast
thermometer, client-computed indicators). Nothing here reads patient data.
"""
from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import ckks_circuit as cc  # noqa: E402

N_SLOTS = 8192  # ring degree 16,384 -> 8,192 slots


@dataclass
class LevelArrays:
    alpha: np.ndarray   # (L,) constant part
    beta: np.ndarray    # (L,) coefficient of the selected bit, in {-1, 0, 1}
    feat: np.ndarray    # (L,) feature index (0 where beta == 0)
    cut: np.ndarray     # (L,) cut m in 1..B-1 (0 where beta == 0)


def branch_coefficients(is_yes: bool, c: int, B: int) -> tuple[int, int, int]:
    """(alpha, beta, cut) for one split edge on integer inputs 0..B-1 with c = ceil(t)."""
    if c <= 0:
        return (0, 0, 0) if is_yes else (1, 0, 0)
    if c >= B:
        return (1, 0, 0) if is_yes else (0, 0, 0)
    return (1, -1, c) if is_yes else (0, 1, c)


def level_arrays(structure: "cc.TreeCircuitStructure", feats: list[str], B: int) -> list[LevelArrays]:
    col = {c: i for i, c in enumerate(feats)}
    out = []
    for lvl in structure.levels:
        L = len(lvl)
        a, b, f, m = (np.zeros(L, dtype=np.int64) for _ in range(4))
        for l, ref in enumerate(lvl):
            if ref.kind == "pad":
                a[l] = 1
                continue
            c = math.ceil(ref.threshold - 1e-9)
            al, be, cu = branch_coefficients(bool(ref.is_yes_edge), c, B)
            a[l], b[l], m[l] = al, be, cu
            f[l] = col[ref.feature] if be != 0 else 0
        out.append(LevelArrays(a, b, f, m))
    return out


def thermometer_bits(Xb: np.ndarray, B: int) -> np.ndarray:
    """(n, F) integer bin indices -> (n, F*(B-1)) bits; column f*(B-1) + (m-1) holds [x_f >= m]."""
    n, F = Xb.shape
    bits = np.zeros((n, F * (B - 1)), dtype=np.float64)
    for m in range(1, B):
        bits[:, np.arange(F) * (B - 1) + (m - 1)] = (Xb >= m)
    return bits


def simulate_margin(levels: list[LevelArrays], leaf_value: np.ndarray, bits: np.ndarray, B: int, chunk: int = 4000) -> np.ndarray:
    """Exact NumPy evaluation of the thermometer circuit for a matrix of bit vectors."""
    n = bits.shape[0]
    out = np.empty(n)
    for s in range(0, n, chunk):
        bc = bits[s : s + chunk]
        prod = np.ones((bc.shape[0], leaf_value.shape[0]))
        for lv in levels:
            col = lv.feat * (B - 1) + np.maximum(lv.cut - 1, 0)
            sel = bc[:, col] * (lv.beta != 0)[None, :]
            prod *= lv.alpha[None, :] + lv.beta[None, :] * sel
        out[s : s + chunk] = prod @ leaf_value
    return out


def used_bit_columns(levels: list[LevelArrays], B: int) -> list[set[int]]:
    """For each level, the set of bit columns (f*(B-1) + m-1) its non-constant slots select."""
    res = []
    for lv in levels:
        nz = lv.beta != 0
        res.append(set((lv.feat[nz] * (B - 1) + lv.cut[nz] - 1).tolist()))
    return res


def folded_plain_vectors(levels: list[LevelArrays], leaf_value: np.ndarray, B: int, n_slots: int = N_SLOTS):
    """Per level: the plaintext constant vector and, per used bit column, the plaintext mask vector (beta on the slots that select it),
    all zero-padded to n_slots. The leaf values are folded into the LAST level. Pad slots beyond the leaves have alpha = 0 in every
    level, so they contribute nothing to the final sum."""
    L = leaf_value.shape[0]
    assert L <= n_slots
    out = []
    for d, lv in enumerate(levels):
        scale = leaf_value if d == len(levels) - 1 else np.ones(L)
        const = np.zeros(n_slots)
        const[:L] = lv.alpha * scale
        masks = {}
        nz = np.flatnonzero(lv.beta != 0)
        cols = lv.feat[nz] * (B - 1) + lv.cut[nz] - 1
        for j in np.unique(cols):
            vec = np.zeros(n_slots)
            sl = nz[cols == j]
            vec[sl] = lv.beta[sl] * scale[sl]
            masks[int(j)] = vec
        out.append((const, masks))
    return out


def _balanced_product(cts):
    if len(cts) == 1:
        return cts[0]
    mid = len(cts) // 2
    return _balanced_product(cts[:mid]) * _balanced_product(cts[mid:])


def encrypt_bits(ctx, bits_row: np.ndarray, n_slots: int = N_SLOTS):
    """Client side of the broadcast protocol: one ciphertext per bit, the bit repeated in every slot."""
    import tenseal as ts
    return [ts.ckks_vector(ctx, [float(b)] * n_slots) for b in bits_row]


def eval_thermo_tenseal(plain, cts, n_slots: int = N_SLOTS) -> tuple[float, dict]:
    """Server side plus decryption for the broadcast protocol. `plain` = folded_plain_vectors(...), `cts` = encrypt_bits(...)."""
    timing = {"select": 0.0, "path_product": 0.0, "sum_decrypt": 0.0}
    t0 = time.perf_counter()
    factors = []
    for const, masks in plain:
        acc = None
        for j, vec in masks.items():
            term = cts[j] * vec.tolist()
            acc = term if acc is None else acc + term
        factors.append(acc + const.tolist())
    timing["select"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    prod = _balanced_product(factors)
    timing["path_product"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    margin = float(prod.sum().decrypt()[0])
    timing["sum_decrypt"] = time.perf_counter() - t0
    return margin, timing


def client_indicators(levels: list[LevelArrays], bits: np.ndarray, B: int) -> np.ndarray:
    """Baseline protocol (client knows the model): the client computes every leaf's path indicator in plaintext. (n, L)."""
    n = bits.shape[0]
    L = levels[0].alpha.shape[0]
    prod = np.ones((n, L))
    for lv in levels:
        col = lv.feat * (B - 1) + np.maximum(lv.cut - 1, 0)
        sel = bits[:, col] * (lv.beta != 0)[None, :]
        prod *= lv.alpha[None, :] + lv.beta[None, :] * sel
    return prod


def eval_client_indicator_tenseal(ctx, indicator_row: np.ndarray, leaf_value: np.ndarray, n_slots: int = N_SLOTS) -> tuple[float, dict]:
    """Baseline protocol on real ciphertexts: the client encrypts its path indicators, the server multiplies by the leaf values and sums."""
    import tenseal as ts
    timing = {}
    t0 = time.perf_counter()
    vec = np.zeros(n_slots)
    vec[: indicator_row.shape[0]] = indicator_row
    ct = ts.ckks_vector(ctx, vec.tolist())
    timing["encrypt"] = time.perf_counter() - t0
    lv = np.zeros(n_slots)
    lv[: leaf_value.shape[0]] = leaf_value
    t0 = time.perf_counter()
    out = (ct * lv.tolist()).sum()
    timing["server"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    margin = float(out.decrypt()[0])
    timing["decrypt"] = time.perf_counter() - t0
    return margin, timing


def serialized_bytes(cts) -> int:
    return int(sum(len(c.serialize()) for c in cts))


def build_packed(levels: list[LevelArrays], leaf_value: np.ndarray, B: int, n_features: int):
    L = leaf_value.shape[0]
    nb = n_features * (B - 1)
    mats = []
    for d, lv in enumerate(levels):
        scale = leaf_value if d == len(levels) - 1 else np.ones(L)
        M = np.zeros((nb, L))
        nz = np.flatnonzero(lv.beta != 0)
        cols = lv.feat[nz] * (B - 1) + lv.cut[nz] - 1
        M[cols, nz] = lv.beta[nz] * scale[nz]
        mats.append((M.tolist(), (lv.alpha * scale).tolist()))
    return mats


def eval_thermo_packed_tenseal(ctx, mats, bits_row: np.ndarray) -> tuple[float, dict]:
    """Packed protocol on real ciphertexts: one client ciphertext, one plaintext matrix product per level (one level), balanced product, sum."""
    import tenseal as ts
    timing = {}
    t0 = time.perf_counter()
    ct = ts.ckks_vector(ctx, [float(b) for b in bits_row])
    timing["encrypt"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    factors = [ct.mm(M) + const for M, const in mats]
    timing["select"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    prod = _balanced_product(factors)
    timing["path_product"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    margin = float(prod.sum().decrypt()[0])
    timing["sum_decrypt"] = time.perf_counter() - t0
    return margin, timing
