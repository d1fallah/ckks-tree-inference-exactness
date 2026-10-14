#!/usr/bin/env python3
r"""
Formal counts for the manuscript's cost proposition (added analysis, DEVIATIONS.md entry "formal counts"): multiplicative depth, ciphertext counts, plaintext and
ciphertext multiplications and rotation calls of the broadcast thermometer, packed thermometer and exact interpolation circuits, derived by running the REAL evaluation
functions (m1c_thermo.encrypt_bits / eval_thermo_tenseal / eval_thermo_packed_tenseal and ckks_circuit.evaluate_circuit_tenseal with the folding evaluator) on a
depth-tracking mock ciphertext, for the five saved boosters of each bin count. Nothing is encrypted; no test-split row, no patient data and no per-stay value is read; the
inputs are the saved model files (split thresholds and leaf values) and an all-zero input row, whose values do not affect any count.

The mock follows the rules of TenSEAL/SEAL that the circuits rely on:
  ciphertext * ciphertext   : one level more than the deeper operand, one ciphertext multiplication
  ciphertext * plaintext    : one level (plaintext multiplication and rescale)
  ciphertext + (ct|plain)   : no level
  sum()                     : TenSEAL's rotate-and-add (utils.cpp sum_vector, read from source, not executed here), no level
  mm(M)                     : TenSEAL's diagonal method (encrypted_vector.h diagonal_ct_vector_matmul, plain_tensor.h get_diagonal; read, not executed):
                              for each i in [0, vector size) with a nonzero (wrapped) diagonal, one plaintext multiplication and, for i >= 1, one rotation
                              call; one level.
Rotation calls are counted as calls (each is realised by one or more key switches on TenSEAL's default power-of-two Galois keys; the number of switches is not counted).
Output: results/m1c_formal_counts.json (aggregate)
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
import m1c_thermo as th  # noqa: E402
import m1c_thermo_final_common as tc  # noqa: E402
from m1c_final_run import build_exact_circuit  # noqa: E402

SLOT_COUNT = 8192  # ring degree 16,384
OUT = os.path.join(fc.RESULTS, "m1c_formal_counts.json")


def below_power2(n: int) -> int:
    """TenSEAL utils.cpp: n if n is a power of two, else the largest power of two below n."""
    assert n >= 1
    if n & (n - 1) == 0:
        return n
    return 1 << (n.bit_length() - 1)


def sum_rotations(n: int) -> int:
    """Rotation calls of TenSEAL's sum_vector for a vector of n slots (utils.cpp)."""
    if n == 1:
        return 0
    bp2 = below_power2(n)
    r = bp2.bit_length() - 1  # the loop i = bp2/2, ..., 1 rotates log2(bp2) times
    if bp2 != n:
        r += 1 + sum_rotations(n - bp2)
    return r


def nonzero_diagonals(M: np.ndarray, slot_count: int = SLOT_COUNT) -> list[int]:
    """Indices i in [0, rows) of the diagonals that TenSEAL's diagonal_ct_vector_matmul does not skip: get_diagonal(-i, slot_count) has an element != 0."""
    n_rows, n_cols = M.shape
    size = min(slot_count, n_rows * n_cols)
    t = np.arange(size)
    out = []
    for i in range(n_rows):
        if np.any(M[(i + t) % n_rows, t % n_cols] != 0):
            out.append(i)
    return out


class Ops:
    def __init__(self):
        self.ct_ct = 0
        self.ct_plain = 0
        self.rotations = 0
        self.rot_mm = 0
        self.rot_sum = 0
        self.max_level = 0
        self.created = 0
        self.diagonals_per_mm = []


class MockCt:
    def __init__(self, ops: Ops, size: int, level: int = 0):
        self.ops, self.size, self.level = ops, size, level
        ops.created += 1

    def _new(self, size, level):
        r = MockCt.__new__(MockCt)
        r.ops, r.size, r.level = self.ops, size, level
        self.ops.max_level = max(self.ops.max_level, level)
        return r

    def __mul__(self, other):
        if isinstance(other, MockCt):
            self.ops.ct_ct += 1
            return self._new(max(self.size, other.size), max(self.level, other.level) + 1)
        self.ops.ct_plain += 1
        return self._new(self.size, self.level + 1)

    def __add__(self, other):
        if isinstance(other, MockCt):
            return self._new(max(self.size, other.size), max(self.level, other.level))
        return self._new(self.size, self.level)

    def sum(self):
        r = sum_rotations(self.size)
        self.ops.rotations += r
        self.ops.rot_sum += r
        return self._new(1, self.level)

    def mm(self, M):
        M = np.asarray(M)
        assert M.shape[0] == self.size, "TenSEAL requires the vector size to equal the matrix row count"
        diag = nonzero_diagonals(M)
        self.ops.ct_plain += len(diag)
        r = sum(1 for i in diag if i >= 1)
        self.ops.rotations += r
        self.ops.rot_mm += r
        self.ops.diagonals_per_mm.append(len(diag))
        return self._new(int(M.shape[1]), self.level + 1)

    def decrypt(self):
        return [0.0]


def run_with_mock(fn):
    """Run fn(ops) with tenseal.ckks_vector replaced by a MockCt factory; return (result, ops)."""
    import tenseal as ts
    ops = Ops()
    real = ts.ckks_vector
    ts.ckks_vector = lambda ctx, vec, *a, **k: MockCt(ops, len(vec))
    try:
        res = fn(ops)
    finally:
        ts.ckks_vector = real
    return res, ops


def record(ops: Ops) -> dict:
    """`created` counts the ciphertexts the client's function creates (one tenseal.ckks_vector call each); every server-side operation yields a new object without it."""
    return {"client_ciphertexts": int(ops.created), "levels": int(ops.max_level), "ct_ct_multiplications": int(ops.ct_ct),
            "ct_plain_multiplications": int(ops.ct_plain), "rotation_calls": int(ops.rotations),
            "rotation_calls_in_matrix_product": int(ops.rot_mm), "rotation_calls_in_sum": int(ops.rot_sum)}


def interpolant_check(B: int) -> dict:
    """Double-precision solve of the interpolation system of m1c_final_common.exact_odd_step (same formulas, without its assertion): residual at the 2B nodes and the
    condition number of the B x B system. fc.exact_odd_step asserts a residual below 1e-8."""
    u = (np.arange(B) + 0.5) / (B - 0.5)
    powers = np.arange(1, 2 * B, 2)
    A = u[:, None] ** powers[None, :]
    c = np.linalg.solve(A, np.ones(B))
    p = np.zeros(2 * B)
    p[powers] = c
    h = -0.5 * p
    h[0] += 0.5
    hv = np.polynomial.polynomial.polyval
    resid = float(max(np.max(np.abs(hv(u, h))), np.max(np.abs(hv(-u, h) - 1.0))))
    return {"nodes": 2 * B, "degree": 2 * B - 1, "max_node_residual": resid, "condition_number": float(np.linalg.cond(A)),
            "passes_code_assertion_1e-8": bool(resid < 1e-8)}


def counts_for(seed: int, B: int) -> dict:
    with open(tc.params_path(seed, B), "r", encoding="utf-8") as f:
        par = json.load(f)
    feats = par["feats"]
    F = len(feats)
    bst = xgb.Booster()
    bst.load_model(tc.booster_path(seed, B))
    structure = cc.extract_ensemble_leaf_circuit(bst, set(feats), tree_depth=6)
    levels = th.level_arrays(structure, feats, B)
    L = int(structure.leaf_value.shape[0])
    bits_row = np.zeros(F * (B - 1))
    res = {"leaf_slots": L, "n_features": F, "bit_columns": F * (B - 1),
           "used_bit_columns_per_level": [len(u) for u in th.used_bit_columns(levels, B)]}
    # broadcast: the real client and server functions
    pv = th.folded_plain_vectors(levels, structure.leaf_value, B)

    def f_b(ops):
        cts = th.encrypt_bits(None, bits_row)
        th.eval_thermo_tenseal(pv, cts)
    _, ops = run_with_mock(f_b)
    res["broadcast"] = record(ops)
    # packed
    mats = th.build_packed(levels, structure.leaf_value, B, F)

    def f_p(ops):
        th.eval_thermo_packed_tenseal(None, mats, bits_row)
    _, ops = run_with_mock(f_p)
    rec = record(ops)
    rec["nonzero_diagonals_per_level"] = [int(x) for x in ops.diagonals_per_mm]
    res["packed"] = rec
    # exact interpolation (folding evaluator), the real evaluation function
    exact = True
    try:
        so, ss, norm, coeffs, k = build_exact_circuit(bst, feats, B)
    except AssertionError as e:
        # exact_odd_step asserts that the double-precision interpolant is exact at its nodes; at this B it is not. The depth and the counts do not depend on the
        # coefficient values, so they are taken from the same evaluator with placeholder coefficients, and the record says so.
        assert "interpolant is not exact" in str(e), e
        exact = False
        k = 2 * B - 1
        so = cc.extract_ensemble_leaf_circuit(bst, set(feats), tree_depth=6)
        ss = so
        norm = {c: {"min": 0.0, "max": B - 0.5} for c in feats}
        coeffs = [np.ones((k + 1, so.n_leaves)) for _ in range(6)]
    x_row = pd.Series({c: 0.0 for c in feats})

    def f_i(ops):
        cc.evaluate_circuit_tenseal(None, ss, coeffs, x_row, norm, 1.0, poly_evaluator=cc.eval_poly_tenseal_folded)
    _, ops = run_with_mock(f_i)
    res["interpolation"] = record(ops)
    res["interpolation"]["degree"] = int(k)
    res["interpolation"]["exact_interpolant_constructed"] = exact
    return res


def main() -> int:
    out = {"note": ("docs/DEVIATIONS.md entry 'formal counts' (added). Counts from running the real evaluation functions on a depth-tracking mock; "
                    "rules of the mock in the module docstring. Saved model files only; no patient data."),
           "slot_count": SLOT_COUNT, "seeds": list(fc.SEEDS), "by_B": {}}
    for B in tc.LEVELS_B:
        k = f"B={B}"
        out["by_B"][k] = {"seeds": {}}
        for s in fc.SEEDS:
            r = counts_for(s, B)
            out["by_B"][k]["seeds"][str(s)] = r
            print(f"seed {s} {k}: L={r['leaf_slots']}  broadcast levels {r['broadcast']['levels']} ct={r['broadcast']['client_ciphertexts']} "
                  f"rot={r['broadcast']['rotation_calls']} ctpl={r['broadcast']['ct_plain_multiplications']} | packed levels {r['packed']['levels']} "
                  f"rot={r['packed']['rotation_calls']} (mm {r['packed']['rotation_calls_in_matrix_product']}) | interp levels {r['interpolation']['levels']} "
                  f"k={r['interpolation']['degree']} ctct={r['interpolation']['ct_ct_multiplications']}", flush=True)
        out["by_B"][k]["interpolant_check"] = interpolant_check(B)
        # design constants that do not vary over seeds
        first = out["by_B"][k]["seeds"][str(fc.SEEDS[0])]
        out["by_B"][k]["design"] = {"bit_columns": first["bit_columns"], "levels": {c: first[c]["levels"] for c in ("broadcast", "packed", "interpolation")},
                                    "client_ciphertexts": {c: first[c]["client_ciphertexts"] for c in ("broadcast", "packed", "interpolation")},
                                    "interpolation_degree": first["interpolation"]["degree"]}
        for c in ("broadcast", "packed", "interpolation"):
            vals = {tuple(v[c][x] for x in ("levels", "client_ciphertexts")) for v in out["by_B"][k]["seeds"].values()}
            assert len(vals) == 1, f"{k} {c}: depth or ciphertext count varies over seeds: {vals}"
    # measured upload ratio, from the stored same-chain summary
    sp = os.path.join(fc.RESULTS, "m1c_thermo_samechain_summary.json")
    if os.path.exists(sp):
        with open(sp, "r", encoding="utf-8") as f:
            sm = json.load(f)
        for B in tc.LEVELS_B:
            k = f"B={B}"
            ub = sm["by_B"][k]["circuits"]["broadcast"]["client_upload_mb"]
            up = sm["by_B"][k]["circuits"]["packed"]["client_upload_mb"]
            out["by_B"][k]["measured_upload_ratio_broadcast_to_packed"] = ub / up
            out["by_B"][k]["expected_ratio_bit_columns"] = out["by_B"][k]["design"]["bit_columns"]
            if "interpolation" in sm["by_B"][k]["circuits"]:
                ui = sm["by_B"][k]["circuits"]["interpolation"]["client_upload_mb"]
                out["by_B"][k]["measured_upload_ratio_interpolation_to_packed"] = ui / up
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"written -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
