#!/usr/bin/env python3
r"""
Regression test of the mock and the counting rules in src/m1c_formal_counts.py against values worked out by hand (no model file, no data).
A rule that is broken here would change the depth, multiplication and rotation counts the manuscript derives. Run in WSL: python src/test_m1c_formal_counts.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import ckks_circuit as cc  # noqa: E402
import m1c_formal_counts as fcn  # noqa: E402
import m1c_thermo as th  # noqa: E402


def test_sum_rotations() -> None:
    # TenSEAL sum_vector: n = 1 none; power of two 2^m: m rotations; otherwise m for the lower power 2^m, plus one rotation and the recursion on the rest.
    expected = {1: 0, 2: 1, 4: 2, 8192: 13, 3: 2, 5: 3, 6: 2 + 1 + 1, 7: 2 + 1 + 2, 9: 3 + 1 + 0}
    for n, r in expected.items():
        assert fcn.sum_rotations(n) == r, (n, fcn.sum_rotations(n), r)
    assert fcn.below_power2(1) == 1 and fcn.below_power2(5) == 4 and fcn.below_power2(8) == 8 and fcn.below_power2(2751) == 2048


def test_nonzero_diagonals() -> None:
    # Square 3 x 3 matrix: the wrapped diagonal i holds M[(i + t) mod 3, t mod 3] for t < 9.
    # The identity is nonzero only on diagonal 0; one element at row 1, column 0 only on diagonal 1; the zero matrix on none.
    assert fcn.nonzero_diagonals(np.eye(3), 8192) == [0]
    M = np.zeros((3, 3))
    M[1, 0] = 1.0
    assert fcn.nonzero_diagonals(M, 8192) == [1]
    M[0, 0] = 1.0
    assert fcn.nonzero_diagonals(M, 8192) == [0, 1]
    assert fcn.nonzero_diagonals(np.zeros((4, 5)), 8192) == []
    # a rectangular 2 x 3 matrix wraps both ways: the element (1, 2) is met on diagonals 0 (at t = 5) and 1 (at t = 2)
    R = np.zeros((2, 3))
    R[1, 2] = 1.0
    assert fcn.nonzero_diagonals(R, 8192) == [0, 1]


def _mock(size: int, level: int = 0):
    ops = fcn.Ops()
    return fcn.MockCt(ops, size, level), ops


def test_levels() -> None:
    z, ops = _mock(8)
    assert (z * z).level == 1 and (z * ([1.0] * 8)).level == 1 and ((z * z) + [0.0]).level == 1
    # power ladder: depth of z^p is ceil(log2 p)
    for p, d in ((1, 0), (2, 1), (3, 2), (4, 2), (5, 3), (8, 3), (9, 4), (14, 4), (30, 5)):
        z, ops = _mock(8)
        r = cc._ct_power(z, p, {1: z})
        assert r.level == d, (p, r.level, d)
    # folding evaluator on degree k: deepest term (c_k z) * z^(k-1) has max(1, ceil(log2(k-1))) + 1 levels: 4, 5, 6 for k = 9, 15, 31
    for k, d in ((9, 4), (15, 5), (31, 6)):
        z, ops = _mock(8)
        coeff = np.ones((k + 1, 8))
        out = cc.eval_poly_tenseal_folded(coeff, z)
        assert out.level == d and ops.max_level == d, (k, out.level, ops.max_level, d)
    # balanced product over six factors: three levels, five ciphertext multiplications
    ops = fcn.Ops()
    cts = [fcn.MockCt(ops, 8) for _ in range(6)]
    r = th._balanced_product(cts)
    assert r.level == 3 and ops.ct_ct == 5


def test_mm_and_sum() -> None:
    z, ops = _mock(3)
    M = np.zeros((3, 3))
    M[0, 0] = 1.0
    M[1, 0] = 1.0   # nonzero diagonals 0 and 1
    r = z.mm(M)
    assert r.level == 1 and r.size == 3
    assert ops.ct_plain == 2 and ops.rotations == 1 and ops.rot_mm == 1  # diagonals 0 and 1; only the step 1 is a rotation call
    s = r.sum()
    assert s.size == 1 and ops.rot_sum == 2 and ops.rotations == 3  # size 3: two rotation calls


def main() -> int:
    test_sum_rotations()
    test_nonzero_diagonals()
    test_levels()
    test_mm_and_sum()
    print("ALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
