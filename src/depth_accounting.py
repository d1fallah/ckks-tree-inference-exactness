"""
M1 -- Multiplicative-depth accounting DERIVED FROM CODE (closure guide 3.3).

The earlier depth table came from a closed form, poly_depth = ceil(log2(k+1)), asserted in a
docstring. That formula omits the plaintext-coefficient multiply the committed evaluator applies to
every computed power before accumulation, and it undercounts k=9 by one level. This module removes
the assertion: it RUNS the committed evaluator functions on a stand-in "ciphertext" that tracks only
its multiplicative level, and reads the depth off the result.

What is executed (not re-implemented):
  * ckks_circuit._ct_power        -- the balanced power ladder
  * ckks_circuit.eval_poly_tenseal -- accumulation with a plaintext coefficient multiply per power
The one exception is the path-product recursion, which is a nested function inside
ckks_circuit.evaluate_circuit_tenseal and therefore not importable. It is reproduced below verbatim
and a source check asserts that the committed function still contains the same recursion.

Level rules (verified against real ciphertexts by src/probe_multiplicative_depth.py):
  ciphertext x ciphertext  -> max(levels) + 1
  ciphertext x plaintext   -> level + 1        (a plaintext multiply also rescales, one level)
  ciphertext + anything    -> max(levels)      (addition costs no depth)

This file needs no TenSEAL at run time of the tracker itself, but importing ckks_circuit does.
"""
from __future__ import annotations

import inspect
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ckks_circuit as cc  # noqa: E402

TREE_DEPTH = 6  # number of factors in each leaf's path product (notation: tree depth)
FRESH_LEVEL = 0


class Tracked:
    """A stand-in ciphertext: records its multiplicative level and, for inspection, the operation that
    produced it. Plaintext operands are anything that is not a Tracked (the evaluator passes lists)."""

    log: list = []  # shared operation log for one walk; reset by `walk_*`

    def __init__(self, level: int, name: str):
        self.level, self.name = level, name

    def __mul__(self, other):
        if isinstance(other, Tracked):
            out = Tracked(max(self.level, other.level) + 1, f"({self.name}*{other.name})")
            Tracked.log.append({"op": "ct*ct", "a": self.name, "a_level": self.level,
                                "b": other.name, "b_level": other.level, "out_level": out.level})
        else:
            out = Tracked(self.level + 1, f"({self.name}*pt)")
            Tracked.log.append({"op": "ct*pt", "a": self.name, "a_level": self.level,
                                "out_level": out.level})
        return out

    def __add__(self, other):
        lvl = max(self.level, other.level) if isinstance(other, Tracked) else self.level
        return Tracked(lvl, f"({self.name}+{getattr(other, 'name', 'pt')})")


def balanced_product(cts):
    """Verbatim copy of the nested _balanced_product in ckks_circuit.evaluate_circuit_tenseal."""
    if len(cts) == 1:
        return cts[0]
    mid = len(cts) // 2
    return balanced_product(cts[:mid]) * balanced_product(cts[mid:])


def assert_product_recursion_matches_committed_code() -> None:
    """Guard against silent drift: the committed nested function must still contain this recursion."""
    src = inspect.getsource(cc.evaluate_circuit_tenseal)
    for needle in ("mid = len(cts) // 2",
                   "return _balanced_product(cts[:mid]) * _balanced_product(cts[mid:])"):
        assert needle in src, f"path-product recursion in ckks_circuit changed ({needle!r} missing): re-derive"


EVALUATORS = {"original": cc.eval_poly_tenseal, "folded": cc.eval_poly_tenseal_folded}


def walk_polynomial(degree: int, evaluator: str = "original") -> tuple[int, list]:
    """Run the COMMITTED polynomial evaluator (the original one by default, or the opt-in coefficient-folding
    one of revision E1) on a fresh tracked ciphertext; return (level, op log)."""
    Tracked.log = []
    coeff_mat = np.zeros((degree + 1, 1))  # only `.shape` and `.tolist()` are used by the evaluator
    out = EVALUATORS[evaluator](coeff_mat, Tracked(FRESH_LEVEL, "z"))
    return out.level, list(Tracked.log)


def walk_circuit(degree: int, n_factors: int = TREE_DEPTH, evaluator: str = "original") -> dict:
    """Full circuit for one leaf: `n_factors` polynomial outputs (one per tree level) combined by the
    balanced product. The leaf-value fold is a plaintext-only pre-multiplication of the last coefficient
    matrix (ckks_circuit.py 'scaled_coeff_vectors[-1] = ...'), so it adds no ciphertext depth; the final
    sum() is rotations and additions, also depth 0."""
    assert_product_recursion_matches_committed_code()
    poly_level, poly_ops = walk_polynomial(degree, evaluator)
    Tracked.log = []
    prod = balanced_product([Tracked(poly_level, f"P{i}") for i in range(n_factors)])
    product_ops = list(Tracked.log)
    return {
        "degree_k": degree,
        "evaluator": evaluator,
        "polynomial_depth": poly_level,
        "path_product_depth": prod.level - poly_level,
        "total_multiplicative_depth": prod.level,
        "polynomial_operation_graph": poly_ops,
        "path_product_operation_graph": product_ops,
    }


def closed_form_depth(degree: int, n_factors: int = TREE_DEPTH) -> dict:
    """The formula the earlier table used, kept only so the discrepancy is checkable."""
    poly = math.ceil(math.log2(degree + 1))
    path = math.ceil(math.log2(n_factors))
    return {"polynomial_depth": poly, "path_product_depth": path, "total_multiplicative_depth": poly + path}


def chain_levels_available(coeff_mod_bit_sizes) -> int:
    """Rescalings a chain sustains: one prime is consumed per level, and the first (data) and last
    (special) primes are not rescaling levels."""
    return len(coeff_mod_bit_sizes) - 2


if __name__ == "__main__":
    for k in (3, 5, 7, 9):
        w = walk_circuit(k)
        f = closed_form_depth(k)
        print(f"k={k}: code-derived poly={w['polynomial_depth']} path={w['path_product_depth']} "
              f"total={w['total_multiplicative_depth']}   closed-form total={f['total_multiplicative_depth']}")
