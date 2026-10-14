"""
M1 -- Multiplicative-depth budget for k=1, as a committed artifact (paper prep, Table 4 / Figure 7).

k=1 is documented today only in docs/k1_compatibility.md's prose ("polynomial 1 + path product 3 = 4"),
which is not machine-checkable against the committed evaluator. This script runs the SAME committed
depth-accounting walk (src/depth_accounting.py:walk_circuit) already used for k in {3,5,7,9} by
src/probe_multiplicative_depth.py, at degree_k=1, and writes the result as its own artifact so
src/paper_data.py has something to point a table/figure cell at.

This does NOT re-derive or touch the locked approximation interval, gamma, or coefficients (unlike
src/derive_circuit_params.py, which must not be re-run -- see M1_CLOSURE_LOG.md and the paper-prep
plan). It is a pure code derivation over src/depth_accounting.py: no data, no model, no random sample.

Output: results/m1_depth_budget_k1.json
"""
from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import depth_accounting as da  # noqa: E402

OUT_PATH = os.path.join(_HERE, "..", "results", "m1_depth_budget_k1.json")
LOCKED_CHAIN = [60, 40, 40, 40, 40, 40, 40, 40, 60]  # identical to every other locked-chain script
TC128_MAX_BITS_AT_N16384 = 438


def main() -> int:
    da.assert_product_recursion_matches_committed_code()
    degree = 1
    derived = da.walk_circuit(degree)
    closed = da.closed_form_depth(degree)
    available = da.chain_levels_available(LOCKED_CHAIN)
    total = derived["total_multiplicative_depth"]
    coeff_bits = LOCKED_CHAIN[0] + 40 * total + LOCKED_CHAIN[-1]
    out = {
        "protocol": "paper-prep code derivation; NOT a new experiment. Runs the same committed "
                    "depth_accounting.walk_circuit used for k in {3,5,7,9} (src/probe_multiplicative_depth.py) "
                    "at degree_k=1. No locked parameter (interval, gamma, coefficients) is touched or re-derived.",
        "degree_k": degree,
        "additional_diagnostic_arm": True,
        "locked_family": [3, 5, 7, 9],
        "code_derived": {
            "polynomial_depth": derived["polynomial_depth"],
            "path_product_depth": derived["path_product_depth"],
            "total_multiplicative_depth": total,
        },
        "closed_form_previously_used_formula": closed,
        "levels_available_on_locked_chain": available,
        "fits_locked_chain": bool(total <= available),
        "slack_levels": available - total,
        "coeff_modulus_bits_needed_for_this_depth": coeff_bits,
        "tc128_max_bits_at_N16384": TC128_MAX_BITS_AT_N16384,
        "would_fit_N16384_at_128_bit_security": bool(coeff_bits <= TC128_MAX_BITS_AT_N16384),
        "polynomial_operation_graph": derived["polynomial_operation_graph"],
        "path_product_operation_graph": derived["path_product_operation_graph"],
        "cross_reference": "docs/k1_compatibility.md (utility result, simulated); "
                            "results/m1_depth_probe.json (the same walk at k in {3,5,7,9}, "
                            "cross-checked there against real ciphertexts)",
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"k=1: total depth {total} (poly {derived['polynomial_depth']} + path {derived['path_product_depth']}), "
          f"{available} levels available, slack {available - total:+d}")
    print(f"written -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
