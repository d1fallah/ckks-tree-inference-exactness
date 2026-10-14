"""
M1 revision E1 -- code-derived multiplicative depth of the coefficient-folding evaluator.

Runs the committed evaluators (ckks_circuit.eval_poly_tenseal and eval_poly_tenseal_folded) on tracked
stand-in ciphertexts (depth_accounting) and asserts that the code-derived polynomial depth equals the depth
measured on real ciphertexts by probe_folded_evaluator_feasibility.py, as probe_multiplicative_depth.py does for
the original evaluator. Also reports whether each total depth fits the 7 levels of the fixed chain.
Needs no test data. Output: results/m1_folded_depth_accounting.json
"""
from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import depth_accounting as da  # noqa: E402
from ckks_instrumented import LOCKED_CHAIN  # noqa: E402
from notation import K_APPROX_DEGREE_VALUES  # noqa: E402

FEAS = os.path.join(_HERE, "..", "results", "m1_folded_evaluator_feasibility.json")
OUT = os.path.join(_HERE, "..", "results", "m1_folded_depth_accounting.json")


def main() -> int:
    with open(FEAS, "r", encoding="utf-8") as f:
        feas = json.load(f)["per_degree"]
    available = da.chain_levels_available(LOCKED_CHAIN)
    rows = {}
    for k in K_APPROX_DEGREE_VALUES:
        orig = da.walk_circuit(k, evaluator="original")
        fold = da.walk_circuit(k, evaluator="folded")
        measured = feas[f"k={k}"]
        assert orig["polynomial_depth"] == measured["poly_levels_original_evaluator"], f"k={k}: original depth differs from measurement"
        assert fold["polynomial_depth"] == measured["poly_levels_folded_evaluator"], f"k={k}: folded depth differs from measurement"
        def counts(walk):  # ciphertext operations of ONE polynomial, read off the evaluator's operation graph
            ops = walk["polynomial_operation_graph"]
            return {"ct_x_ct": sum(o["op"] == "ct*ct" for o in ops), "ct_x_pt": sum(o["op"] == "ct*pt" for o in ops)}
        rows[f"k={k}"] = {
            "polynomial_operations_original": counts(orig), "polynomial_operations_folded": counts(fold),
            "polynomial_depth_original": orig["polynomial_depth"], "polynomial_depth_folded": fold["polynomial_depth"],
            "total_depth_original": orig["total_multiplicative_depth"], "total_depth_folded": fold["total_multiplicative_depth"],
            "levels_available": available,
            "fits_original": orig["total_multiplicative_depth"] <= available,
            "fits_folded": fold["total_multiplicative_depth"] <= available,
            "measured_polynomial_levels_original": measured["poly_levels_original_evaluator"],
            "measured_polynomial_levels_folded": measured["poly_levels_folded_evaluator"],
            "code_derived_equals_measured": True,
        }
        c0, c1 = counts(orig), counts(fold)
        print(f"   operations per polynomial: original {c0}  folded {c1}")
        print(f"k={k}: poly {orig['polynomial_depth']}->{fold['polynomial_depth']}  total {orig['total_multiplicative_depth']}->"
              f"{fold['total_multiplicative_depth']}  (levels available {available})")
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"purpose": "revision E1: code-derived depth of the folding evaluator vs real-ciphertext measurement",
                   "measured_source": "results/m1_folded_evaluator_feasibility.json", "per_degree": rows}, f, indent=2)
    print(f"written -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
