"""
M1 -- Interpretation table for the independent LWE estimate (closure guide 4.2). Pure post-processing of two committed artifacts; needs no Sage.

results/m1_lwe_security_estimate.json (raw estimator output for the locked chain and the Standard's 438-bit reference row) and
results/m1_lwe_toolchain_sanity.json (how the same estimator and cost models score the Standard's five PUBLISHED 128-bit rows) are combined
under a rule fixed here, in code, so the reading is reproducible and not a matter of prose:

  * A cost model is "on the HE Standard's scale" iff it scores ALL five published 128-bit rows within TOL_BITS (5 bits) of 128.
  * For a model that is on the Standard's scale, the locked chain "meets 128 bits" iff its estimate >= 128 (the raw threshold applies).
  * For a model that is NOT on the Standard's scale, its absolute number is NOT compared with 128; only the margin over the Standard's own
    438-bit row, scored by the same model, is reported.

Output: results/m1_lwe_security_summary.json
"""
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda *p: os.path.join(_HERE, "..", *p)  # noqa: E731
TOL_BITS = 5.0
TARGET = 128


def main() -> int:
    est = json.load(open(R("results", "m1_lwe_security_estimate.json"), encoding="utf-8"))
    san = json.load(open(R("results", "m1_lwe_toolchain_sanity.json"), encoding="utf-8"))
    models = sorted({r["cost_model"] for r in est["runs"]})
    out = {"rule": {"on_standard_scale_iff": f"all five published 128-bit rows scored within {TOL_BITS} bits of {TARGET}",
                    "meets_128_iff": "on the Standard's scale AND locked-chain estimate >= 128; off-scale models are not compared with 128"},
           "estimator_commit": est["estimator"]["commit"], "models": {}}
    for m in models:
        locked = next(r for r in est["runs"] if r["cost_model"] == m and r["parameter_set"].startswith("locked"))
        ref = next(r for r in est["runs"] if r["cost_model"] == m and r["parameter_set"].startswith("HE Standard"))
        key = "ADPS16 classical core-SVP" if m.startswith("ADPS16") else "MATZOV (estimator default)"
        gaps = [row[key]["gap_to_published_128"] for row in san["rows"]]
        on_scale = all(abs(g) <= TOL_BITS for g in gaps)
        out["models"][m] = {
            "locked_chain_bits": locked["estimated_security_bits"], "locked_chain_cheapest_attack": locked["cheapest_attack"],
            "standard_438_bit_row_bits_same_model": ref["estimated_security_bits"],
            "margin_over_standard_row_bits": round(locked["estimated_security_bits"] - ref["estimated_security_bits"], 3),
            "published_128bit_rows_scored_gap_bits": gaps, "on_standard_scale": on_scale,
            "locked_chain_meets_128": (bool(locked["estimated_security_bits"] >= TARGET) if on_scale else None),
            "reading": (f"on the Standard's scale (all published rows within {TOL_BITS} bits of 128); locked chain {'meets' if locked['estimated_security_bits'] >= TARGET else 'does NOT meet'} 128"
                        if on_scale else
                        "NOT on the Standard's scale (undershoots the published rows by "
                        f"{min(-g for g in gaps):.0f}-{max(-g for g in gaps):.0f} bits); its absolute number is not compared with 128; "
                        f"locked chain is {locked['estimated_security_bits'] - ref['estimated_security_bits']:+.1f} bits relative to the Standard's own 438-bit row"),
        }
    with open(R("results", "m1_lwe_security_summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    for m, v in out["models"].items():
        print(f"{m}\n  locked {v['locked_chain_bits']} bits | Standard 438-bit row {v['standard_438_bit_row_bits_same_model']} | margin {v['margin_over_standard_row_bits']:+.1f}\n  {v['reading']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
