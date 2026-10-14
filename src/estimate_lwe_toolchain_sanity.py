"""
M1 -- Toolchain sanity check for the independent LWE estimate (closure guide 4.2).

The estimate in estimate_lwe_security.py reports very different absolute numbers under two cost models (raw classical core-SVP vs the
estimator's default MATZOV model). Before interpreting either, this checks how the SAME estimator and the SAME two cost models score rows
that the HE Security Standard PUBLISHES as 128-bit (ternary secret, sigma = 3.2, classical): n = 1024 / 2048 / 4096 / 8192 / 16384 with
max log2(q) = 27 / 54 / 109 / 218 / 438. If a model reproduces the published label on those rows, its number for the locked chain is on the
Standard's scale; if it does not, that model's absolute number must not be compared with "128".

Runs in the Sage venv (~/.venvs/m1-estimator) against the pinned lattice-estimator checkout. Output: results/m1_lwe_toolchain_sanity.json
"""
import json
import math
import os
import subprocess
import sys
import time

EST_DIR = os.path.expanduser("~/lattice-estimator")
sys.path.insert(0, EST_DIR)
from estimator import LWE, ND, RC  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(_HERE, "..", "results", "m1_lwe_toolchain_sanity.json")
SIGMA = 3.2
HES_ROWS = [(1024, 27), (2048, 54), (4096, 109), (8192, 218), (16384, 438)]  # (n, max log2 q at 128-bit classical, ternary secret)
MODELS = [("ADPS16 classical core-SVP", RC.ADPS16), ("MATZOV (estimator default)", RC.MATZOV)]


def estimate_bits(n: int, log2_q: int, cost_model) -> tuple[float, str]:
    params = LWE.Parameters(n=n, q=2 ** log2_q, Xs=ND.UniformMod(3), Xe=ND.DiscreteGaussian(SIGMA))
    res = LWE.estimate(params, red_cost_model=cost_model, deny_list=("arora-gb", "bkw"), quiet=True)
    best = min(((math.log2(float(r["rop"])), a) for a, r in res.items() if "rop" in r and float(r["rop"]) > 0))
    return round(best[0], 2), best[1]


def main() -> int:
    rows = []
    for n, lq in HES_ROWS:
        row = {"n": n, "log2_q": lq, "published_security_bits": 128}
        for name, cm in MODELS:
            t0 = time.time()
            bits, atk = estimate_bits(n, lq, cm)
            row[name] = {"estimated_bits": bits, "cheapest_attack": atk, "gap_to_published_128": round(bits - 128, 2), "seconds": round(time.time() - t0, 1)}
            print(f"n={n:>5d} log2q={lq:>3d}  {name:28s} -> {bits:7.2f} bits ({atk}), {bits - 128:+.1f} vs published 128", flush=True)
        rows.append(row)
    try:
        commit = subprocess.check_output(["git", "-C", EST_DIR, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"
    out = {"purpose": "does this estimator and cost model reproduce the HE Standard's own published 128-bit rows?",
           "estimator_commit": commit, "secret": "ternary uniform", "error": f"discrete Gaussian sigma={SIGMA}", "rows": rows}
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"written -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
