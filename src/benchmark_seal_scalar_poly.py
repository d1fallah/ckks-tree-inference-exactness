#!/usr/bin/env python3
"""
Added analysis S2.4a (docs/step2_prespec_m1.md, section D): the polynomial step on ONE ciphertext through the
two SEAL evaluators (original power ladder, coefficient-folding), with the frozen comparison polynomial of
degree k applied with the same scalar coefficients in every slot.

One ciphertext of 2,751 slots (the largest model's leaf count), locked parameters (N = 16,384,
chain [60, 40 x 7, 60]). Original and folding evaluations are interleaved repetition by repetition
(5 discarded warm-up repetitions, 20 timed). The original evaluator does not fit k = 9 on the locked chain, so
k = 9 is folding-only. Accuracy is the maximum absolute error of the decrypted result against the frozen
polynomial evaluated in plaintext. The companion benchmark_openfhe_ps.py runs the same polynomial through
OpenFHE's Paterson-Stockmeyer evaluation.

Output: results/m1_seal_scalar_poly_benchmark.json
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import tenseal as ts

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import ckks_circuit as cc  # noqa: E402
import ckks_instrumented as ci  # noqa: E402

OUT_PATH = os.path.join(_HERE, "..", "results", "m1_seal_scalar_poly_benchmark.json")
N_SLOTS, N_WARMUP, N_TIMED, SEED = 2751, 5, 20, 42


def main() -> int:
    art = cc.load_circuit_frozen_artifacts()
    a = art["a_half_width"]
    soft = art["poly_coeffs"]["targets"]["soft_step"]
    rng = np.random.default_rng(SEED)
    z = rng.uniform(-a, a, N_SLOTS)
    ctx = ci.make_ckks_context()
    results = {}
    for k in (3, 5, 7, 9):
        power = np.asarray(soft[f"k={k}"]["coefficients_power_basis"], dtype=float)
        coeff_mat = np.tile(power[:, None], (1, N_SLOTS))
        truth = np.polynomial.polynomial.polyval(z, power)
        evals = {"folding": cc.eval_poly_tenseal_folded}
        if k != 9:
            evals["original"] = cc.eval_poly_tenseal
        times = {name: [] for name in evals}
        err = {name: 0.0 for name in evals}
        for rep in range(N_WARMUP + N_TIMED):
            ct = ts.ckks_vector(ctx, z.tolist())
            order = list(evals) if rep % 2 == 0 else list(reversed(list(evals)))
            for name in order:
                t0 = time.perf_counter()
                res = evals[name](coeff_mat, ct)
                dt = time.perf_counter() - t0
                if rep >= N_WARMUP:
                    times[name].append(dt)
                if rep == N_WARMUP:
                    err[name] = float(np.max(np.abs(np.asarray(res.decrypt()) - truth)))
        results[f"k={k}"] = {
            name: {"mean_s": float(np.mean(v)), "sd_s": float(np.std(v, ddof=1)), "n_timed": len(v),
                   "max_abs_error_vs_frozen_polynomial": err[name]} for name, v in times.items()}
        print(f"k={k}: " + "  ".join(f"{n} {r['mean_s']:.3f}+/-{r['sd_s']:.3f}s err {r['max_abs_error_vs_frozen_polynomial']:.1e}"
                                      for n, r in results[f"k={k}"].items()), flush=True)
    out = {
        "protocol": "docs/step2_prespec_m1.md section D (added analysis S2.4a)",
        "backend": "REAL TenSEAL/SEAL, locked parameters", "n_slots": N_SLOTS, "n_warmup_discarded": N_WARMUP,
        "interleaving": "original and folding alternate within every repetition (order swapped on odd repetitions)",
        "input": f"{N_SLOTS} uniform values on [-a, a], a = {a}, seed {SEED}; same scalar coefficients in every slot",
        "by_k": results,
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"written -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
