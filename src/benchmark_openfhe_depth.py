#!/usr/bin/env python3
"""
Pilot 3 (docs/m1c_pilot3_prespec.md, committed before this file): levels consumed by OpenFHE's
scalar-coefficient Chebyshev evaluation (EvalChebyshevFunction) at k = 11, 13, 15, in the context of
benchmark_openfhe_ps.py (ring degree 16,384, 128-bit classical security, depth 7, 40-bit scale). Library-level
depth measurement on synthetic inputs, one repetition, no timing reported. Runs in the Python 3.10 venv:
    ~/.venvs/m1-openfhe310/bin/python src/benchmark_openfhe_depth.py
Output: results/m1c_openfhe_depth.json
"""
from __future__ import annotations

import json
import os

import numpy as np
from numpy.polynomial import chebyshev as C
from openfhe import (CCParamsCKKSRNS, GenCryptoContext, KeySwitchTechnique, PKESchemeFeature, ScalingTechnique, SecurityLevel)

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(_HERE, "..", "results", "m1c_openfhe_depth.json")
PILOT2 = os.path.join(_HERE, "..", "results", "m1c_pilot2.json")
N_SLOTS, SEED = 2751, 42
RING_DIM, DEPTH, SCALE_BITS, FIRST_BITS = 16384, 7, 40, 60


def main() -> int:
    with open(PILOT2, "r", encoding="utf-8") as f:
        gamma = json.load(f)["configs"]["per_feature|threshold_decision"]["gamma"]
    z = np.random.default_rng(SEED).uniform(-1.0, 1.0, N_SLOTS)

    params = CCParamsCKKSRNS()
    params.SetSecurityLevel(SecurityLevel.HEStd_128_classic)
    params.SetRingDim(RING_DIM)
    params.SetMultiplicativeDepth(DEPTH)
    params.SetScalingModSize(SCALE_BITS)
    params.SetFirstModSize(FIRST_BITS)
    params.SetScalingTechnique(ScalingTechnique.FLEXIBLEAUTO)
    params.SetBatchSize(8192)
    params.SetKeySwitchTechnique(KeySwitchTechnique.HYBRID)
    params.SetNumLargeDigits(DEPTH + 1)
    cc = GenCryptoContext(params)
    for feat in (PKESchemeFeature.PKE, PKESchemeFeature.KEYSWITCH, PKESchemeFeature.LEVELEDSHE, PKESchemeFeature.ADVANCEDSHE):
        cc.Enable(feat)
    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)

    def soft_step(x):
        return 1.0 / (1.0 + np.exp(gamma * x))

    results = {}
    for k in (9, 11, 13, 15):  # k = 9 repeats a stored measurement as a consistency check
        entry: dict = {}
        try:
            ct = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(z.tolist()))
            S = cc.EvalChebyshevFunction(soft_step, ct, -1.0, 1.0, k)
            pt = cc.Decrypt(S, keys.secretKey)
            pt.SetLength(N_SLOTS)
            got = np.asarray(pt.GetRealPackedValue())
            ref = C.Chebyshev.interpolate(soft_step, deg=k, domain=[-1.0, 1.0])(z)
            entry = {"ok": True, "levels_consumed_scalar": int(S.GetLevel() - ct.GetLevel()),
                     "max_abs_error_vs_numpy_interpolant": float(np.max(np.abs(got - ref)))}
        except Exception as exc:  # the library may reject a degree that exceeds the context's depth
            entry = {"ok": False, "error": str(exc)[:300]}
        results[f"k={k}"] = entry
        print(f"k={k}: {entry}", flush=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"protocol": "docs/m1c_pilot3_prespec.md", "backend": "REAL OpenFHE CKKS (library-level depth)",
                   "context": {"ring_dim": RING_DIM, "multiplicative_depth": DEPTH, "scaling_mod_bits": SCALE_BITS,
                               "first_mod_bits": FIRST_BITS, "security": "HEStd_128_classic",
                               "scaling_technique": "FLEXIBLEAUTO", "key_switching": "HYBRID, num_large_digits = 8"},
                   "gamma_used": gamma, "interval": [-1.0, 1.0], "n_slots": N_SLOTS, "by_k": results}, f, indent=2)
    print(f"written -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
