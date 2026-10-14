#!/usr/bin/env python3
"""
Added analysis S2.4b (docs/step2_prespec_m1.md, section D): the same comparison polynomial as
benchmark_seal_scalar_poly.py evaluated with OpenFHE's Paterson-Stockmeyer Chebyshev evaluation
(EvalChebyshevFunction), on one ciphertext of 2,751 slots.

Runs in the Python 3.10 venv that holds the `openfhe` wheel (the wheel ships a cp310 build only):
    ~/.venvs/m1-openfhe310/bin/python src/benchmark_openfhe_ps.py

Context: ring degree 16,384, 128-bit classical security, scaling modulus 40 bits, first modulus 60 bits,
multiplicative depth 7 (the depth of the locked SEAL chain). OpenFHE and SEAL differ in RNS layout,
rescaling and key switching, so this is a library-level comparison under a matched ring degree and security
target, not a swap of one evaluator inside SEAL.

Two variants per degree: (A) the scalar polynomial alone; (B) the scalar polynomial followed by the per-slot
affine step a circuit with per-slot coefficients needs (out = A + B*S(z) + C*z with plaintext vectors A, B, C,
which costs one more level). The circuit's per-slot coefficient vectors cannot be passed to a scalar-coefficient
Chebyshev evaluator, which is why (B) exists. Accuracy is the maximum absolute error against the frozen
polynomial (power basis) evaluated in plaintext, which also checks that OpenFHE's interpolant is the same polynomial.

Output: results/m1_openfhe_ps_benchmark.json
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
from openfhe import (CCParamsCKKSRNS, GenCryptoContext, KeySwitchTechnique, PKESchemeFeature, ScalingTechnique, SecurityLevel)

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(_HERE, "..", "results", "m1_openfhe_ps_benchmark.json")
COEFFS_PATH = os.path.join(_HERE, "..", "data", "poly_coeffs_v1.json")
N_SLOTS, N_WARMUP, N_TIMED, SEED = 2751, 5, 20, 42
RING_DIM, DEPTH, SCALE_BITS, FIRST_BITS = 16384, 7, 40, 60


def main() -> int:
    with open(COEFFS_PATH, "r", encoding="utf-8") as f:
        frozen = json.load(f)
    a, gamma = frozen["interval"][1], frozen["gamma"]
    rng = np.random.default_rng(SEED)
    z = rng.uniform(-a, a, N_SLOTS)
    pat = rng.integers(0, 3, N_SLOTS)  # per-slot affine pattern, values irrelevant to timing
    A_vec = np.where(pat == 0, 1.0, 0.0)
    B_vec = np.where(pat == 1, 1.0, -1.0)
    C_vec = np.where(pat == 2, -1.0, 0.0)

    params = CCParamsCKKSRNS()
    params.SetSecurityLevel(SecurityLevel.HEStd_128_classic)
    params.SetRingDim(RING_DIM)
    params.SetMultiplicativeDepth(DEPTH)
    params.SetScalingModSize(SCALE_BITS)
    params.SetFirstModSize(FIRST_BITS)
    params.SetScalingTechnique(ScalingTechnique.FLEXIBLEAUTO)
    params.SetBatchSize(8192)
    # One RNS digit per tower (one special modulus), the structure of the locked SEAL chain. OpenFHE's default digit
    # count adds too many special primes for ring degree 16,384 at 128-bit security (the library then asks for 32,768).
    params.SetKeySwitchTechnique(KeySwitchTechnique.HYBRID)
    params.SetNumLargeDigits(DEPTH + 1)
    cc = GenCryptoContext(params)
    for feat in (PKESchemeFeature.PKE, PKESchemeFeature.KEYSWITCH, PKESchemeFeature.LEVELEDSHE, PKESchemeFeature.ADVANCEDSHE):
        cc.Enable(feat)
    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    pt_A, pt_B, pt_C = (cc.MakeCKKSPackedPlaintext(v.tolist()) for v in (A_vec, B_vec, C_vec))

    def soft_step(x):
        return 1.0 / (1.0 + np.exp(gamma * x))

    def decrypt(ct):
        pt = cc.Decrypt(ct, keys.secretKey)
        pt.SetLength(N_SLOTS)
        return np.asarray(pt.GetRealPackedValue())

    results = {}
    for k in (3, 5, 7, 9):
        power = np.asarray(frozen["targets"]["soft_step"][f"k={k}"]["coefficients_power_basis"], dtype=float)
        truth_S = np.polynomial.polynomial.polyval(z, power)
        truth_B = A_vec + B_vec * truth_S + C_vec * z
        t_A, t_B = [], []
        info = {}
        for rep in range(N_WARMUP + N_TIMED):
            ct = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(z.tolist()))
            t0 = time.perf_counter()
            S = cc.EvalChebyshevFunction(soft_step, ct, -a, a, k)
            dt_A = time.perf_counter() - t0
            t0 = time.perf_counter()
            S2 = cc.EvalChebyshevFunction(soft_step, ct, -a, a, k)
            out = cc.EvalAdd(cc.EvalAdd(cc.EvalMult(S2, pt_B), cc.EvalMult(ct, pt_C)), pt_A)
            dt_B = time.perf_counter() - t0
            if rep >= N_WARMUP:
                t_A.append(dt_A)
                t_B.append(dt_B)
            if rep == N_WARMUP:
                info = {"levels_consumed_scalar": int(S.GetLevel() - ct.GetLevel()),
                        "levels_consumed_with_per_slot_affine": int(out.GetLevel() - ct.GetLevel()),
                        "max_abs_error_scalar_vs_frozen_polynomial": float(np.max(np.abs(decrypt(S) - truth_S))),
                        "max_abs_error_affine_vs_frozen_polynomial": float(np.max(np.abs(decrypt(out) - truth_B)))}
        results[f"k={k}"] = {
            "scalar_polynomial": {"mean_s": float(np.mean(t_A)), "sd_s": float(np.std(t_A, ddof=1)), "n_timed": len(t_A)},
            "with_per_slot_affine": {"mean_s": float(np.mean(t_B)), "sd_s": float(np.std(t_B, ddof=1)), "n_timed": len(t_B)},
            **info}
        r = results[f"k={k}"]
        print(f"k={k}: scalar {r['scalar_polynomial']['mean_s']:.3f}s (levels {info['levels_consumed_scalar']}, err "
              f"{info['max_abs_error_scalar_vs_frozen_polynomial']:.1e})  +affine {r['with_per_slot_affine']['mean_s']:.3f}s "
              f"(levels {info['levels_consumed_with_per_slot_affine']})", flush=True)
    import openfhe
    out_doc = {
        "protocol": "docs/step2_prespec_m1.md section D (added analysis S2.4b)",
        "backend": "REAL OpenFHE CKKS (library-level comparison; see module docstring)",
        "openfhe_version": getattr(openfhe, "__version__", "1.5.1.0.22.4 (pip wheel)"),
        "context": {"ring_dim": RING_DIM, "multiplicative_depth": DEPTH, "scaling_mod_bits": SCALE_BITS,
                    "first_mod_bits": FIRST_BITS, "security": "HEStd_128_classic", "scaling_technique": "FLEXIBLEAUTO",
                    "key_switching": "HYBRID, num_large_digits = depth + 1 = 8 (one special modulus)"},
        "n_slots": N_SLOTS, "n_warmup_discarded": N_WARMUP,
        "input": f"{N_SLOTS} uniform values on [-a, a], a = {a}, seed {SEED}",
        "by_k": results,
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out_doc, f, indent=2)
    print(f"written -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
