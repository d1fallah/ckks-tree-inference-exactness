#!/usr/bin/env python3
"""
Verify the locked CKKS parameter set (N, coefficient-modulus chain) against the
Homomorphic Encryption Security Standard tables, per Gate M1-0 Sec 5's open
checklist item ("security level lambda, to be confirmed against the LWE
estimator before the final benchmark").

This does not invoke the standalone lattice-estimator Sage script by Albrecht,
Player and Scott directly. It uses Microsoft SEAL's own bundled implementation
of the resulting Homomorphic Encryption Security Standard tables (the same
standard those estimator runs produced), via TenSEAL's `sealapi` submodule,
which exposes SEAL's C++ API 1:1 in Python:

  - `CoeffModulus.MaxBitCount(N, sec_level)` returns the published ceiling.
  - Constructing a `SEALContext` with security-level enforcement turned on
    and checking `parameters_set()` is SEAL's own authoritative accept/reject
    decision for whether a parameter set meets that security level -- this is
    the same check SEAL performs internally before allowing key generation.

Run inside the WSL m1 venv (TenSEAL is not installed for Windows Python):
    wsl -e ~/.venvs/m1/bin/python src/verify_ckks_security_level.py

Output: results/m1_security_level_verification.json
"""
from __future__ import annotations

import json
import os

import tenseal.sealapi as sealapi

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(_HERE, "..", "results", "m1_security_level_verification.json")

# Locked in m1-0-implementation-lock.md Sec 5. Sized once for the deepest
# circuit (k=9, depth 7) and held fixed across the whole k-sweep.
N_POLY_MODULUS_DEGREE = 16384
COEFF_MOD_BIT_SIZES = [60, 40, 40, 40, 40, 40, 40, 40, 60]


def main() -> int:
    coeff_mod = sealapi.CoeffModulus.Create(N_POLY_MODULUS_DEGREE, COEFF_MOD_BIT_SIZES)
    actual_bits = sum(m.bit_count() for m in coeff_mod)

    parms = sealapi.EncryptionParameters(sealapi.SCHEME_TYPE.CKKS)
    parms.set_poly_modulus_degree(N_POLY_MODULUS_DEGREE)
    parms.set_coeff_modulus(coeff_mod)

    results = {}
    for name, level in (("TC128", sealapi.SEC_LEVEL_TYPE.TC128), ("TC192", sealapi.SEC_LEVEL_TYPE.TC192)):
        ceiling = sealapi.CoeffModulus.MaxBitCount(N_POLY_MODULUS_DEGREE, level)
        ctx = sealapi.SEALContext(parms, True, level)
        results[name] = {
            "standard_max_bits_for_N": ceiling,
            "parameters_set_under_enforcement": bool(ctx.parameters_set()),
        }

    out = {
        "N_poly_modulus_degree": N_POLY_MODULUS_DEGREE,
        "coeff_modulus_bit_sizes": COEFF_MOD_BIT_SIZES,
        "actual_total_bits": actual_bits,
        "method": (
            "tenseal.sealapi (Microsoft SEAL C++ API bindings): "
            "CoeffModulus.MaxBitCount() against the Homomorphic Encryption "
            "Security Standard tables, cross-checked by constructing a "
            "SEALContext with security-level enforcement and reading "
            "parameters_set()."
        ),
        "by_security_level": results,
        "conclusion": (
            f"N={N_POLY_MODULUS_DEGREE}, {actual_bits}-bit chain satisfies "
            f"TC128 (ceiling {results['TC128']['standard_max_bits_for_N']} bits) "
            "but not TC192 -- lambda=128 is confirmed, not merely asserted."
            if results["TC128"]["parameters_set_under_enforcement"]
            else "FAILED TC128 -- the locked N/chain does not meet 128-bit security. Do not sign Gate M1-0 as-is."
        ),
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(json.dumps(out, indent=2))
    print(f"\nwritten -> {OUT}")
    return 0 if results["TC128"]["parameters_set_under_enforcement"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
