"""
M1 -- Record the ACTUAL coefficient-modulus chain SEAL builds for the locked parameters (input to the independent LWE estimate,
closure guide 4.2). Runs in the m1 venv (needs TenSEAL); the estimator itself runs in a separate venv and reads this file.

Nothing is changed: this reproduces SEAL's own CoeffModulus.Create(N, bit_sizes) for the locked N and chain and records every
prime, its bit length, the total, the scale, and the SEAL / TenSEAL versions.

Output: results/m1_ckks_modulus_chain.json
"""
import json
import os
import sys

import tenseal as ts
from tenseal import sealapi

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from ckks_instrumented import LOCKED_CHAIN, LOCKED_N, LOCKED_SCALE_BITS  # noqa: E402

OUT_PATH = os.path.join(_HERE, "..", "results", "m1_ckks_modulus_chain.json")


def main() -> int:
    primes = [int(m.value()) for m in sealapi.CoeffModulus.Create(LOCKED_N, LOCKED_CHAIN)]
    prod = 1
    for p in primes:
        prod *= p
    out = {
        "N_poly_modulus_degree": LOCKED_N,
        "requested_bit_sizes": LOCKED_CHAIN,
        "scale_bits": LOCKED_SCALE_BITS,
        "primes": primes,
        "prime_bit_lengths": [p.bit_length() for p in primes],
        "total_modulus_bits_sum_of_bit_lengths": sum(p.bit_length() for p in primes),
        "log2_of_product_of_primes": round(__import__("math").log2(prod), 6),
        "note": ("Key-generation security is with respect to the FULL modulus including the special (last) prime; "
                 "ciphertexts at deeper levels use a prefix of the chain but are encrypted under the same secret and "
                 "public key, so the full product is the relevant q for the security estimate."),
        "tenseal_version": getattr(ts, "__version__", "unknown"),
        "seal_reference": "Microsoft SEAL as bundled with TenSEAL (version string not exposed by tenseal.sealapi)",
        "seal_secret_key_distribution": "ternary {-1, 0, 1}, uniform (SEAL documented default; not read back from the binary)",
        "seal_error_distribution": "centered discrete Gaussian, standard deviation 3.20 (SEAL default), truncated at 6 sigma (SEAL documented default; not read back from the binary)",
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"{len(primes)} primes, total {out['total_modulus_bits_sum_of_bit_lengths']} bits, log2(product)={out['log2_of_product_of_primes']}")
    print(f"written -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
