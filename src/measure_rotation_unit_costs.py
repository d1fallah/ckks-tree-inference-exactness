#!/usr/bin/env python3
"""
Unit costs for the design memo on a server-side feature-placement redesign (docs/m1_server_side_placement_memo.md).
Measured at the locked parameters (N = 16,384, chain [60, 40 x 7, 60]) on one 8,192-slot ciphertext, through the
SEAL API directly (TenSEAL's CKKSVector does not expose rotation):

  * one rotation by a power of two (one key switch) and one by a step that is not a power of two (SEAL decomposes it
    into several key switches with the default Galois key set);
  * ciphertext x plaintext-vector multiplication followed by rescaling, with the plaintext pre-encoded;
  * ciphertext + plaintext addition;
  * the number of Galois keys in the default set and the serialized size of the Galois keys, hence bytes per key.

Aggregate timings only (mean and SD over 30 repetitions after 5 warm-up). Output: results/m1_rotation_unit_costs.json
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import tenseal.sealapi as sealapi

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import ckks_instrumented as ci  # noqa: E402

OUT_PATH = os.path.join(_HERE, "..", "results", "m1_rotation_unit_costs.json")
N_WARMUP, N_TIMED = 5, 30


def time_op(fn):
    for _ in range(N_WARMUP):
        fn()
    t = []
    for _ in range(N_TIMED):
        t0 = time.perf_counter()
        fn()
        t.append(time.perf_counter() - t0)
    return {"mean_s": float(np.mean(t)), "sd_s": float(np.std(t, ddof=1)), "n_timed": N_TIMED}


def main() -> int:
    ctx = ci.make_ckks_context()
    h = ci.make_seal_handles(ctx)
    sc, encoder, encryptor, scale = h["seal_context"], h["encoder"], h["encryptor"], h["scale"]
    evaluator = sealapi.Evaluator(sc)
    gk = ctx.galois_keys().data
    rlk = ctx.relin_keys().data
    n = encoder.slot_count()
    rng = np.random.default_rng(42)
    pt = sealapi.Plaintext()
    encoder.encode(rng.uniform(-1, 1, n).tolist(), scale, pt)
    ct = sealapi.Ciphertext()
    encryptor.encrypt(pt, ct)
    pt_vec = sealapi.Plaintext()
    encoder.encode(rng.uniform(-1, 1, n).tolist(), scale, pt_vec)
    out_ct = sealapi.Ciphertext()

    def rotate(steps):
        return lambda: evaluator.rotate_vector(ct, steps, gk, out_ct)

    def mult_plain_rescale():
        tmp = sealapi.Ciphertext()
        evaluator.multiply_plain(ct, pt_vec, tmp)
        evaluator.rescale_to_next_inplace(tmp)

    def add_plain():
        tmp = sealapi.Ciphertext()
        evaluator.add_plain(ct, pt, tmp)

    out = {
        "protocol": "unit costs for the design memo; measured, not estimated; SEAL API through tenseal.sealapi",
        "slots": int(n),
        "rotate_by_1_power_of_two": time_op(rotate(1)),
        "rotate_by_16_power_of_two": time_op(rotate(16)),
        "rotate_by_15_non_power_of_two": time_op(rotate(15)),
        "rotate_by_37_non_power_of_two": time_op(rotate(37)),
        "multiply_plain_preencoded_and_rescale": time_op(mult_plain_rescale),
        "add_plain_preencoded": time_op(add_plain),
        "key_sizes_bytes": ci.context_key_sizes(ctx),
    }
    out["default_galois_key_count"] = int(gk.size())
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    for k, v in out.items():
        if isinstance(v, dict) and "mean_s" in v:
            print(f"{k}: {v['mean_s'] * 1000:.2f} +/- {v['sd_s'] * 1000:.2f} ms")
    print("key sizes:", out["key_sizes_bytes"], "default Galois key count:", out["default_galois_key_count"])
    print(f"written -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
