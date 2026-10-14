"""
M1 -- Instrumented real-CKKS evaluation (closure guide 4.1): a finer timing and size breakdown of the SAME circuit that
ckks_circuit.evaluate_circuit_tenseal evaluates. Nothing in ckks_circuit.py is modified.

Two kinds of measurement, kept separate so neither contaminates the other:

  1. THE ONLINE PATH (what the latency claim rests on). The real TenSEAL calls, timed with time.perf_counter():
       client_preprocess   z computation and clamping, in the clear   (compute_level_inputs)
       encode_encrypt      ts.ckks_vector(ctx, z) for each of the 6 level inputs -- TenSEAL fuses encode and encrypt
       poly_eval           eval_poly_tenseal for each level
       path_product        balanced product of the 6 branch ciphertexts
       sum                 output_ct.sum() (rotations and additions, homomorphic)
       decrypt_decode      total.decrypt()[0] -- TenSEAL fuses decrypt and decode
       wall_clock_total    perf_counter around the whole path (NOT the sum of buckets, so untimed glue is included)
  2. THE DECOMPOSITION (separates the two fused steps). After the online path has finished, the same inputs and the same
     output ciphertext are pushed through SEAL directly (tenseal.sealapi CKKSEncoder / Encryptor / Decryptor) and each
     step is timed alone: encode, encrypt, decrypt, decode. These replicate what TenSEAL does internally; they are NOT
     part of wall_clock_total, and the difference between (encode + encrypt) and the fused TenSEAL call is reported as the
     wrapper overhead, not hidden.

Ciphertext sizes come from ct.serialize(). Peak memory is measured by measure_ckks_latency.py in a separate process per k
(VmHWM from /proc/self/status), because a process-lifetime peak cannot be attributed to a single k otherwise.

The locked CKKS parameters live in exactly one place here (make_ckks_context) for all NEW scripts. Values are asserted
equal to notation.N_POLY_MODULUS_DEGREE; they are not changed.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import tenseal as ts
from tenseal import sealapi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ckks_circuit as cc  # noqa: E402
import depth_accounting as da  # noqa: E402
from notation import N_POLY_MODULUS_DEGREE  # noqa: E402

# Locked parameters (m1-0-implementation-lock.md; results/m1_security_level_verification.json). Not tunable here.
LOCKED_N = 16384
LOCKED_CHAIN = [60, 40, 40, 40, 40, 40, 40, 40, 60]
LOCKED_SCALE_BITS = 40
assert LOCKED_N == N_POLY_MODULUS_DEGREE, "locked N drifted from notation.py"


def make_ckks_context(with_galois_keys: bool = True):
    ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=LOCKED_N, coeff_mod_bit_sizes=LOCKED_CHAIN)
    ctx.global_scale = 2**LOCKED_SCALE_BITS
    if with_galois_keys:
        ctx.generate_galois_keys()
    return ctx


def make_seal_handles(ctx) -> dict:
    sc = ctx.seal_context().data
    return {"encoder": sealapi.CKKSEncoder(sc), "encryptor": sealapi.Encryptor(sc, ctx.public_key().data),
            "decryptor": sealapi.Decryptor(sc, ctx.secret_key().data), "seal_context": sc, "scale": 2.0**LOCKED_SCALE_BITS}


def evaluate_instrumented(ctx, seal, structure, coeff_vectors, x_row, norm_bounds, a_half_width, decompose: bool = True,
                          poly_evaluator=None):
    """Returns (margin, record). Same arithmetic as cc.evaluate_circuit_tenseal; see the module docstring."""
    rec: dict = {}
    t_start = time.perf_counter()

    t0 = time.perf_counter()
    z_levels = cc.compute_level_inputs(structure, x_row, norm_bounds, a_half_width)
    rec["client_preprocess_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    level_cts = [ts.ckks_vector(ctx, z.tolist()) for z in z_levels]
    rec["encode_encrypt_fused_s"] = time.perf_counter() - t0

    # identical leaf-value fold to ckks_circuit.evaluate_circuit_tenseal: plaintext-only, adds no ciphertext depth
    scaled = list(coeff_vectors)
    scaled[-1] = coeff_vectors[-1] * structure.leaf_value[None, :]

    t0 = time.perf_counter()
    evaluator = poly_evaluator or cc.eval_poly_tenseal  # default path unchanged; the folding evaluator is opt-in (revision E1)
    branch_cts = [evaluator(cm, ct) for cm, ct in zip(scaled, level_cts)]
    rec["poly_eval_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    output_ct = da.balanced_product(branch_cts)  # same recursion as the committed nested function (asserted in depth_accounting)
    rec["path_product_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    total = output_ct.sum()
    rec["sum_rotations_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    margin = float(total.decrypt()[0])
    rec["decrypt_decode_fused_s"] = time.perf_counter() - t0

    rec["wall_clock_total_s"] = time.perf_counter() - t_start
    rec["sum_of_buckets_s"] = sum(rec[k] for k in ("client_preprocess_s", "encode_encrypt_fused_s", "poly_eval_s",
                                                   "path_product_s", "sum_rotations_s", "decrypt_decode_fused_s"))

    # ---- sizes (serialization is outside the timed path) ----
    rec["input_ciphertext_bytes_each"] = [len(ct.serialize()) for ct in level_cts]
    rec["input_ciphertext_bytes_total"] = int(sum(rec["input_ciphertext_bytes_each"]))
    rec["output_ciphertext_bytes"] = len(total.serialize())

    if decompose:
        enc, encryptor, decryptor, sc, scale = (seal["encoder"], seal["encryptor"], seal["decryptor"],
                                                seal["seal_context"], seal["scale"])
        t_encode = t_encrypt = 0.0
        for z in z_levels:
            pl = sealapi.Plaintext()
            t0 = time.perf_counter(); enc.encode(z.tolist(), scale, pl); t_encode += time.perf_counter() - t0
            ct = sealapi.Ciphertext(sc)
            t0 = time.perf_counter(); encryptor.encrypt(pl, ct); t_encrypt += time.perf_counter() - t0
        pl_out = sealapi.Plaintext()
        t0 = time.perf_counter(); decryptor.decrypt(total.ciphertext()[0], pl_out); t_decrypt = time.perf_counter() - t0
        t0 = time.perf_counter(); enc.decode_double(pl_out); t_decode = time.perf_counter() - t0
        rec.update({
            "encode_s": t_encode, "encrypt_s": t_encrypt, "decrypt_s": t_decrypt, "decode_s": t_decode,
            "tenseal_wrapper_overhead_encode_encrypt_s": rec["encode_encrypt_fused_s"] - (t_encode + t_encrypt),
            "tenseal_wrapper_overhead_decrypt_decode_s": rec["decrypt_decode_fused_s"] - (t_decrypt + t_decode),
        })
    return margin, rec


def context_key_sizes(ctx) -> dict:
    """Client-side key material, serialized. The secret key is never counted as shipped."""
    with_gal = len(ctx.serialize(save_public_key=True, save_secret_key=False, save_galois_keys=True, save_relin_keys=True))
    no_gal = len(ctx.serialize(save_public_key=True, save_secret_key=False, save_galois_keys=False, save_relin_keys=True))
    return {"evaluation_context_with_galois_keys_bytes": with_gal,
            "evaluation_context_without_galois_keys_bytes": no_gal,
            "galois_keys_bytes": with_gal - no_gal}


def reset_peak_rss() -> bool:
    """Reset this process's peak-RSS counter (VmHWM) so a following read reflects only what happens after the reset.
    Uses /proc/self/clear_refs (value 5); returns False if the kernel does not support it."""
    try:
        with open("/proc/self/clear_refs", "w") as f:
            f.write("5")
        return True
    except OSError:
        return False


def proc_memory_kb() -> dict:
    """Current and peak resident set size of THIS process (Linux/WSL /proc)."""
    out = {}
    with open("/proc/self/status", "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith(("VmRSS", "VmHWM")):
                k, v = line.split(":")
                out[k] = int(v.split()[0])
    return {"rss_kb": out.get("VmRSS"), "peak_rss_kb": out.get("VmHWM")}
