#!/usr/bin/env python3
"""
Added analysis S2.3b (docs/step2_prespec_m1.md, section C): the total per-patient cost of a FIVE-label
prediction through the client/server split of track_b/m1_runtime.py (k = 3, original evaluator).

For each of the 25 frozen seed-42 patients and each of the five labels the client encrypts the level
ciphertexts, the server evaluates, and the client decrypts. Reported (aggregate only): encrypt, server and
decrypt time per label and summed over the five labels, bytes uploaded per patient (per label and summed),
bytes returned, and the serialized public/secret context sizes. The first 5 patients (20-24 of the sample, as
in the earlier protocol) are discarded warm-up; patients 0-19 are timed. A correctness gate checks the
decrypted margin against the circuit-faithful simulation.

Output: results/m1_five_label_cost.json
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "track_b"))
import m1_runtime as rt  # noqa: E402

OUT_PATH = os.path.join(_HERE, "..", "results", "m1_five_label_cost.json")
SAMPLE_PATH = os.path.join(_HERE, "..", "data", "m1_real_ckks_samples", "seed42.csv")
COHORT_PATH = os.path.join(_HERE, "..", "data", "m1_cohort.parquet")
FEATURES_PATH = os.path.join(_HERE, "..", "data", "m1_features.parquet")
N_WARMUP, N_TIMED = 5, 20
GATE_TOL = 5e-4


def main() -> int:
    t_start = time.perf_counter()
    circuit = rt.M1Circuit()
    client = rt.M1Client(circuit)
    server = rt.M1Server(circuit)
    labels = circuit.labels

    cohort = pd.read_parquet(COHORT_PATH)
    features = pd.read_parquet(FEATURES_PATH)
    feature_cols = circuit.feature_cols
    frame = features.merge(cohort[["stay_id", "subject_id"]], on="stay_id", how="inner").set_index("stay_id")
    sample = pd.read_csv(SAMPLE_PATH)
    X = frame.loc[sample["stay_id"].to_numpy(), feature_cols].reset_index(drop=True)
    import ckks_circuit as cc
    # reference for the correctness gate: the circuit-faithful simulation (the plaintext model differs from the circuit by the
    # polynomial approximation, by design)
    sim = {lab: np.array([cc.evaluate_circuit_numpy(circuit.structure[lab], circuit.coeffs[lab], X.iloc[i], circuit.norm_bounds,
                                                       circuit.a_half_width) for i in range(len(X))]) for lab in labels}

    warm = list(range(len(X) - N_WARMUP, len(X)))
    timed = list(range(N_TIMED))
    rec = {lab: {"enc": [], "srv": [], "dec": [], "upload_bytes": [], "return_bytes": [], "gate": []} for lab in labels}
    per_patient_total = []
    for i in warm + timed:
        total = 0.0
        for lab in labels:
            t0 = time.perf_counter()
            cts = client.encrypt(lab, X.iloc[i])
            t1 = time.perf_counter()
            out = server.evaluate(lab, cts)
            t2 = time.perf_counter()
            margin = client.decrypt(out)
            t3 = time.perf_counter()
            total += t3 - t0
            if i in timed:
                r = rec[lab]
                r["enc"].append(t1 - t0); r["srv"].append(t2 - t1); r["dec"].append(t3 - t2)
                r["upload_bytes"].append(sum(len(c) for c in cts)); r["return_bytes"].append(len(out))
                r["gate"].append(abs(margin - float(sim[lab][i])))
        if i in timed:
            per_patient_total.append(total)
        print(f"  patient {i}: {'warm-up' if i in warm else 'timed'}", flush=True)

    def ms(v):
        return {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1))}

    per_label = {lab: {"encrypt_s": ms(r["enc"]), "server_s": ms(r["srv"]), "decrypt_s": ms(r["dec"]),
                       "upload_bytes_per_patient": int(np.mean(r["upload_bytes"])),
                       "return_bytes_per_patient": int(np.mean(r["return_bytes"])),
                       "max_abs_margin_error_vs_simulation": float(np.max(r["gate"]))} for lab, r in rec.items()}
    assert all(v["max_abs_margin_error_vs_simulation"] <= GATE_TOL for v in per_label.values()), "correctness gate failed"
    out = {
        "protocol": "docs/step2_prespec_m1.md section C (added analysis S2.3b)",
        "backend": "REAL TenSEAL/SEAL ciphertext execution via track_b/m1_runtime.py (client/server split)",
        "k": circuit.k, "evaluator": "original", "n_warmup_discarded": N_WARMUP, "n_timed_patients": N_TIMED,
        "labels": list(labels),
        "per_label": per_label,
        "five_label_total_per_patient": {
            "wall_clock_s": ms(per_patient_total),
            "server_s_sum_over_labels": float(sum(v["server_s"]["mean"] for v in per_label.values())),
            "upload_bytes_sum_over_labels": int(sum(v["upload_bytes_per_patient"] for v in per_label.values())),
            "return_bytes_sum_over_labels": int(sum(v["return_bytes_per_patient"] for v in per_label.values())),
        },
        "context_bytes": {
            "public_context_incl_evaluation_keys": os.path.getsize(os.path.join(rt.DEFAULT_MODELS_DIR, "ckks_public_context.bin")),
            "secret_context": os.path.getsize(os.path.join(rt.DEFAULT_KEYS_DIR, "ckks_secret_context.bin")),
        },
        "runtime_seconds": time.perf_counter() - t_start,
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    t = out["five_label_total_per_patient"]
    print(f"five labels: {t['wall_clock_s']['mean']:.2f} +/- {t['wall_clock_s']['sd']:.2f} s/patient, "
          f"{t['upload_bytes_sum_over_labels'] / 1e6:.1f} MB uploaded; written -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
