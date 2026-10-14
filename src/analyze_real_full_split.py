"""
M1 revision E2 -- aggregation and statistics for the full-test-split real-ciphertext run.
Reads only the unit files written by eval_ckks_real_full_split.py (no ciphertext is evaluated here).

Metrics (pre-specification, docs/revision_prespec_m1.md E2): real-ciphertext macro-F1 and per-label F1 against plaintext with the paired
bootstrap 95% CI of |F1_plain - F1_real| (2,000 resamples, the existing utility statistic, eval_metrics); sign agreement with the simulator
on every evaluation; maximum and normalized margin error. Criterion fixed in advance: real |dF1| <= 0.03. One seed (42), k = 3, original
evaluator.   Output: results/m1_real_full_split.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np
from sklearn.metrics import f1_score

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import eval_ckks_real_full_split as e2  # noqa: E402
import eval_metrics as em  # noqa: E402

OUT_PATH = os.path.join(_HERE, "..", "results", "m1_real_full_split.json")
PARTIAL_PATH = os.path.join(_HERE, "..", "results", "m1_real_full_split_PARTIAL.json")
N_BOOTSTRAP, BOOTSTRAP_SEED, TOLERANCE = 2000, 42, 0.03


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partial", action="store_true",
                    help="monitoring only: analyze the chunks that already exist for all five labels and write "
                         "results/m1_real_full_split_PARTIAL.json, which is never used by the manuscript")
    args = ap.parse_args()
    test_df, _, label_cols = e2.load_test()
    if args.partial:
        n_chunks_all = -(-len(test_df) // e2.UNIT_SIZE)
        done = [ch for ch in range(n_chunks_all) if all(os.path.exists(e2.unit_path(li, ch)) for li in range(len(label_cols)))]
        keep = np.concatenate([np.arange(ch * e2.UNIT_SIZE, min((ch + 1) * e2.UNIT_SIZE, len(test_df))) for ch in done])
        print(f"PARTIAL: {len(done)} of {n_chunks_all} chunks complete for all labels ({len(keep)} stays)")
        test_df = test_df.iloc[keep].reset_index(drop=True)
        chunk_ids = done
    else:
        chunk_ids = list(range(-(-len(test_df) // e2.UNIT_SIZE)))
    n_test = len(test_df)
    with open(e2.BASELINE_PATH, "r", encoding="utf-8") as f:
        baseline = json.load(f)
    seed_rec = next(r for r in baseline["per_seed_results"] if r["seed"] == e2.SEED)
    thr = {lab: pl["tuned_threshold_margin"] for lab, pl in zip(label_cols, seed_rec["per_label"])}

    real = np.full((n_test, len(label_cols)), np.nan)
    sim = np.zeros_like(real)
    plain = np.zeros_like(real)
    y = np.zeros_like(real, dtype=int)
    failures, seconds, h = [], 0.0, hashlib.sha256()
    for li, lab in enumerate(label_cols):
        pos = 0
        for ch in chunk_ids:
            path = e2.unit_path(li, ch)
            assert os.path.exists(path), f"missing unit {path}: the run is incomplete"
            z = np.load(path)
            lo = pos
            hi = pos + len(z["stay_id"])
            pos = hi
            assert np.array_equal(z["stay_id"], test_df["stay_id"].to_numpy()[lo:hi]), f"stay mismatch in {path}"
            real[lo:hi, li], sim[lo:hi, li], plain[lo:hi, li], y[lo:hi, li] = z["real"], z["sim"], z["plain"], z["y"]
            assert np.array_equal(y[lo:hi, li], test_df[lab].to_numpy(dtype=int)[lo:hi]), f"label mismatch in {path}"
            failures += json.loads(str(z["failures"][0]))
            seconds += float(z["seconds"][0])
            h.update(z["real"].tobytes())

    ok = ~np.isnan(real)  # a failed evaluation is excluded and counted, never replaced
    n_eval, n_fail = int(ok.sum()), int((~ok).sum())
    keep_rows = ok.all(axis=1)  # macro-F1 needs the five labels of a stay together
    thr_vec = np.array([thr[l] for l in label_cols])
    pred = lambda m: (m >= thr_vec[None, :]).astype(int)
    P, S, R = pred(plain), pred(sim), pred(np.where(ok, real, 0.0))
    Y = y
    yk, Pk, Sk, Rk = Y[keep_rows], P[keep_rows], S[keep_rows], R[keep_rows]
    f1_plain, f1_sim, f1_real = em.macro_f1(yk, Pk), em.macro_f1(yk, Sk), em.macro_f1(yk, Rk)
    boot_real = em.paired_bootstrap_delta_f1_ci(yk, Pk, Rk, n_resamples=N_BOOTSTRAP, seed=BOOTSTRAP_SEED)
    boot_sim = em.paired_bootstrap_delta_f1_ci(yk, Pk, Sk, n_resamples=N_BOOTSTRAP, seed=BOOTSTRAP_SEED)

    per_label = {}
    for li, lab in enumerate(label_cols):
        m = ok[:, li]
        err = np.abs(real[m, li] - sim[m, li])
        span = float(sim[m, li].max() - sim[m, li].min()) or 1.0
        per_label[lab] = {
            "n_evaluations": int(m.sum()), "n_positive": int(Y[m, li].sum()),
            "f1_plaintext": float(f1_score(Y[m, li], P[m, li], zero_division=0)),
            "f1_simulated": float(f1_score(Y[m, li], S[m, li], zero_division=0)),
            "f1_real": float(f1_score(Y[m, li], R[m, li], zero_division=0)),
            "sign_agreement_real_vs_simulated": float(np.mean(np.sign(real[m, li]) == np.sign(sim[m, li]))),
            "n_sign_disagree_real_vs_simulated": int(np.sum(np.sign(real[m, li]) != np.sign(sim[m, li]))),
            "decision_agreement_real_vs_simulated_at_threshold": float(np.mean(R[m, li] == S[m, li])),
            "decision_agreement_real_vs_plaintext_at_threshold": float(np.mean(R[m, li] == P[m, li])),
            "n_decision_disagree_real_vs_simulated": int(np.sum(R[m, li] != S[m, li])),
            "max_abs_margin_error_real_vs_simulated": float(err.max()),
            "normalized_mae_real_vs_simulated": float(err.mean() / span),
            "mean_abs_margin_error_real_vs_simulated": float(err.mean()),
            "frozen_threshold_margin": thr[lab],
        }

    err_all = np.abs(real[ok] - sim[ok])
    out = {
        "protocol": "docs/revision_prespec_m1.md (E2)",
        "backend": "REAL TenSEAL/SEAL ciphertext execution (ckks_circuit.evaluate_circuit_tenseal, original evaluator), next to the SIMULATED numpy circuit and the plaintext model",
        "configuration": {"seed": e2.SEED, "k": e2.K, "polynomial_evaluator": "original", "labels": label_cols,
                          "n_test_stays": n_test, "unit_size": e2.UNIT_SIZE, "one_seed_only": True},
        "n_evaluations": n_eval, "n_evaluations_expected": n_test * len(label_cols), "n_failed": n_fail, "failures": failures[:50],
        "n_stays_used_for_macro_f1": int(keep_rows.sum()),
        "macro_f1": {"plaintext": f1_plain, "simulated": f1_sim, "real": f1_real},
        "delta_f1_real_vs_plaintext": {**boot_real, "n_resamples": N_BOOTSTRAP, "bootstrap_seed": BOOTSTRAP_SEED},
        "delta_f1_simulated_vs_plaintext": {**boot_sim, "n_resamples": N_BOOTSTRAP, "bootstrap_seed": BOOTSTRAP_SEED},
        "tolerance": TOLERANCE, "criterion_real_delta_f1_within_tolerance": bool(boot_real["point_estimate"] <= TOLERANCE),
        "agreement_all_evaluations": {
            "n_sign_agree_real_vs_simulated": int(np.sum(np.sign(real[ok]) == np.sign(sim[ok]))),
            "sign_agreement_real_vs_simulated": float(np.mean(np.sign(real[ok]) == np.sign(sim[ok]))),
            "n_decision_disagree_real_vs_simulated_at_threshold": int(np.sum(R[ok] != S[ok])),
            "max_abs_margin_error_real_vs_simulated": float(err_all.max()),
            "mean_abs_margin_error_real_vs_simulated": float(err_all.mean())},
        "per_label": per_label,
        "total_encrypt_evaluate_decrypt_cpu_seconds_across_workers": round(seconds, 1),
        "sha256_of_real_margins": h.hexdigest(),
    }
    if args.partial:
        out["PARTIAL_monitoring_only"] = True
    with open(PARTIAL_PATH if args.partial else OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: out[k] for k in ("n_evaluations", "n_failed", "macro_f1", "delta_f1_real_vs_plaintext",
                                             "delta_f1_simulated_vs_plaintext", "criterion_real_delta_f1_within_tolerance",
                                             "agreement_all_evaluations")}, indent=2))
    print(f"written -> {PARTIAL_PATH if args.partial else OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
