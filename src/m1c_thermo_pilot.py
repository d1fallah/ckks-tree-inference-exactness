#!/usr/bin/env python3
"""
Validity pilot for the model-private thermometer protocol. Executes docs/m1c_thermo_pilot_prespec.md (committed before this file existed).
Validation split only, seed 42; the test split is never loaded. Refuses to run unless the Question gate "M1 model-private thermometer protocol"
in GATES.md carries an "approved by" line. Aggregate output only: results/m1c_thermo_pilot.json.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import tenseal as ts
import xgboost as xgb
from sklearn.metrics import roc_auc_score

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import ckks_circuit as cc  # noqa: E402
import m1c_final_common as fc  # noqa: E402
import m1c_thermo as th  # noqa: E402
from m1c_final_run import build_exact_circuit  # noqa: E402
from test_m1c_thermo import check_every_split  # noqa: E402

OUT = os.path.join(fc.RESULTS, "m1c_thermo_pilot.json")
GATE = 5e-4
LEVELS_B = (5, 8, 16)
N_REAL = 100
CHAIN4 = [60, 40, 40, 40, 40, 60]
CHAIN7 = [60, 40, 40, 40, 40, 40, 40, 40, 60]
TITLE = "Question gate, M1 model-private thermometer protocol"


def make_ctx(n, chain, scale_bits):
    ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=n, coeff_mod_bit_sizes=chain)
    ctx.global_scale = 2**scale_bits
    ctx.generate_galois_keys()
    return ctx


def summarize(errs, agree, secs, upload, n, gaps=None):
    r = {"n_stays": n, "max_abs_margin_error": float(max(errs)), "mean_abs_margin_error": float(np.mean(errs)),
         "decisions_agreeing": int(agree), "gate": GATE, "all_within_gate": bool(max(errs) <= GATE),
         "passes_circuit_gate": bool(max(errs) <= GATE and agree == n),
         "mean_seconds_per_stay": float(np.mean(secs)), "sd_seconds_per_stay": float(np.std(secs, ddof=1)),
         "client_upload_bytes": int(upload)}
    if gaps is not None:
        r["min_abs_gap_to_f1_threshold_sampled"] = float(np.min(gaps))
        r["error_below_min_gap"] = bool(max(errs) < np.min(gaps))
    return r


def key_bytes(ctx) -> dict:
    """One-off key material, serialized with TenSEAL defaults (no secret key). Galois keys are the default power-of-two set."""
    full = len(ctx.serialize(save_public_key=True, save_secret_key=False, save_galois_keys=True, save_relin_keys=True))
    no_gal = len(ctx.serialize(save_public_key=True, save_secret_key=False, save_galois_keys=False, save_relin_keys=True))
    no_relin = len(ctx.serialize(save_public_key=True, save_secret_key=False, save_galois_keys=False, save_relin_keys=False))
    return {"public_key_and_params": no_relin, "relin_keys": no_gal - no_relin, "galois_keys": full - no_gal, "total": full}


def main() -> int:
    t_start = time.perf_counter()
    fc.require_signed_entry(TITLE)
    with open(fc.DECISION_PATH, "r", encoding="utf-8") as f:
        use_acuity = bool(json.load(f)["use_acuity"])
    fr = fc.load_frames(use_acuity, need_test=False)
    assert fr["te"] is None
    feats, yva, ytr = fr["feats"], fr["yva"], fr["ytr"]
    clf_fp = fc.fit_xgb(fr["Xtr"], ytr, 42)
    m_fp = clf_fp.get_booster().predict(xgb.DMatrix(fr["Xva"], feature_names=feats), output_margin=True)
    auc_fp = float(roc_auc_score(yva, m_fp))
    ops_fp = fc.operating_points_midpoint(yva, m_fp)
    f1_fp = fc.decision_metrics(yva, m_fp, ops_fp["f1"])["f1"]
    print(f"full precision validation AUROC {auc_fp:.4f}, F1 {f1_fp:.3f}", flush=True)

    ctx4 = make_ctx(16384, CHAIN4, 40)
    ctx7 = make_ctx(16384, CHAIN7, 40)
    keys = {"n16384_four_level": key_bytes(ctx4), "n16384_locked_seven_level": key_bytes(ctx7)}
    print("key bytes", keys, flush=True)
    results = {}
    for B in LEVELS_B:
        t_b = time.perf_counter()
        edges, binner = fc.make_binner(fr["Xtr"], feats, B)
        Xtr_b, Xva_b = binner(fr["Xtr"]), binner(fr["Xva"])
        clf = fc.fit_xgb(Xtr_b, ytr, 42)
        bst = clf.get_booster()
        m_va = bst.predict(xgb.DMatrix(Xva_b, feature_names=feats), output_margin=True)
        auc = float(roc_auc_score(yva, m_va))
        ops = fc.operating_points_midpoint(yva, m_va)
        R = {"B": B, "auroc_val": auc, "auroc_cost_vs_full_precision": auc_fp - auc,
             "f1_val_quantized": fc.decision_metrics(yva, m_va, ops["f1"])["f1"], "f1_val_full_precision": f1_fp,
             "operating_points_margin": ops}
        R["n_splits_enumerated_ok"] = check_every_split(bst, B)
        R["val_gap_to_threshold"] = {name: {"min_abs_gap": float(np.min(np.abs(m_va - ops[name]))),
                                            "n_stays_within_gate": int((np.abs(m_va - ops[name]) <= GATE).sum())} for name in ("f1", "sens80")}
        structure = cc.extract_ensemble_leaf_circuit(bst, set(feats), tree_depth=6)
        levels = th.level_arrays(structure, feats, B)
        Xb = Xva_b.to_numpy(dtype=np.float64)
        bits_all = th.thermometer_bits(Xb, B)
        sim = th.simulate_margin(levels, structure.leaf_value, bits_all, B)
        R["simulation"] = {"max_abs_margin_difference_vs_booster": float(np.max(np.abs(sim - m_va)))}
        for name in ("f1", "sens80"):
            R["simulation"][f"n_differing_{name}"] = int(((sim >= ops[name]) != (m_va >= ops[name])).sum())
        print(f"B={B}: AUROC {auc:.4f} (cost {auc_fp - auc:.4f}); {R['n_splits_enumerated_ok']} splits OK; simulated max diff "
              f"{R['simulation']['max_abs_margin_difference_vs_booster']:.1e}; differing F1 {R['simulation']['n_differing_f1']}, "
              f"80%-sens {R['simulation']['n_differing_sens80']}", flush=True)

        rng = np.random.default_rng(42)
        pp = (m_va >= ops["f1"]).astype(int)
        idx = np.concatenate([rng.choice(np.flatnonzero(pp == 1), N_REAL // 2, replace=False),
                              rng.choice(np.flatnonzero(pp == 0), N_REAL // 2, replace=False)])
        thr = ops["f1"]
        gap_idx = np.abs(m_va[idx] - thr)

        def agree_count(reals):
            return sum(int((r >= thr) == (m_va[i] >= thr)) for r, i in zip(reals, idx))

        # protocol 1: broadcast thermometer, 4-level chain
        pv = th.folded_plain_vectors(levels, structure.leaf_value, B)
        errs, secs, reals = [], [], []
        upload = None
        for i in idx:
            t0 = time.perf_counter()
            cts = th.encrypt_bits(ctx4, bits_all[i])
            real, tm = th.eval_thermo_tenseal(pv, cts)
            secs.append(time.perf_counter() - t0)
            errs.append(abs(real - m_va[i]))
            reals.append(real)
            if upload is None:
                upload = th.serialized_bytes(cts)
        R["broadcast_thermometer_4level"] = summarize(errs, agree_count(reals), secs, upload, len(idx), gaps=gap_idx)
        # protocol 2: packed thermometer, 4-level chain
        mats = th.build_packed(levels, structure.leaf_value, B, len(feats))
        errs, secs, reals = [], [], []
        for i in idx:
            t0 = time.perf_counter()
            real, tm = th.eval_thermo_packed_tenseal(ctx4, mats, bits_all[i])
            secs.append(time.perf_counter() - t0)
            errs.append(abs(real - m_va[i]))
            reals.append(real)
        up = len(ts.ckks_vector(ctx4, [float(b) for b in bits_all[idx[0]]]).serialize())
        R["packed_thermometer_4level"] = summarize(errs, agree_count(reals), secs, up, len(idx), gaps=gap_idx)
        # protocol 4: client-computed indicators (client knows the model), 4-level chain
        ind = th.client_indicators(levels, bits_all[idx], B)
        errs, secs, reals = [], [], []
        for row, i in zip(ind, idx):
            t0 = time.perf_counter()
            real, tm = th.eval_client_indicator_tenseal(ctx4, row, structure.leaf_value)
            secs.append(time.perf_counter() - t0)
            errs.append(abs(real - m_va[i]))
            reals.append(real)
        vec = np.zeros(th.N_SLOTS)
        vec[: ind.shape[1]] = ind[0]
        R["client_indicator_baseline_4level"] = summarize(errs, agree_count(reals), secs,
                                                          len(ts.ckks_vector(ctx4, vec.tolist()).serialize()), len(idx), gaps=gap_idx)
        if B == 5:
            # protocol 3: exact interpolation, client knows thresholds, locked chain (degree 9, folding evaluator)
            so, ss, norm, coeffs, k = build_exact_circuit(bst, feats, B)
            errs, secs, reals = [], [], []
            for i in idx:
                t0 = time.perf_counter()
                real, tm = cc.evaluate_circuit_tenseal(ctx7, ss, coeffs, Xva_b.iloc[i], norm, 1.0, poly_evaluator=cc.eval_poly_tenseal_folded)
                secs.append(time.perf_counter() - t0)
                errs.append(abs(real - m_va[i]))
                reals.append(real)
            one = len(ts.ckks_vector(ctx7, [0.0] * structure.leaf_value.shape[0]).serialize())
            R["exact_interpolation_7level"] = summarize(errs, agree_count(reals), secs, 6 * one, len(idx), gaps=gap_idx)
            R["exact_interpolation_7level"]["degree"] = k
            # optional small-ring arm, 20 stays
            try:
                ctx8 = make_ctx(8192, [40, 30, 30, 30, 30, 40], 30)
                keys["n8192"] = key_bytes(ctx8)
                pv8 = th.folded_plain_vectors(levels, structure.leaf_value, B, n_slots=4096)
                errs, secs, reals = [], [], []
                for i in idx[:20]:
                    t0 = time.perf_counter()
                    cts = th.encrypt_bits(ctx8, bits_all[i], n_slots=4096)
                    real, tm = th.eval_thermo_tenseal(pv8, cts, n_slots=4096)
                    secs.append(time.perf_counter() - t0)
                    errs.append(abs(real - m_va[i]))
                    reals.append(real)
                R["broadcast_thermometer_ring8192"] = summarize(errs, sum(int((r >= thr) == (m_va[i] >= thr)) for r, i in zip(reals, idx[:20])),
                                                               secs, th.serialized_bytes(cts), 20, gaps=gap_idx[:20])
            except Exception as exc:
                R["broadcast_thermometer_ring8192"] = {"error": str(exc)[:300]}
        for key in ("broadcast_thermometer_4level", "packed_thermometer_4level", "client_indicator_baseline_4level",
                    "exact_interpolation_7level", "broadcast_thermometer_ring8192"):
            if key in R and "max_abs_margin_error" in R[key]:
                r = R[key]
                print(f"B={B} {key}: max err {r['max_abs_margin_error']:.2e}, agree {r['decisions_agreeing']}/{r['n_stays']}, "
                      f"{r['mean_seconds_per_stay']:.2f} s/stay, upload {r['client_upload_bytes'] / 1e6:.2f} MB", flush=True)
        R["runtime_seconds"] = time.perf_counter() - t_b
        results[f"B={B}"] = R
        with open(OUT.replace(".json", "_partial.json"), "w", encoding="utf-8") as f:
            json.dump(results, f)

    out = {"protocol": "docs/m1c_thermo_pilot_prespec.md", "scope": "seed 42, validation split only, test split not read",
           "use_acuity": use_acuity, "auroc_full_precision_val": auc_fp, "f1_full_precision_val": f1_fp,
           "chains": {"four_level": CHAIN4, "locked_seven_level": CHAIN7},
           "seeded_serialization_check": "TenSEAL symmetric encryption gave 1,053,170 bytes against 1,052,708 for public-key encryption on synthetic data: no reduction",
           "key_bytes_one_off": keys, "reference_points_prespec_addendum": "docs/m1c_thermo_pilot_prespec.md, Addendum 2026-10-12", "results": results, "runtime_seconds": time.perf_counter() - t_start}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"written -> {OUT} ({out['runtime_seconds']:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
