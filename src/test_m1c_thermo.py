#!/usr/bin/env python3
"""
Synthetic-data tests of src/m1c_thermo.py (no patient data). Checks, for each bin count B:
  1. every split of every booster: for every integer input 0..B-1 the thermometer formula equals the booster's own predicate x < t;
  2. the NumPy thermometer simulator reproduces the booster margins (1e-6) on random integer inputs;
  3. the real TenSEAL circuit on a 4-level chain reproduces the simulator (5e-4) on a few rows, and its serialized upload and latency are reported.
Run:  python src/test_m1c_thermo.py        (WSL, ~/.venvs/m1)
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
import xgboost as xgb

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import ckks_circuit as cc  # noqa: E402
import m1c_thermo as th  # noqa: E402

F = 9
FEATS = [f"f{i}" for i in range(F)]


def make_booster(B: int, seed: int, n: int = 30000):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, B, size=(n, F)).astype(float)
    z = (X[:, 0] - B / 2) * 0.6 + (X[:, 1] > B // 2) * 1.2 - (X[:, 2] - B / 2) * 0.4 + 0.3 * rng.standard_normal(n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(z - 1.5)))).astype(int)
    clf = xgb.XGBClassifier(n_estimators=50, max_depth=6, tree_method="hist", subsample=0.8, colsample_bytree=0.8,
                            base_score=0.5, random_state=seed, eval_metric="logloss")
    clf.fit(pd.DataFrame(X, columns=FEATS), y)
    return clf, X


def check_every_split(booster, B: int) -> int:
    df = booster.trees_to_dataframe()
    internal = df[df["Feature"] != "Leaf"]
    n = 0
    for feat, thr in zip(internal["Feature"], internal["Split"]):
        t = float(thr)
        c = int(np.ceil(t - 1e-9))
        for x in range(B):
            yes_true = x < t
            al, be, cu = th.branch_coefficients(True, c, B)
            bit = (x >= cu) if be != 0 else 0
            yes_formula = al + be * bit
            no_formula = (lambda a, b, u: a + b * ((x >= u) if b != 0 else 0))(*th.branch_coefficients(False, c, B))
            assert yes_formula == int(yes_true), (feat, t, x, "yes")
            assert no_formula == int(not yes_true), (feat, t, x, "no")
        n += 1
    return n


def main() -> int:
    import tenseal as ts
    for B in (5, 8, 16):
        clf, X = make_booster(B, seed=42)
        bst = clf.get_booster()
        n_checked = check_every_split(bst, B)
        structure = cc.extract_ensemble_leaf_circuit(bst, set(FEATS), tree_depth=6)
        levels = th.level_arrays(structure, FEATS, B)
        rng = np.random.default_rng(7)
        Xt = rng.integers(0, B, size=(2000, F)).astype(float)
        plain = bst.predict(xgb.DMatrix(pd.DataFrame(Xt, columns=FEATS), feature_names=FEATS), output_margin=True)
        sim = th.simulate_margin(levels, structure.leaf_value, th.thermometer_bits(Xt, B), B)
        err = float(np.max(np.abs(sim - plain)))
        assert err < 1e-5, f"B={B}: simulator differs from the booster by {err}"
        ind = th.client_indicators(levels, th.thermometer_bits(Xt, B), B)
        assert np.max(np.abs(ind @ structure.leaf_value - plain)) < 1e-5
        used = th.used_bit_columns(levels, B)
        print(f"B={B}: {n_checked} splits enumerated OK; simulator vs booster max |diff| {err:.1e}; "
              f"distinct bit columns per level {[len(u) for u in used]} of {F * (B - 1)}", flush=True)

        # real ciphertexts on a 4-level chain
        ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=16384, coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 60])
        ctx.global_scale = 2**40
        ctx.generate_galois_keys()
        plain_vecs = th.folded_plain_vectors(levels, structure.leaf_value, B)
        bits = th.thermometer_bits(Xt[:3], B)
        errs, up, secs = [], 0, []
        for i in range(3):
            t0 = time.perf_counter()
            cts = th.encrypt_bits(ctx, bits[i])
            t_enc = time.perf_counter() - t0
            up = th.serialized_bytes(cts)
            real, timing = th.eval_thermo_tenseal(plain_vecs, cts)
            secs.append((t_enc, timing["select"], timing["path_product"], timing["sum_decrypt"]))
            errs.append(abs(real - plain[i]))
        print(f"B={B}: real (4-level chain) max |real - booster| {max(errs):.2e}; upload {up / 1e6:.1f} MB for {F * (B - 1)} ciphertexts; "
              f"encrypt {np.mean([s[0] for s in secs]):.2f}s select {np.mean([s[1] for s in secs]):.2f}s "
              f"product {np.mean([s[2] for s in secs]):.2f}s sum {np.mean([s[3] for s in secs]):.2f}s", flush=True)
        assert max(errs) < 5e-4, f"B={B}: real circuit differs from the booster by {max(errs)}"
        # packed protocol: one client ciphertext, plaintext matrix product per level
        mats = th.build_packed(levels, structure.leaf_value, B, F)
        errs3, secs3, up3 = [], [], 0
        import tenseal as _ts
        for i in range(3):
            t0 = time.perf_counter()
            real, tm = th.eval_thermo_packed_tenseal(ctx, mats, bits[i])
            secs3.append(time.perf_counter() - t0)
            errs3.append(abs(real - plain[i]))
        up3 = len(_ts.ckks_vector(ctx, [float(b) for b in bits[0]]).serialize())
        print(f"B={B}: PACKED real (4-level chain) max error {max(errs3):.2e}; upload {up3 / 1e6:.2f} MB for 1 ciphertext; "
              f"{np.mean(secs3):.2f} s per stay (encrypt {tm['encrypt']:.2f} select {tm['select']:.2f} product {tm['path_product']:.2f})", flush=True)
        assert max(errs3) < 5e-4, f"B={B}: packed circuit differs from the booster by {max(errs3)}"
        # baseline: client-computed indicators on real ciphertexts (depth 1)
        errs2 = []
        for i in range(3):
            row = th.client_indicators(levels, bits[i : i + 1], B)[0]
            t0 = time.perf_counter()
            real, tm = th.eval_client_indicator_tenseal(ctx, row, structure.leaf_value)
            errs2.append(abs(real - plain[i]))
        print(f"B={B}: client-indicator baseline max error {max(errs2):.2e}", flush=True)
    print("ALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
