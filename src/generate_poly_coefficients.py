#!/usr/bin/env python3
"""
Generate and freeze the comparison-polynomial coefficient sets c3/c5/c7/c9.

Per the guide §3.1/§3.2, these are generated ONCE, after the global
approximation interval `a` is frozen, then versioned and reused unchanged for
every seed and every k. Regenerating them per seed or per run would make the
k-sweep uncontrolled.

Two candidate target functions are evaluated, because the guide names both
("minimax/Remez approximation of a step or sigmoid-transformed step") and the
choice materially changes the error profile:

  HARD  b(z) = 1 if z < 0 else 0
        The true branch decision. Discontinuous at z=0, so any polynomial
        approximation suffers Gibbs ringing there, and the error does not
        decay uniformly with k.

  SOFT  b(z) = sigmoid(-gamma * z)
        A deliberately smoothed step. Well-conditioned and what FHE practice
        actually uses, but it introduces a fixed smoothing error that does NOT
        vanish as k grows -- so the error floor is set by gamma, not by k.

Both are computed and reported so the M1-0 choice rests on measured error
rather than assertion. gamma is fixed by a stated rule (not tuned): it is set
so the smoothed step reaches SIGMOID_EDGE_LEVEL at a reference point on the
pooled |z| distribution, i.e. gamma = logit(SIGMOID_EDGE_LEVEL) / z_ref.

v2 correction (2026-09-18, see m1-0-implementation-lock.md Amendment v1.2):
v1 anchored z_ref at the interval EDGE a (the 99.9th percentile) rather than
at a point representative of where most data actually falls. Gate M1-A
(M1_Gate_A_Report.md) measured the consequence directly: at that anchoring,
the branch function is only 0.71/0.29 -- nowhere near decisive -- at the
MEDIAN observed |z|, so most leaves get non-negligible weight from paths
XGBoost would not have taken, and increasing k does not fix this (k only
sharpens the polynomial's fit to an already-too-gradual target). Measured
across candidate anchors {p50, p75, p90, p95, p99.9} on the frozen 50-tree
models, p75 gave the best k=9 result and a strong overall profile
(agreement 0.927 -> 0.960, normalized MAE 0.033 -> 0.021 for k=3 -> k=9);
p99.9 (v1) is flat at ~0.85 agreement regardless of k; p90/p95 partially
improve then plateau or worsen at k=9. (Correction 2026-09-21: p50, p75, p90
and p95 are all non-decreasing across k in that table, so monotonicity did
not distinguish p75.)

Provenance audit (2026-09-21, docs/gamma_selection_protocol.md): the original
five-candidate comparison has no producing script or artifact and its split
cannot be established, and the Gate M1-A runs around it were scored on test
rows. The anchor is therefore no longer a literal in this file. It is READ from
results/m1_gamma_anchor_selection.json, written by src/select_gamma_anchor.py
under a criterion fixed in writing before it ran
(docs/anchor_reselection_preregistration.md): validation split, k=5 only,
label-free agreement. That re-selection returned p75, so the frozen
coefficients are unchanged. Run select_gamma_anchor.py before this script.

Approximation method: Chebyshev projection of degree k on [-a, a] (numpy's
`Chebyshev.interpolate`, which samples at Chebyshev nodes). This is near-
minimax, deterministic, and reproducible from this script alone.

Output: data/poly_coeffs_v1.json  (frozen, versioned, committed)
        results/m1_poly_approximation_error.json

Usage:
    python generate_poly_coefficients.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from numpy.polynomial import chebyshev as C

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from notation import K_APPROX_DEGREE_VALUES  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
CIRCUIT_PARAMS = os.path.join(_HERE, "..", "results", "m1_circuit_params.json")
# Optional overrides (default unchanged) so a regeneration check can write elsewhere and never touch the frozen artifacts.
COEFF_OUT = os.environ.get("M1_POLY_COEFF_OUT") or os.path.join(_HERE, "..", "data", "poly_coeffs_v1.json")
ERROR_OUT = os.environ.get("M1_POLY_ERROR_OUT") or os.path.join(_HERE, "..", "results", "m1_poly_approximation_error.json")
ANCHOR_SELECTION = os.path.join(_HERE, "..", "results", "m1_gamma_anchor_selection.json")

# Pre-registered, not tuned: the smoothed step reaches this level at the
# ANCHOR_PERCENTILE reference point (see v2 correction above), not at the
# interval edge.
SIGMOID_EDGE_LEVEL = 0.99


def resolve_anchor_percentile() -> float:
    """The percentile of the pooled |z| distribution (results/m1_circuit_params.json abs_diff_quantiles) used as
    the gamma reference point, READ from the audited selection artifact rather than hardcoded. Fails loudly if the
    artifact is missing, so no run can silently fall back to an unaudited literal."""
    if not os.path.exists(ANCHOR_SELECTION):
        raise SystemExit(f"{ANCHOR_SELECTION} not found: run select_gamma_anchor.py first "
                         "(docs/anchor_reselection_preregistration.md)")
    with open(ANCHOR_SELECTION, "r", encoding="utf-8") as f:
        sel = json.load(f)
    assert sel["scoring_split"].startswith("VALIDATION"), "anchor was not selected on the validation split"
    p = sel["selected_percentile"]
    return int(p) if float(p).is_integer() else float(p)

# Dense grid for measuring approximation error (not used for fitting).
N_ERROR_GRID = 200_001


def hard_step(z: np.ndarray) -> np.ndarray:
    return (z < 0).astype(float)


def soft_step(z: np.ndarray, gamma: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(gamma * z))


def fit_and_measure(target, a: float, degree: int, grid: np.ndarray) -> dict:
    cheb = C.Chebyshev.interpolate(target, deg=degree, domain=[-a, a])
    approx = cheb(grid)
    truth = target(grid)
    err = approx - truth
    # Error excluding a narrow band around the discontinuity, reported
    # separately because for the HARD target the band dominates the max and
    # would otherwise hide how well the polynomial behaves everywhere else.
    band = np.abs(grid) > (0.05 * a)
    return {
        "degree": degree,
        "coefficients_chebyshev_basis": [float(c) for c in cheb.coef],
        "coefficients_power_basis": [float(c) for c in cheb.convert(kind=np.polynomial.Polynomial).coef],
        "max_abs_error": float(np.abs(err).max()),
        "rmse": float(np.sqrt((err**2).mean())),
        "max_abs_error_outside_5pct_band": float(np.abs(err[band]).max()),
        "rmse_outside_5pct_band": float(np.sqrt((err[band] ** 2).mean())),
        "value_at_minus_a": float(cheb(-a)),
        "value_at_plus_a": float(cheb(a)),
        "overshoot_above_1": float(max(0.0, approx.max() - 1.0)),
        "undershoot_below_0": float(max(0.0, -approx.min())),
    }


def main() -> int:
    with open(CIRCUIT_PARAMS, "r", encoding="utf-8") as f:
        params = json.load(f)
    a = params["approximation_interval"]["a_half_width"]
    quantiles = params["approximation_interval"]["abs_diff_quantiles"]
    ANCHOR_PERCENTILE = resolve_anchor_percentile()
    z_ref = quantiles[str(ANCHOR_PERCENTILE)]

    gamma = float(np.log(SIGMOID_EDGE_LEVEL / (1 - SIGMOID_EDGE_LEVEL)) / z_ref)
    grid = np.linspace(-a, a, N_ERROR_GRID)

    hard = {f"k={deg}": fit_and_measure(hard_step, a, deg, grid) for deg in K_APPROX_DEGREE_VALUES}
    soft = {
        f"k={deg}": fit_and_measure(lambda z, deg=deg: soft_step(z, gamma), a, deg, grid)
        for deg in K_APPROX_DEGREE_VALUES
    }

    frozen = {
        "version": "v2",
        "v2_correction": (
            "gamma now anchored at the p75 reference point of the pooled |z| "
            "distribution, not the interval edge (p99.9) used in v1. See "
            "M1_Gate_A_Report.md and m1-0-implementation-lock.md Amendment "
            "v1.2 for the measured comparison across candidate anchors."
        ),
        "generated_from_interval": a,
        "interval": [-a, a],
        "interval_source": "results/m1_circuit_params.json (pooled over all 5 seeds, training split only)",
        "sigmoid_edge_level_preregistered": SIGMOID_EDGE_LEVEL,
        "gamma_anchor_percentile": ANCHOR_PERCENTILE,
        "gamma_anchor_z_ref": z_ref,
        "gamma": gamma,
        "approximation_method": "Chebyshev interpolation at Chebyshev nodes (numpy Chebyshev.interpolate)",
        "targets": {"hard_step": hard, "soft_step": soft},
        "note": (
            "Both target families are frozen here. M1-0 selects ONE as the locked "
            "branch-smoothing function; the other is retained only as the "
            "justification evidence for that choice and is never used in the sweep."
        ),
    }

    os.makedirs(os.path.dirname(COEFF_OUT), exist_ok=True)
    with open(COEFF_OUT, "w", encoding="utf-8") as f:
        json.dump(frozen, f, indent=2)

    summary = {
        "a": a,
        "gamma": gamma,
        "hard_step": {kk: {m: v[m] for m in ("max_abs_error", "rmse", "max_abs_error_outside_5pct_band", "rmse_outside_5pct_band", "overshoot_above_1", "undershoot_below_0")} for kk, v in hard.items()},
        "soft_step": {kk: {m: v[m] for m in ("max_abs_error", "rmse", "max_abs_error_outside_5pct_band", "rmse_outside_5pct_band", "overshoot_above_1", "undershoot_below_0")} for kk, v in soft.items()},
    }
    os.makedirs(os.path.dirname(ERROR_OUT), exist_ok=True)
    with open(ERROR_OUT, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"interval a = {a}   gamma = {gamma:.4f}  (edge level {SIGMOID_EDGE_LEVEL})")
    print()
    for name, block in (("HARD step", hard), ("SOFT step (sigmoid)", soft)):
        print(f"--- {name} ---")
        print(f"  {'k':>3}  {'max|err|':>10}  {'rmse':>10}  {'max|err| off-band':>18}  {'overshoot':>10}")
        for kk, v in block.items():
            print(f"  {kk.split('=')[1]:>3}  {v['max_abs_error']:>10.5f}  {v['rmse']:>10.5f}  "
                  f"{v['max_abs_error_outside_5pct_band']:>18.5f}  {v['overshoot_above_1']:>10.5f}")
        print()
    print(f"frozen coefficients -> {COEFF_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
