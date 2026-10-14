#!/usr/bin/env python3
"""
Single-command reproduction entrypoint for M1's Track A headline results,
per CLAUDE.md Sec 1.3 and guide v4.1 Sec 7 ("Scripts committed: ...
reproduce.sh").

This orchestrates the pipeline stages in dependency order. It does NOT
re-run the MIMIC-IV-ED cohort/feature extraction or the ATC crosswalk by
default (those require the raw MIMIC-IV-ED files and the live RxNav API;
--from-scratch enables them) -- by default it starts from the committed
data/m1_cohort.parquet and data/m1_features.parquet, which is what makes a
clean clone reproducible without external dependencies.

CKKS-path stages (plaintext training, circuit derivation, coefficient
generation, Gate M1-A, the k-sweep simulation) run in the ~/.venvs/m1 venv
(TenSEAL). The TFHE stage runs in the separate ~/.venvs/m1-concreteml venv
(pinned xgboost 1.6.2), per m1_vendor_provenance.md's isolated-venv
rationale. This script assumes both venvs already exist at those paths
(see environment_snapshot.md for how they were built) and does not attempt
to create them.

Usage:
    python experiments/run_reproducible_m1.py --config configs/paper_track_m1.yaml
    python experiments/run_reproducible_m1.py --config configs/paper_track_m1.yaml --from-scratch
    python experiments/run_reproducible_m1.py --config configs/paper_track_m1.yaml --stage k_sweep
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
MODULE_ROOT = os.path.join(_HERE, "..")
SRC = os.path.join(MODULE_ROOT, "src")

M1_VENV_PY = os.path.expanduser("~/.venvs/m1/bin/python")
CONCRETEML_VENV_PY = os.path.expanduser("~/.venvs/m1-concreteml/bin/python")
ESTIMATOR_VENV_PY = os.path.expanduser("~/.venvs/m1-estimator/bin/python")  # Sage (passagemath) + a pinned malb/lattice-estimator checkout

# (stage name, venv, script, [args]) in dependency order.
FROM_SCRATCH_STAGES = [
    ("atc_crosswalk", M1_VENV_PY, "build_atc_crosswalk.py", []),
    ("cohort", M1_VENV_PY, "build_cohort.py", []),
    ("features", M1_VENV_PY, "features.py", []),
]
# Closure-guide stages (M1_CLOSURE_LOG.md) are interleaved in dependency order. Notes:
#   * select_gamma_anchor precedes poly_coefficients: the coefficient generator READS the anchor from that stage's artifact
#     (docs/anchor_reselection_preregistration.md) and refuses to run without it. ~55 min, ~4 GB.
#   * encrypted_singleseed (eval_ckks_encrypted.py) was previously NOT a stage although select_k.py reads its output, so k* was not
#     reproducible by the single documented command; it now is.
#   * The heavy simulation stages (select_gamma_anchor, k_sweep, k_sweep_k1, scaling_multiseed) each need ~4 GB: run one at a time on a
#     ~8 GB host. The orchestrator is sequential, so this holds. The latency stages want a quiet machine.
CORE_STAGES = [
    ("plaintext_baseline", M1_VENV_PY, "train_plaintext.py", []),
    ("circuit_params", M1_VENV_PY, "derive_circuit_params.py", []),
    ("select_gamma_anchor", M1_VENV_PY, "select_gamma_anchor.py", []),
    ("poly_coefficients", M1_VENV_PY, "generate_poly_coefficients.py", []),
    ("security_level", M1_VENV_PY, "verify_ckks_security_level.py", []),
    ("modulus_chain", M1_VENV_PY, "dump_ckks_modulus_chain.py", []),
    ("lwe_security_estimate", ESTIMATOR_VENV_PY, "estimate_lwe_security.py", []),
    ("depth_probe", M1_VENV_PY, "probe_multiplicative_depth.py", []),
    ("gate_m1a", M1_VENV_PY, "run_gate_m1a.py", []),
    ("k_sweep", M1_VENV_PY, "eval_ckks.py", []),
    ("k_sweep_k1", M1_VENV_PY, "eval_ckks_k1.py", []),
    ("encrypted_singleseed", M1_VENV_PY, "eval_ckks_encrypted.py", []),
    ("real_ckks_samples", M1_VENV_PY, "draw_real_ckks_samples.py", []),
    ("encrypted_multiseed", M1_VENV_PY, "eval_ckks_encrypted_multiseed.py", []),
    ("latency_breakdown", M1_VENV_PY, "measure_ckks_latency.py", []),
    ("tfhe_calibration", CONCRETEML_VENV_PY, "eval_tfhe.py", []),
    # scaling_ablation's OWN OUTPUT (m1_scaling_ablation.json) is in paper_data.SUPERSEDED_ARTIFACTS --
    # the single-seed pilot's headline cell did not replicate; scaling_multiseed below is the result
    # every paper asset actually sources. The stage is kept (not deleted) only as the reproducibility
    # gate scaling_multiseed's own seed-42 cells are checked against -- see run_scaling_multiseed.py's
    # "reproducibility gate vs the committed pilot" section. No table or figure may read its output.
    ("scaling_ablation", M1_VENV_PY, "run_scaling.py", []),
    ("scaling_multiseed", M1_VENV_PY, "run_scaling_multiseed.py", []),
    ("scaling_memory", M1_VENV_PY, "measure_scaling_memory.py", []),
    ("depth_budget_k1", M1_VENV_PY, "derive_depth_budget_k1.py", []),
    ("framing_evidence", M1_VENV_PY, "analyze_framing_evidence.py", []),
    ("k_selection", M1_VENV_PY, "select_k.py", []),
    ("per_label_selection", M1_VENV_PY, "analyze_per_label_selection.py", []),
    # Evidence freeze runs BEFORE table/figure generation, not after: tables and figures are then
    # validated against a manifest that was pinned first, so a later, uncommitted change to a source
    # JSON is caught by verify_assets below rather than silently baked into a table.
    ("evidence_freeze", M1_VENV_PY, "freeze_evidence.py", []),
    ("latex_tables", M1_VENV_PY, "generate_latex_tables.py", []),
    ("figures", M1_VENV_PY, "generate_figures.py", []),
    # Writes the F1/F2 TikZ SOURCES only (paper_assets/figures_src/*.tex); pure text generation,
    # no TeX toolchain needed, so it belongs in the venv pipeline. Compiling those sources to PDF
    # needs Windows-side pdflatex (paper/build_diagrams.ps1) and stays a documented manual step,
    # same as paper/build_paper.ps1 itself -- see that script's own docstring for why.
    ("diagram_sources", M1_VENV_PY, "build_diagrams.py", []),
    ("paper_macros", M1_VENV_PY, "generate_paper_macros.py", []),
    ("verify_assets", M1_VENV_PY, "verify_paper_assets.py", []),
    ("notation_gate", M1_VENV_PY, "check_notation.py", [".."]),
]


def run_stage(name: str, venv_py: str, script: str, extra_args: list[str]) -> None:
    if not os.path.exists(venv_py):
        raise SystemExit(f"stage '{name}': venv python not found at {venv_py} -- see environment_snapshot.md")
    cmd = [venv_py, script] + extra_args
    print(f"\n=== stage: {name} ({os.path.basename(venv_py)} venv) ===")
    print(f"$ {' '.join(cmd)}  (cwd={SRC})")
    result = subprocess.run(cmd, cwd=SRC)
    if result.returncode != 0:
        raise SystemExit(f"stage '{name}' failed with exit code {result.returncode}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="e.g. configs/paper_track_m1.yaml (read for a config-consistency print only -- stage scripts read their own frozen artifacts, not this file, so no config value is silently duplicated)")
    parser.add_argument("--from-scratch", action="store_true", help="also rebuild the ATC crosswalk, cohort, and features from raw MIMIC-IV-ED (requires the RxNav API and module2/mimic4ed-benchmark-main/)")
    parser.add_argument("--stage", default=None, help="run only this single stage by name, for debugging")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    print(f"config: {args.config}")
    print(f"  seeds={cfg['model']['seeds']}  k_values={cfg['circuit']['k_values']}  "
          f"utility_tolerance={cfg['evaluation']['utility_tolerance_delta_f1']}")

    stages = (FROM_SCRATCH_STAGES if args.from_scratch else []) + CORE_STAGES
    if args.stage:
        stages = [s for s in stages if s[0] == args.stage]
        if not stages:
            raise SystemExit(f"unknown stage '{args.stage}'")

    for name, venv_py, script, extra_args in stages:
        run_stage(name, venv_py, script, extra_args)

    print("\nAll stages completed. See results/*.json for outputs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
