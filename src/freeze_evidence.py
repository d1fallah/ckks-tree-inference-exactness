"""
M1 -- Final evidence freeze (Q1 Paper Finalization Guide Sec 5).

Generates, in one pass, everything the guide's evidence freeze requires:
  - results/m1_evidence_manifest.json  -- SHA-256 of every artifact the manuscript can cite, git
    HEAD, tree-clean status, the three venvs' installed packages, the superseded-artifact deny
    list (src/paper_data.py), and a dated amendment log.
  - M1_FINAL_EVIDENCE_FREEZE.md         -- rendered from docs/freeze_template.md by filling its
    <!-- GENERATED:... --> markers; the surrounding prose is never touched.
  - environment/requirements-*.txt      -- machine-installable per-venv requirement files.

This document is GENERATED, never hand-written, specifically so it cannot silently drift from the
artifacts it certifies -- src/verify_paper_assets.py re-checks every hash it records.

Amendment discipline (guide Sec 5: "no scientific parameter should be changed after this freeze
without a dated amendment"): if results/m1_evidence_manifest.json already exists and ANY recorded
hash would change, this script REFUSES to overwrite it unless run with --amend "<date> <reason>",
in which case the change is appended to manifest["amendments"] rather than silently replacing the
prior record.

Usage:
    python freeze_evidence.py                          # first freeze, or a no-op re-run
    python freeze_evidence.py --amend "2026-09-22 fixed a table caption typo"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(_HERE, "..")
sys.path.insert(0, _HERE)
from paper_data import SUPERSEDED_ARTIFACTS  # noqa: E402

MANIFEST_PATH = os.path.join(ROOT, "results", "m1_evidence_manifest.json")
FREEZE_MD_PATH = os.path.join(ROOT, "M1_FINAL_EVIDENCE_FREEZE.md")
TEMPLATE_PATH = os.path.join(ROOT, "docs", "freeze_template.md")
ENV_DIR = os.path.join(ROOT, "environment")

VENVS = {
    "m1": os.path.expanduser("~/.venvs/m1/bin/python"),
    "m1-concreteml": os.path.expanduser("~/.venvs/m1-concreteml/bin/python"),
    "m1-estimator": os.path.expanduser("~/.venvs/m1-estimator/bin/python"),
}

# Files whose bytes are part of the frozen evidence. Globs are expanded at run time.
#
# `protected=True` marks SCIENTIFIC EVIDENCE in the guide's Sec 5 sense (results, frozen data,
# locked configuration): a hash change here is refused without --amend, per "no scientific
# parameter should be changed after this freeze without a dated amendment." `protected=False`
# marks tooling (src/*.py, the generator and verifier scripts themselves): these are hashed and
# recorded for provenance -- a table's regeneration command is meaningless without knowing which
# script version produced it -- but they are EXPECTED to keep changing during paper-prep tooling
# work (fixing a caption, adding a table), and gating every such edit behind a dated amendment
# would make the amendment log meaningless noise rather than a record of real protocol changes.
MANIFEST_GLOBS = [
    ("results/*.json", True),
    ("data/models/*.json", True),
    ("data/poly_coeffs_v1.json", True),
    ("data/atc_crosswalk.json", True),
    ("data/m1_cohort.parquet", True),
    ("data/m1_features.parquet", True),
    ("configs/paper_track_m1.yaml", True),
    ("src/*.py", False),
]
PROTECTED_GLOBS = {pattern for pattern, protected in MANIFEST_GLOBS if protected}

# The guide's Sec 6 result -> artifact map, plus the regeneration command for each principal table.
RESULT_ARTIFACT_MAP = [
    ("Plaintext performance", "results/m1_plaintext_baseline.json", "src/train_plaintext.py"),
    ("k=1/3/5/7/9 utility (SIMULATED)", "results/m1_ckks_sweep_simulated.json, results/m1_ckks_sweep_k1_simulated.json",
     "src/eval_ckks.py; src/eval_ckks_k1.py"),
    ("Real CKKS validation", "results/m1_encrypted_multiseed.json", "src/eval_ckks_encrypted_multiseed.py"),
    ("k-selection", "results/m1_k_selection.json", "src/select_k.py"),
    ("Scaling (utility)", "results/m1_scaling_multiseed.json", "src/run_scaling_multiseed.py"),
    ("Scaling (memory)", "results/m1_scaling_memory.json", "src/measure_scaling_memory.py"),
    ("Depth (k in 3,5,7,9)", "results/m1_depth_probe.json", "src/probe_multiplicative_depth.py"),
    ("Depth (k=1)", "results/m1_depth_budget_k1.json", "src/derive_depth_budget_k1.py"),
    ("Security", "results/m1_lwe_security_estimate.json, results/m1_lwe_security_summary.json",
     "src/estimate_lwe_security.py; src/summarize_lwe_security.py"),
    ("Per-label selection", "results/m1_per_label_selection.json", "src/analyze_per_label_selection.py"),
    ("TFHE calibration reference", "results/m1_tfhe_calibration.json", "src/eval_tfhe.py"),
    ("Latency/memory/ciphertext breakdown", "results/m1_latency_breakdown.json", "src/measure_ckks_latency.py"),
    ("Gamma selection", "docs/gamma_selection_protocol.md, results/m1_gamma_anchor_selection.json",
     "src/select_gamma_anchor.py"),
    ("Leakage/threshold audit", "docs/leakage_and_threshold_audit.md", "(audit document; no script)"),
    ("Reproduction (end to end)", "results/e2e_rerun_2026-09-21/", "experiments/run_reproducible_m1.py"),
]


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", ROOT, *args], text=True).strip()
    except Exception:
        return "unavailable"


def collect_manifest_files() -> dict[str, str]:
    import glob

    self_path = os.path.relpath(MANIFEST_PATH, ROOT).replace(os.sep, "/")
    out: dict[str, str] = {}
    for pattern, _required in MANIFEST_GLOBS:
        for path in sorted(glob.glob(os.path.join(ROOT, pattern))):
            rel = os.path.relpath(path, ROOT).replace(os.sep, "/")
            if rel == self_path:
                continue  # the manifest cannot include a hash of itself
            out[rel] = sha256_file(path)
    return out


def pip_freeze(venv_python: str) -> list[str] | None:
    if not os.path.exists(venv_python):
        return None
    try:
        out = subprocess.check_output([venv_python, "-m", "pip", "list", "--format=freeze"], text=True)
        return sorted(out.strip().splitlines())
    except Exception as exc:  # pragma: no cover - environment-dependent
        return [f"# pip freeze failed: {exc}"]


def estimator_commit() -> str | None:
    path = os.path.join(ROOT, "results", "m1_lwe_security_estimate.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return d.get("estimator", {}).get("commit")


def write_environment_files(freeze: list[str] | None, nodeps_pkgs: list[str], concreteml: list[str] | None,
                            estimator: list[str] | None, est_commit: str | None) -> None:
    os.makedirs(ENV_DIR, exist_ok=True)

    def strip_pkgs(lines: list[str] | None, drop_prefixes: tuple[str, ...]) -> list[str]:
        if lines is None:
            return []
        return [ln for ln in lines if not any(ln.lower().startswith(p.lower()) for p in drop_prefixes)]

    with open(os.path.join(ENV_DIR, "requirements-m1.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("# generated by src/freeze_evidence.py -- do not hand-edit\n")
        f.write("# main Track A venv (~/.venvs/m1): TenSEAL/CKKS + plaintext XGBoost track\n")
        f.write("# xgboost is installed separately, --no-deps, from requirements-m1-nodeps.txt --\n")
        f.write("# installing it normally here pulls in unused CUDA/NCCL wheels.\n")
        f.write("\n".join(strip_pkgs(freeze, ("xgboost", "nvidia-"))) + "\n")

    with open(os.path.join(ENV_DIR, "requirements-m1-nodeps.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("# generated by src/freeze_evidence.py -- install with: pip install --no-deps -r this-file\n")
        f.write("\n".join(nodeps_pkgs) + "\n")

    with open(os.path.join(ENV_DIR, "requirements-m1-concreteml.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("# generated by src/freeze_evidence.py -- isolated Concrete-ML/TFHE venv (~/.venvs/m1-concreteml)\n")
        f.write("--extra-index-url https://download.pytorch.org/whl/cpu\n")
        f.write("\n".join(concreteml or ["# venv not found at generation time"]) + "\n")

    with open(os.path.join(ENV_DIR, "requirements-m1-estimator.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("# generated by src/freeze_evidence.py -- lattice-estimator venv (~/.venvs/m1-estimator)\n")
        f.write("\n".join(estimator or ["# venv not found at generation time"]) + "\n")

    with open(os.path.join(ENV_DIR, "setup_venvs.sh"), "w", encoding="utf-8", newline="\n") as f:
        commit_line = est_commit or "<see results/m1_lwe_security_estimate.json:estimator.commit>"
        f.write(f"""#!/usr/bin/env bash
# generated by src/freeze_evidence.py -- the only file that knows install ORDER and flags.
# Three isolated venvs are required: the tracks pin mutually incompatible dependency versions
# (numpy/xgboost/scikit-learn), and there is no combination of pins that satisfies all three at once.
set -euo pipefail

python3.11 -m venv ~/.venvs/m1
~/.venvs/m1/bin/pip install -r environment/requirements-m1.txt
~/.venvs/m1/bin/pip install --no-deps -r environment/requirements-m1-nodeps.txt

python3.11 -m venv ~/.venvs/m1-concreteml
~/.venvs/m1-concreteml/bin/pip install -r environment/requirements-m1-concreteml.txt

python3.11 -m venv ~/.venvs/m1-estimator
~/.venvs/m1-estimator/bin/pip install -r environment/requirements-m1-estimator.txt
git clone https://github.com/malb/lattice-estimator ~/lattice-estimator
git -C ~/lattice-estimator checkout {commit_line}
""")


def render_freeze_doc(manifest: dict) -> None:
    with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
        template = f.read()

    def fill(name: str, body: str, text: str) -> str:
        start, end = f"<!-- GENERATED:{name} -->", f"<!-- /GENERATED:{name} -->"
        assert start in text and end in text, f"template missing {name} markers"
        pre, rest = text.split(start, 1)
        _mid, post = rest.split(end, 1)
        return f"{pre}{start}\n{body}\n{end}{post}"

    header = (
        f"- **Frozen:** {manifest['frozen_at_utc']}\n"
        f"- **Evidence commit (git HEAD at freeze time):** `{manifest['git_head']}`\n"
        f"- **Working tree clean for evidence paths (`results/`, `data/`, `src/`, `configs/`):** "
        f"{manifest['git_working_tree_clean_for_evidence_paths']}\n"
        f"- **Manifest:** `results/m1_evidence_manifest.json` ({len(manifest['files'])} files hashed)\n"
        f"- **CKKS configuration:** locked, N=16384, chain [60,40x7,60], scale 2^40, security 128-bit "
        f"(TC128, SEAL-enforced; independent estimate 138.5 bits MATZOV / 107.2 bits core-SVP)\n"
        f"- **Coefficient version:** {manifest['coefficient_version']}, gamma anchor percentile "
        f"{manifest['gamma_anchor_percentile']}, gamma = {manifest['gamma']}\n"
        f"- **Threshold policy:** {manifest['threshold_policy']}\n"
        f"- **Model:** {manifest['model_config']}\n"
        f"- **Seeds:** {manifest['seeds']}\n"
        f"- **Split:** {manifest['split']}\n"
    )
    text = fill("header", header, template)

    rows = "\n".join(f"| {name} | `{artifact}` | `{cmd}` |" for name, artifact, cmd in RESULT_ARTIFACT_MAP)
    artifact_map = "| Manuscript result | Artifact | Regeneration command |\n|---|---|---|\n" + rows
    text = fill("artifact_map", artifact_map, text)

    commands = (
        "```\n"
        "python experiments/run_reproducible_m1.py --config configs/paper_track_m1.yaml   # all stages\n"
        "python experiments/run_reproducible_m1.py --config configs/paper_track_m1.yaml "
        "--stage <name>   # one stage\n"
        "```\n"
        "Stage names are listed in `experiments/run_reproducible_m1.py`'s `CORE_STAGES`. "
        "Each stage's own regeneration command (its script, run from `src/` in the `~/.venvs/m1` venv "
        "unless noted otherwise) is in the artifact map above."
    )
    text = fill("commands", commands, text)

    manifest_summary = (
        f"{len(manifest['files'])} files hashed across results/, data/models/, the frozen coefficient "
        f"file, the cohort and feature parquets, the config, and every src/*.py. "
        f"Full listing: `results/m1_evidence_manifest.json`."
    )
    text = fill("manifest_summary", manifest_summary, text)

    env_lines = []
    for venv, pkgs in manifest["venvs"].items():
        status = f"{len(pkgs)} packages" if pkgs is not None else "NOT FOUND at freeze time"
        env_lines.append(f"- `~/.venvs/{venv}`: {status} (see `environment/requirements-{venv}.txt`)")
    text = fill("environment", "\n".join(env_lines), text)

    superseded = "\n".join(f"- `{name}` -- {reason}" for name, reason in sorted(SUPERSEDED_ARTIFACTS.items()))
    text = fill("superseded", superseded, text)

    amendments = manifest.get("amendments", [])
    amend_text = "\n".join(f"- **{a['date_and_reason']}**" for a in amendments) if amendments else "None yet."
    text = fill("amendments", amend_text, text)

    with open(FREEZE_MD_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def load_baseline_config() -> dict:
    path = os.path.join(ROOT, "results", "m1_plaintext_baseline.json")
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return d


def load_coeff_meta() -> dict:
    path = os.path.join(ROOT, "results", "m1_encrypted_multiseed.json")
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return d.get("coefficients", {})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--amend", default=None, help='"<date> <reason>" -- required if a hash changed since the last freeze')
    args = parser.parse_args()

    files = collect_manifest_files()
    baseline = load_baseline_config()
    coeff = load_coeff_meta()

    freeze_lines = pip_freeze(VENVS["m1"])
    # xgboost is separated out for the --no-deps requirements file.
    nodeps = [ln for ln in (freeze_lines or []) if ln.lower().startswith("xgboost")]
    concreteml_lines = pip_freeze(VENVS["m1-concreteml"])
    estimator_lines = pip_freeze(VENVS["m1-estimator"])

    import fnmatch

    def is_protected(rel_path: str) -> bool:
        return any(fnmatch.fnmatch(rel_path, pattern) for pattern in PROTECTED_GLOBS)

    prior_amendments = []
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            prior = json.load(f)
        prior_amendments = prior.get("amendments", [])
        changed = {p for p in files if prior.get("files", {}).get(p) != files[p]}
        changed |= {p for p in prior.get("files", {}) if p not in files}
        protected_changed = {p for p in changed if is_protected(p)}
        if protected_changed and not args.amend:
            print("REFUSING to overwrite the evidence freeze: the following SCIENTIFIC EVIDENCE artifacts "
                  "changed hash", file=sys.stderr)
            for p in sorted(protected_changed):
                print(f"  {p}", file=sys.stderr)
            print('Re-run with --amend "<date> <reason>" if this is an intentional, dated amendment.', file=sys.stderr)
            return 1
        if protected_changed and args.amend:
            prior_amendments = prior_amendments + [{"date_and_reason": args.amend, "changed_files": sorted(protected_changed)}]
        # Non-protected (tooling) changes are applied silently -- no amendment needed, but still
        # visible in the diff between successive manifest.json commits for anyone who looks.

    manifest = {
        "frozen_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_head": git("rev-parse", "--short", "HEAD"),
        "git_working_tree_clean_for_evidence_paths": git("status", "--short", "--", "results", "data", "src", "configs") == "",
        "files": files,
        "venvs": {
            "m1": freeze_lines,
            "m1-concreteml": concreteml_lines,
            "m1-estimator": estimator_lines,
        },
        "superseded_artifacts": SUPERSEDED_ARTIFACTS,
        "model_config": f"{baseline['n_estimators']} trees, max_depth {baseline['max_depth']}, "
                         f"MultiOutputClassifier (one binary XGBoost booster per label)",
        "seeds": baseline["seeds"],
        "split": baseline["split"],
        "threshold_policy": baseline["threshold_policy"],
        "coefficient_version": coeff.get("version", "unknown"),
        "gamma": coeff.get("gamma", "unknown"),
        "gamma_anchor_percentile": coeff.get("gamma_anchor_percentile", "unknown"),
        "amendments": prior_amendments,
    }

    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    write_environment_files(freeze_lines, nodeps, concreteml_lines, estimator_lines, estimator_commit())
    render_freeze_doc(manifest)

    print(f"manifest: {len(files)} files hashed -> {MANIFEST_PATH}")
    print(f"freeze doc -> {FREEZE_MD_PATH}")
    print(f"environment files -> {ENV_DIR}/")
    if manifest["amendments"]:
        print(f"{len(manifest['amendments'])} amendment(s) recorded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
