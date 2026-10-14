"""
Cohort loader for the M1 continuous-feature task (gate: GATES.md, Question gate signed 2026-10-11).

Inputs are the MIMIC-IV-ED triage vital signs and age; the outcome is the M2 critical outcome
(in-hospital death, or ICU stay beginning within 12 h of ED exit), without any landmark.

The benchmark functions are not reimplemented. They are extracted by name from the vendored
`module2/.../Benchmark_scripts/helpers.py` with `ast` and executed as written, because importing
that file pulls in TensorFlow, which the M1 environment does not have. The vital-sign valid range
is parsed the same way from `data_general_processing.py`, as `m2_hybrid_common.py` already does.

Patient-level rows are held in memory only. Nothing here prints or writes rows; callers print
aggregates (CLAUDE.md section 1.2).
"""
from __future__ import annotations

import ast
import os
import re
import sys
from datetime import timedelta

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from mimic_paths import default_mimic_root  # noqa: E402

ICU_TRANSFER_HOURS = 12
VITAL_FEATURES = ["triage_temperature", "triage_heartrate", "triage_resprate", "triage_o2sat",
                  "triage_sbp", "triage_dbp", "triage_pain"]
FEATURE_COLS = VITAL_FEATURES + ["age"]
OUTCOME_COL = "outcome_critical"
_NEEDED = ("convert_str_to_float", "add_age", "add_inhospital_mortality", "add_outcome_icu_transfer",
           "convert_temp_to_celcius", "remove_outliers", "outlier_removal_imputation",
           "merge_edstays_patients_on_subject", "merge_edstays_admissions_on_subject",
           "merge_edstays_triage_on_subject")


def _bench_dir() -> str:
    return os.path.join(default_mimic_root(), "Benchmark_scripts")


def vendored_helpers() -> dict:
    """The named benchmark functions, executed from their source text."""
    path = os.path.join(_bench_dir(), "helpers.py")
    with open(path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())
    ns: dict = {"pd": pd, "np": np, "re": re, "timedelta": timedelta}
    found = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in _NEEDED:
            exec(compile(ast.Module([node], []), path, "exec"), ns)  # noqa: S102 - in-repo source
            found.add(node.name)
    missing = set(_NEEDED) - found
    if missing:
        raise RuntimeError(f"vendored helpers not found: {sorted(missing)}")
    return ns


def vitals_valid_range() -> dict:
    path = os.path.join(_bench_dir(), "data_general_processing.py")
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    start = text.index("vitals_valid_range = {")
    end = text.index("\n}\n", start) + 3
    ns: dict = {}
    exec(text[start:end], ns)  # noqa: S102 - in-repo source
    return ns["vitals_valid_range"]


def build_cohort(extra: tuple = ()) -> tuple[pd.DataFrame, dict]:
    """Return (frame, flow). The frame has stay_id, subject_id, FEATURE_COLS, any `extra` columns (e.g. triage_acuity) and OUTCOME_COL, with
    out-of-range vital signs already set missing by the benchmark rule (no imputation here).
    `flow` is the aggregate cohort-flow count record."""
    h = vendored_helpers()
    root = default_mimic_root()
    ed = os.path.join(root, "mimic-iv-ed-2.2", "mimic-iv-ed-2.2", "ed")
    hosp = os.path.join(root, "mimic-iv-clinical")

    edstays = pd.read_csv(os.path.join(ed, "edstays.csv.gz"), compression="gzip")
    edstays["intime"] = pd.to_datetime(edstays["intime"])
    edstays["outtime"] = pd.to_datetime(edstays["outtime"])
    patients = pd.read_csv(os.path.join(hosp, "patients.csv.gz"), compression="gzip")
    patients["dod"] = pd.to_datetime(patients["dod"])
    admissions = pd.read_csv(os.path.join(hosp, "admissions.csv.gz"), compression="gzip",
                             usecols=["subject_id", "hadm_id", "dischtime"])
    admissions["dischtime"] = pd.to_datetime(admissions["dischtime"])
    icu = pd.read_csv(os.path.join(hosp, "icustays.csv.gz"), compression="gzip")
    icu["intime"] = pd.to_datetime(icu["intime"])
    triage = pd.read_csv(os.path.join(ed, "triage.csv.gz"), compression="gzip")
    triage = triage.rename(columns={v: f"triage_{v}" for v in
                                    ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "pain", "acuity"]})
    triage["triage_pain"] = triage["triage_pain"].apply(h["convert_str_to_float"]).astype(float)

    flow = {"n_ed_stays": int(len(edstays))}
    df = h["merge_edstays_patients_on_subject"](edstays, patients)
    df = h["merge_edstays_admissions_on_subject"](df, admissions)
    df = h["add_age"](df)
    df = h["add_inhospital_mortality"](df)
    df = h["add_outcome_icu_transfer"](df, icu, ICU_TRANSFER_HOURS)
    df[OUTCOME_COL] = (df["outcome_inhospital_mortality"]
                       | df[f"outcome_icu_transfer_{ICU_TRANSFER_HOURS}h"]).astype(int)
    df = h["merge_edstays_triage_on_subject"](df, triage)
    flow["n_after_merges"] = int(len(df))

    df = df[df["age"] >= 18]
    flow["n_adult"] = int(len(df))
    has_vital = df[VITAL_FEATURES].notna().any(axis=1)
    df = df[has_vital]
    flow["n_with_a_triage_vital"] = int(len(df))

    df = h["convert_temp_to_celcius"](df.copy())
    df = h["remove_outliers"](df, vitals_valid_range())
    keep = ["stay_id", "subject_id"] + FEATURE_COLS + list(extra) + [OUTCOME_COL]
    return df[keep].reset_index(drop=True), flow
