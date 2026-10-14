#!/usr/bin/env python3
"""
Build M1's cohort: ED stays -> demographics -> chronic-condition labels.

Pipeline (per m1-notation-gate.md's label-space decision and the locked
"hadm_id-linked hosp diagnoses_icd + Elixhauser chronic grouper" decision):

    ed/edstays.csv.gz  (all ED stays, demographics, hadm_id link)
        -> join hosp/patients.csv.gz on subject_id  (anchor_age -> age at stay)
        -> join hosp/diagnoses_icd.csv.gz on hadm_id  (chronic-condition labels)
        -> Elixhauser comorbidity grouper (ICD-9 + ICD-10 aware) per diagnosis code
        -> select the top-K (5 target, 10 ceiling) most prevalent categories
        -> multi-hot label matrix over that fixed label space

Chronic-condition label source: hosp/diagnoses_icd rather than ed/diagnosis.csv.
ed/diagnosis.csv captures the acute ED presenting problem; hosp/diagnoses_icd,
linked through the admission's hadm_id, is what a published chronic-comorbidity
grouper is designed to run over. This trades cohort size (only ED stays that
result in an admission carry a hadm_id) for label validity -- both numbers are
reported in the cohort-flow table below, never silently dropped.

Grouper choice -- Elixhauser, not AHRQ CCSR (a deviation from the plan's
initially-proposed CCSR-primary default, decided and recorded here rather than
left implicit): AHRQ's CCSR reference file (confirmed downloadable, live-tested
2026-09-11: https://hcup-us.ahrq.gov/toolssoftware/ccsr/DXCCSR_v2025-1.zip) is
ICD-10-CM only. MIMIC-IV's hosp/diagnoses_icd spans the ICD-9-to-10 transition
(icd_version value_counts, verified 2026-09-11: 3,455,747 ICD-10 rows,
2,908,741 ICD-9 rows -- 45.7% of all diagnosis rows are ICD-9). Using CCSR
alone would silently drop nearly half the diagnosis history and bias the
cohort toward the more recent era. The Elixhauser comorbidity index (Quan H,
Sundararajan V, Halfon P, et al. "Coding algorithms for defining Comorbidities
in ICD-9-CM and ICD-10 administrative data." Med Care. 2005 Nov;43(11):1130-9)
already has both an ICD-9 and an ICD-10 code list -- and per the standing
project rule to build on vendored code rather than reimplement, it is already
vendored in this project's module2/mimic4ed-benchmark-main tree
(Benchmark_scripts/medcodes/diagnoses/comorbidities.py::elixhauser), imported
directly here rather than re-copied. Its ~29 categories (hypertension,
diabetes with/without complication, congestive heart failure, chronic
pulmonary disease, renal failure, etc.) are also a closer semantic match to
"chronic condition" than CCSR's ~530 general-purpose clinical categories,
which mix acute and chronic presentations with no built-in chronicity flag.

Usage:
    python build_cohort.py [--top-k N] [--output data/m1_cohort.parquet]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mimic_paths import (  # noqa: E402
    default_mimic_root,
    load_diagnoses_icd,
    load_edstays,
    load_medrecon,
    load_patients,
)

# module2/mimic4ed-benchmark-main/Benchmark_scripts is where the vendored
# `medcodes` package (Elixhauser/Charlson comorbidity mappers) lives.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))


def _add_medcodes_to_path() -> None:
    for candidate in (
        os.path.join(_PROJECT_ROOT, "module 2", "mimic4ed-benchmark-main", "Benchmark_scripts"),
        os.path.join(_PROJECT_ROOT, "module2", "mimic4ed-benchmark-main", "Benchmark_scripts"),
    ):
        if os.path.isdir(candidate) and candidate not in sys.path:
            sys.path.insert(0, candidate)


_add_medcodes_to_path()
from medcodes.diagnoses.comorbidities import elixhauser  # noqa: E402

TOP_K_TARGET = 5
TOP_K_CEILING = 10

DEFAULT_COHORT_OUTPUT = os.path.join(_HERE, "..", "data", "m1_cohort.parquet")
DEFAULT_LABELSPACE_OUTPUT = os.path.join(_HERE, "..", "results", "m1_label_space.json")
DEFAULT_FLOW_OUTPUT = os.path.join(_HERE, "..", "results", "m1_cohort_flow.json")


def build_diagnosis_comorbidity_map(diagnoses: pd.DataFrame) -> dict[int, set[str]]:
    """hadm_id -> set of Elixhauser category names present in that admission."""
    hadm_categories: dict[int, set[str]] = {}
    # Group once, iterate per-admission -- calling elixhauser() per (code, version)
    # pair with memoization avoids re-running prefix matching for repeated codes.
    cache: dict[tuple[str, int], list[str]] = {}
    for hadm_id, group in diagnoses.groupby("hadm_id", sort=False):
        cats: set[str] = set()
        for icd_code, icd_version in zip(group["icd_code"], group["icd_version"]):
            key = (icd_code, int(icd_version))
            if key not in cache:
                try:
                    cache[key] = elixhauser(icd_code, icd_version=int(icd_version))
                except Exception:
                    cache[key] = []
            cats.update(cache[key])
        hadm_categories[hadm_id] = cats
    return hadm_categories


def compute_age_at_stay(edstays: pd.DataFrame, patients: pd.DataFrame) -> pd.Series:
    """Standard MIMIC-IV age reconstruction: age at time T = anchor_age +
    (T.year - anchor_year). intime's year is used as T."""
    merged = edstays[["subject_id", "intime"]].merge(
        patients[["subject_id", "anchor_age", "anchor_year"]], on="subject_id", how="left"
    )
    intime_year = pd.to_datetime(merged["intime"]).dt.year
    age = merged["anchor_age"] + (intime_year - merged["anchor_year"])
    age.index = edstays.index
    return age


def build_cohort(top_k: int = TOP_K_TARGET) -> tuple[pd.DataFrame, dict, dict]:
    if not (5 <= top_k <= TOP_K_CEILING):
        raise ValueError(f"top_k must be between 5 and {TOP_K_CEILING} per the locked label-space decision, got {top_k}")

    flow: dict[str, int] = {}

    edstays = load_edstays()
    flow["total_ed_stays"] = len(edstays)

    with_hadm = edstays[edstays["hadm_id"].notna()].copy()
    flow["ed_stays_with_hadm_id"] = len(with_hadm)
    with_hadm["hadm_id"] = with_hadm["hadm_id"].astype("int64")

    patients = load_patients()
    linked_to_patients = with_hadm[with_hadm["subject_id"].isin(patients["subject_id"])]
    flow["stays_linked_to_patients_table"] = len(linked_to_patients)

    diagnoses = load_diagnoses_icd()
    diagnoses = diagnoses[diagnoses["hadm_id"].notna()].copy()
    diagnoses["hadm_id"] = diagnoses["hadm_id"].astype("int64")
    hadm_with_diagnosis = set(diagnoses["hadm_id"].unique())
    stays_with_diagnosis = with_hadm[with_hadm["hadm_id"].isin(hadm_with_diagnosis)]
    flow["stays_with_at_least_one_hosp_diagnosis"] = len(stays_with_diagnosis)

    medrecon = load_medrecon()
    stay_ids_with_meds = set(medrecon["stay_id"].unique())
    stays_with_meds = stays_with_diagnosis[stays_with_diagnosis["stay_id"].isin(stay_ids_with_meds)]
    flow["stays_with_at_least_one_medrecon_entry"] = len(stays_with_meds)

    cohort = stays_with_meds.copy()

    # Restrict diagnosis lookups to only the hadm_ids actually in the cohort,
    # so the Elixhauser pass (the slow step) doesn't do wasted work on
    # admissions that were already excluded by the medication-availability filter.
    cohort_hadm_ids = set(cohort["hadm_id"].unique())
    diagnoses_in_cohort = diagnoses[diagnoses["hadm_id"].isin(cohort_hadm_ids)]
    hadm_categories = build_diagnosis_comorbidity_map(diagnoses_in_cohort)

    # Prevalence across the cohort (count of admissions with >=1 diagnosis in
    # that category), used to select the top-K label space.
    category_counts: Counter = Counter()
    for hadm_id in cohort["hadm_id"]:
        for cat in hadm_categories.get(hadm_id, ()):
            category_counts[cat] += 1

    total_cohort_size = len(cohort)
    prevalence_table = [
        {"category": cat, "n_admissions": n, "prevalence": round(n / total_cohort_size, 4)}
        for cat, n in category_counts.most_common()
    ]
    selected_labels = [row["category"] for row in prevalence_table[:top_k]]

    age = compute_age_at_stay(cohort, patients)
    cohort = cohort.assign(age=age)
    # `patients.csv` is the canonical patient dimension table, so its gender is
    # the one carried forward; `edstays.gender` is kept only long enough to
    # cross-check. A 2026-09-11 audit found the reverse had been true in an
    # earlier version -- the merge's suffixes=("", "_patients") left the
    # *edstays* column holding the bare name "gender", so the edstays value was
    # silently the one written out, contradicting this module's own
    # documentation. The rename below makes the intended source explicit rather
    # than depending on which frame won an unsuffixed column name.
    cohort = cohort.rename(columns={"gender": "gender_edstays"}).merge(
        patients[["subject_id", "gender"]].drop_duplicates("subject_id"),
        on="subject_id",
        how="left",
    )
    # Cross-check: edstays.gender vs patients.gender should agree; report any mismatch honestly rather than assuming.
    gender_mismatch = int((cohort["gender"] != cohort["gender_edstays"]).sum())

    label_cols = {}
    for cat in selected_labels:
        col_name = "label_" + cat.replace(" ", "_").replace(",", "").replace("/", "_")
        label_cols[col_name] = [1 if cat in hadm_categories.get(h, ()) else 0 for h in cohort["hadm_id"]]
    labels_df = pd.DataFrame(label_cols, index=cohort.index)

    n_any_positive = int((labels_df.sum(axis=1) > 0).sum())

    result = pd.concat(
        [cohort[["subject_id", "hadm_id", "stay_id", "age", "gender"]].reset_index(drop=True), labels_df.reset_index(drop=True)],
        axis=1,
    )

    label_space_report = {
        "top_k": top_k,
        "selected_labels": selected_labels,
        "full_prevalence_table_all_elixhauser_categories": prevalence_table,
        "cohort_size_for_prevalence": total_cohort_size,
        "n_stays_with_at_least_one_selected_label": n_any_positive,
        "fraction_stays_with_at_least_one_selected_label": round(n_any_positive / total_cohort_size, 4) if total_cohort_size else 0.0,
        "gender_field_mismatch_edstays_vs_patients": gender_mismatch,
    }

    flow["final_cohort_size"] = len(result)

    return result, label_space_report, flow


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=TOP_K_TARGET)
    parser.add_argument("--output", default=DEFAULT_COHORT_OUTPUT)
    parser.add_argument("--label-space-output", default=DEFAULT_LABELSPACE_OUTPUT)
    parser.add_argument("--flow-output", default=DEFAULT_FLOW_OUTPUT)
    args = parser.parse_args()

    print(f"MIMIC root: {default_mimic_root()}", flush=True)
    cohort, label_space_report, flow = build_cohort(top_k=args.top_k)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    cohort.to_parquet(args.output, index=False)

    os.makedirs(os.path.dirname(args.label_space_output), exist_ok=True)
    with open(args.label_space_output, "w", encoding="utf-8") as f:
        json.dump(label_space_report, f, indent=2)

    os.makedirs(os.path.dirname(args.flow_output), exist_ok=True)
    with open(args.flow_output, "w", encoding="utf-8") as f:
        json.dump(flow, f, indent=2)

    print("--- cohort flow ---")
    print(json.dumps(flow, indent=2))
    print("--- label space ---")
    print(json.dumps({k: v for k, v in label_space_report.items() if k != "full_prevalence_table_all_elixhauser_categories"}, indent=2))
    print(f"\ncohort written to {args.output} ({len(cohort)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
