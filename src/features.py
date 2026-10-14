#!/usr/bin/env python3
"""
Build M1's fixed-length feature vector: hierarchical ATC embeddings (with an
ETC fallback for unmapped medications) + demographics.

The feature vector must be a *fixed, bounded* length -- CKKS ciphertext
packing (Gate M1-0's "ciphertext packing layout" item) needs a known slot
count decided once, not a per-stay variable-length vector. Three feature
groups, each documented with the real cardinality/coverage numbers that
justify its size (never an invented round number):

  1. ATC level-1 presence flags (up to 14 anatomical-group letters actually
     observed in the crosswalk -- a small, complete hierarchy top, always
     included regardless of how much of the cohort's medication volume maps
     to ATC at all).
  2. Top-N ETC (Enhanced Therapeutic Class) presence flags, by prevalence in
     the *training* cohort only (never fit on the full cohort, to avoid
     leaking test-set medication-frequency information into the feature
     definition itself) -- the documented fallback for medrecon rows whose
     NDC does not resolve to an ATC class. ETC has no ATC-style prefix
     hierarchy (1,201 distinct flat codes, verified 2026-09-11; top-20 by
     frequency cover ~41% of medrecon rows, top-50 cover ~63%), so its
     features are presence flags over the N most prevalent codes rather than
     a level-based hierarchy.
  3. Two cheap aggregate counts (total medication count, distinct drug count
     for the stay) plus demographics (age, gender).

Usage:
    python features.py --cohort data/m1_cohort.parquet
                        --atc-cache data/atc_crosswalk.json
                        --output data/m1_features.parquet
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mimic_paths import load_medrecon  # noqa: E402
from splits import patient_level_split  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_COHORT_PATH = os.path.join(_HERE, "..", "data", "m1_cohort.parquet")
DEFAULT_ATC_CACHE_PATH = os.path.join(_HERE, "..", "data", "atc_crosswalk.json")
DEFAULT_OUTPUT_PATH = os.path.join(_HERE, "..", "data", "m1_features.parquet")
DEFAULT_FEATURE_REPORT_PATH = os.path.join(_HERE, "..", "results", "m1_feature_space.json")

MIMIC_NULL_NDC = "0"
ALL_ATC1_LETTERS = list("ABCDGHJLMNPRSV")  # the 14 WHO ATC level-1 anatomical groups


def build_ndc_to_atc_levels(atc_cache: dict) -> dict[str, tuple[set[str], set[str]]]:
    """Precompute NDC -> (set of ATC level-1 letters, set of ATC level-2 codes).

    Two things this deliberately does, both fixing 2026-09-11 audit findings:

    1. It keeps **every** ATC class a drug maps to, not just the first. 1,948 of
       the 5,161 mapped NDCs (37.7%) carry more than one ATC class, and 1,420
       of them (27.5%) span more than one level-1 anatomical letter -- taking
       `classes[0]` discarded that, and worse, made the feature depend on
       RxClass's response ordering, which is not a documented stable guarantee.
    2. It derives level 2 (therapeutic subgroup, e.g. "R03") as well as level 1
       (anatomical group, e.g. "R"). The crosswalk resolves 86 distinct level-2
       codes across the cohort; using only the 14 level-1 letters threw away
       most of the hierarchy the roadmap's "hierarchical ATC embeddings"
       actually asks for.

    Precomputed once (O(distinct NDCs), ~9,300 entries) rather than per
    medrecon row -- medrecon has ~2.9M rows and this is on the hot path.
    """
    mapping: dict[str, tuple[set[str], set[str]]] = {}
    for ndc, entry in atc_cache.items():
        classes = entry.get("atc_classes") or []
        if not classes:
            continue
        lvl1 = {c["classId"][0:1] for c in classes if len(c["classId"]) >= 1}
        lvl2 = {c["classId"][0:3] for c in classes if len(c["classId"]) >= 3}
        mapping[ndc] = (lvl1, lvl2)
    return mapping


def all_level2_codes(atc_cache: dict) -> list[str]:
    """The fixed level-2 vocabulary, sorted for a deterministic column order.

    Taken from the crosswalk rather than from the training split: the ATC
    hierarchy is an external, fixed classification, not something estimated
    from data, so there is no leakage in using its full vocabulary. (The ETC
    fallback vocabulary is different -- that one IS prevalence-ranked on
    training patients only, see `select_top_etc_codes`.)
    """
    codes: set[str] = set()
    for entry in atc_cache.values():
        for c in entry.get("atc_classes") or []:
            if len(c["classId"]) >= 3:
                codes.add(c["classId"][0:3])
    return sorted(codes)


def build_stay_medication_index(medrecon: pd.DataFrame, atc_cache: dict) -> dict[int, dict]:
    """stay_id -> {'atc1_set', 'atc2_set', 'etccode_set', 'n_meds', 'n_distinct'}"""
    ndc_to_levels = build_ndc_to_atc_levels(atc_cache)
    index: dict[int, dict] = {}
    for stay_id, group in medrecon.groupby("stay_id", sort=False):
        ndc_strs = group["ndc"].astype(str)
        atc1_set: set[str] = set()
        atc2_set: set[str] = set()
        etccode_set: set[str] = set()
        n_distinct_real = 0
        seen_ndcs: set[str] = set()
        for ndc, etccode in zip(ndc_strs, group["etccode"]):
            is_sentinel = ndc == MIMIC_NULL_NDC
            if not is_sentinel and ndc not in seen_ndcs:
                seen_ndcs.add(ndc)
                n_distinct_real += 1
            levels = None if is_sentinel else ndc_to_levels.get(ndc)
            if levels is not None:
                atc1_set.update(levels[0])
                atc2_set.update(levels[1])
            elif pd.notna(etccode):
                # Reached when the NDC did not resolve to ATC *or* when MIMIC
                # recorded its "0" no-NDC sentinel. The sentinel means "no NDC
                # was captured for this medication", not "no medication" -- an
                # earlier version skipped these rows before this branch, losing
                # 4,219 medication records with a perfectly usable ETC code
                # across 8,012 stays (4.95% of the cohort).
                etccode_set.add(str(etccode))
        index[stay_id] = {
            "n_meds": len(group),
            "n_distinct": n_distinct_real,
            "atc1_set": atc1_set,
            "atc2_set": atc2_set,
            "etccode_set": etccode_set,
        }
    return index


def select_top_etc_codes(stay_index: dict, train_stay_ids: set[int], top_n: int) -> list[str]:
    """Prevalence computed on the training split only, to avoid leaking
    test-set medication-frequency information into the feature definition."""
    counts: Counter = Counter()
    for stay_id in train_stay_ids:
        for code in stay_index.get(stay_id, {}).get("etccode_set", ()):
            counts[code] += 1
    return [code for code, _ in counts.most_common(top_n)]


def build_features(
    cohort: pd.DataFrame,
    medrecon: pd.DataFrame,
    atc_cache: dict,
    train_subject_ids: set[int],
    top_n_etc: int = 30,
) -> tuple[pd.DataFrame, dict]:
    stay_index = build_stay_medication_index(medrecon, atc_cache)
    train_stay_ids = set(cohort.loc[cohort["subject_id"].isin(train_subject_ids), "stay_id"])
    top_etc_codes = select_top_etc_codes(stay_index, train_stay_ids, top_n_etc)
    level2_codes = all_level2_codes(atc_cache)

    empty = {"n_meds": 0, "n_distinct": 0, "atc1_set": set(), "atc2_set": set(), "etccode_set": set()}
    rows = []
    for _, r in cohort.iterrows():
        info = stay_index.get(r["stay_id"], empty)
        row = {
            "stay_id": r["stay_id"],
            "age": r["age"],
            "gender_M": 1 if r["gender"] == "M" else 0,
            "n_meds": info["n_meds"],
            "n_distinct_meds": info["n_distinct"],
        }
        for letter in ALL_ATC1_LETTERS:
            row[f"atc1_{letter}"] = 1 if letter in info["atc1_set"] else 0
        for code in level2_codes:
            row[f"atc2_{code}"] = 1 if code in info["atc2_set"] else 0
        for code in top_etc_codes:
            row[f"etc_{code}"] = 1 if code in info["etccode_set"] else 0
        rows.append(row)

    features = pd.DataFrame(rows)
    feature_report = {
        "n_atc1_flags": len(ALL_ATC1_LETTERS),
        "atc1_letters": ALL_ATC1_LETTERS,
        "n_atc2_flags": len(level2_codes),
        "atc2_codes": level2_codes,
        "top_n_etc_requested": top_n_etc,
        "top_n_etc_actual": len(top_etc_codes),
        "top_etc_codes_selected_on_train_split": top_etc_codes,
        "n_aggregate_features": 2,
        "n_demographic_features": 2,
        "total_feature_dim": len([c for c in features.columns if c != "stay_id"]),
    }
    return features, feature_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", default=DEFAULT_COHORT_PATH)
    parser.add_argument("--atc-cache", default=DEFAULT_ATC_CACHE_PATH)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--feature-report", default=DEFAULT_FEATURE_REPORT_PATH)
    parser.add_argument("--top-n-etc", type=int, default=30)
    args = parser.parse_args()

    cohort = pd.read_parquet(args.cohort)
    with open(args.atc_cache, "r", encoding="utf-8") as f:
        atc_cache = json.load(f)
    medrecon = load_medrecon()
    medrecon = medrecon[medrecon["stay_id"].isin(cohort["stay_id"])]

    # The ETC feature vocabulary is prevalence-ranked, so it must be selected on
    # training patients only. Uses the one shared split definition (splits.py)
    # rather than a second local copy -- see that module's docstring for why.
    train_subject_ids, _val_ids, _test_ids = patient_level_split(cohort)

    features, feature_report = build_features(cohort, medrecon, atc_cache, train_subject_ids, top_n_etc=args.top_n_etc)
    feature_report["etc_vocabulary_selected_on"] = "train split only (splits.patient_level_split)"
    feature_report["n_train_subjects_used_for_etc_selection"] = len(train_subject_ids)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    features.to_parquet(args.output, index=False)
    os.makedirs(os.path.dirname(args.feature_report), exist_ok=True)
    with open(args.feature_report, "w", encoding="utf-8") as f:
        json.dump(feature_report, f, indent=2)

    print(json.dumps(feature_report, indent=2))
    print(f"\nfeatures written to {args.output} ({len(features)} rows, {feature_report['total_feature_dim']} feature columns)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
