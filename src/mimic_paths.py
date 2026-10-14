"""
M1's MIMIC-IV-ED / MIMIC-IV path resolver.

Extends the pattern established in `module 4/src/helpers.py::default_mimic_root`
(the module-2 directory has been spelled both `module 2` and `module2` across
this project's history; both are probed). M1 never copies MIMIC data of its
own -- it reads directly from module2's vendored tree, same as M4.

M1 needs three tables `module 4/src/helpers.py` does not use:
  - ed/medrecon.csv.gz      (medications, for the ATC crosswalk)
  - hosp diagnoses_icd      (chronic-condition labels)
  - hosp patients           (anchor_age, for demographics)

The gz release is used throughout (`mimic-iv-ed-2.2/` and `mimic-iv-clinical/`)
rather than the uncompressed `mimic4_full/` copy -- both are the same data
(row counts verified identical on 2026-09-11: edstays 425,088 rows, hosp
diagnoses_icd 6,364,489 rows in both), and using one canonical copy avoids a
second silent-drift risk of the two ever being regenerated out of sync.
"""
from __future__ import annotations

import functools
import os

import pandas as pd


def default_mimic_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(here, "..", ".."))
    candidates = [
        os.path.join(project_root, "module 2", "mimic4ed-benchmark-main"),
        os.path.join(project_root, "module2", "mimic4ed-benchmark-main"),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return candidates[0]


@functools.lru_cache(maxsize=1)
def load_medrecon(mimic_root: str | None = None) -> pd.DataFrame:
    """ed/medrecon.csv.gz -- subject_id, stay_id, charttime, name, gsn, ndc,
    etc_rn, etccode, etcdescription. No ATC codes are present (verified
    2026-09-11): `ndc` is the field the RxNav crosswalk keys off of; `etccode`
    /`etcdescription` are the documented fallback for NDCs that fail to map.

    `etccode` is read as pandas' nullable "Int64" (not the default float64
    pandas infers from a numeric column containing NaNs) so downstream code
    that stringifies it for feature-column names gets "2747", not "2747.0".
    """
    root = mimic_root or default_mimic_root()
    path = os.path.join(root, "mimic-iv-ed-2.2", "mimic-iv-ed-2.2", "ed", "medrecon.csv.gz")
    return pd.read_csv(path, compression="gzip", dtype={"ndc": "string", "gsn": "string", "etccode": "Int64"})


@functools.lru_cache(maxsize=1)
def load_edstays(mimic_root: str | None = None) -> pd.DataFrame:
    root = mimic_root or default_mimic_root()
    path = os.path.join(root, "mimic-iv-ed-2.2", "mimic-iv-ed-2.2", "ed", "edstays.csv.gz")
    return pd.read_csv(path, compression="gzip")


@functools.lru_cache(maxsize=1)
def load_diagnoses_icd(mimic_root: str | None = None) -> pd.DataFrame:
    """hosp diagnoses_icd -- linked via hadm_id, NOT the ED-encounter diagnosis
    table (`ed/diagnosis.csv.gz`). Per the locked label-source decision, the
    admission-level ICD list is what feeds the CCSR chronic-condition grouper;
    ed/diagnosis.csv.gz captures the acute ED presenting problem, not chronic
    history, and is not used for labels.
    """
    root = mimic_root or default_mimic_root()
    path = os.path.join(root, "mimic-iv-clinical", "diagnoses_icd.csv.gz")
    return pd.read_csv(path, compression="gzip", dtype={"icd_code": "string"})


@functools.lru_cache(maxsize=1)
def load_patients(mimic_root: str | None = None) -> pd.DataFrame:
    root = mimic_root or default_mimic_root()
    path = os.path.join(root, "mimic-iv-clinical", "patients.csv.gz")
    return pd.read_csv(path, compression="gzip")
