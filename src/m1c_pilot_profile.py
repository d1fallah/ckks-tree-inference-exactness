#!/usr/bin/env python3
"""
Validity-gate pilot, step 1 (data realism and scenario consistency), M1 continuous-feature task.

Aggregate output only: counts, missingness, ranges and quantiles of each input, outcome prevalence by
split. No row is printed or written (CLAUDE.md section 1.2). Writes results/m1c_pilot_profile.json.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import m1c_cohort as mc  # noqa: E402
from splits import SPLIT_SEED, patient_level_split  # noqa: E402

OUT = os.path.join(_HERE, "..", "results", "m1c_pilot_profile.json")


def main() -> int:
    df, flow = mc.build_cohort()
    train_ids, val_ids, test_ids = patient_level_split(df)
    assert not (train_ids & val_ids) and not (train_ids & test_ids) and not (val_ids & test_ids)
    part = {"train": df["subject_id"].isin(train_ids), "val": df["subject_id"].isin(val_ids),
            "test": df["subject_id"].isin(test_ids)}
    out = {
        "gate": "GATES.md Question gate, M1 continuous-feature task (signed 2026-10-11); Validity pilot step 1",
        "outcome_definition": "M2 critical outcome: in-hospital death or ICU stay starting within "
                              f"{mc.ICU_TRANSFER_HOURS} h of ED exit (vendored helpers.py functions)",
        "inputs": mc.FEATURE_COLS,
        "input_source": "ed/triage.csv.gz (+ patients.anchor_age/anchor_year for age); ed/vitalsign is not read",
        "flow": flow,
        "n_stays": int(len(df)), "n_subjects": int(df["subject_id"].nunique()),
        "split_seed": SPLIT_SEED,
        "split": {k: {"stays": int(m.sum()), "subjects": int(df.loc[m, "subject_id"].nunique()),
                      "outcome_prevalence": float(df.loc[m, mc.OUTCOME_COL].mean())} for k, m in part.items()},
        "features": {},
    }
    for c in mc.FEATURE_COLS:
        s = df[c]
        v = s.dropna().to_numpy(dtype=float)
        q = np.quantile(v, [0.001, 0.01, 0.25, 0.5, 0.75, 0.99, 0.999]) if len(v) else []
        out["features"][c] = {
            "missing_rate": float(s.isna().mean()),
            "n_distinct": int(s.nunique()),
            "min": float(v.min()), "max": float(v.max()),
            "quantiles_0.1_1_25_50_75_99_99.9": [float(x) for x in q],
            "is_binary": bool(s.nunique() <= 2),
            "missing_rate_by_outcome": {
                "outcome0": float(s[df[mc.OUTCOME_COL] == 0].isna().mean()),
                "outcome1": float(s[df[mc.OUTCOME_COL] == 1].isna().mean()),
            },
        }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: out[k] for k in ("flow", "n_stays", "n_subjects", "split")}, indent=1))
    for c, d in out["features"].items():
        print(f"{c:22s} missing={d['missing_rate']:.3f} distinct={d['n_distinct']:4d} "
              f"range=[{d['min']:.2f},{d['max']:.2f}] median={d['quantiles_0.1_1_25_50_75_99_99.9'][3]:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
