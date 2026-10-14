#!/usr/bin/env python3
"""Aggregate check of the measurement lattice of each input (training split): share of non-missing values that are
multiples of candidate steps, and the training median used for imputation. No rows printed (CLAUDE.md 1.2)."""
import json, os, sys
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import m1c_cohort as mc
from splits import patient_level_split

df, _ = mc.build_cohort()
train_ids, _, _ = patient_level_split(df)
tr = df[df["subject_id"].isin(train_ids)]
steps = {"triage_temperature": [1/18, 1/36, 0.1, 0.05], "triage_heartrate": [1, 0.5], "triage_resprate": [1, 0.5],
         "triage_o2sat": [1, 0.5], "triage_sbp": [1, 0.5], "triage_dbp": [1, 0.5], "triage_pain": [1, 0.5], "age": [1]}
out = {}
for c in mc.FEATURE_COLS:
    v = tr[c].dropna().to_numpy(dtype=float)
    med = float(np.median(v))
    out[c] = {"n": int(len(v)), "median": med,
              "share_multiple_of": {str(round(s, 5)): float(np.mean(np.abs(v / s - np.round(v / s)) <= 1e-6)) for s in steps[c]}}
    print(c, out[c]["median"], out[c]["share_multiple_of"], flush=True)
with open(os.path.join(_HERE, "..", "results", "m1c_lattice_check.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
