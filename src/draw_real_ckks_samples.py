"""
M1 -- Draw and FREEZE the multi-seed real-CKKS validation samples (closure guide 3.5).

Executes docs/real_ckks_multiseed_protocol.md. The output files are committed BEFORE the execution script runs.

Scope and provenance
  * Population:   the TEST split (splits.patient_level_split, SPLIT_SEED=42). Used only to measure fidelity of
                  the simulation against real ciphertexts; no threshold, anchor or coefficient is selected here.
  * Stratum:      plaintext predicted class = (plaintext margin >= the seed's frozen validation-tuned margin
                  threshold). Depends on neither the polynomial coefficients nor gamma.
  * Allocation:   12 predicted-positive + 13 predicted-negative per seed, without replacement.
  * RNG:          numpy.random.default_rng([SAMPLING_SUBSEED, model_seed]) -- separate from the model seed.
  * Labels:       one label (hypertension) for every seed. No outcome column is ever read.

Writes data/m1_real_ckks_samples/seed{S}.csv and data/m1_real_ckks_samples/manifest.json.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
import xgboost as xgb

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from splits import MODEL_SEEDS, patient_level_split  # noqa: E402

COHORT_PATH = os.path.join(_HERE, "..", "data", "m1_cohort.parquet")
FEATURES_PATH = os.path.join(_HERE, "..", "data", "m1_features.parquet")
MODEL_DIR = os.path.join(_HERE, "..", "data", "models")
BASELINE_PATH = os.path.join(_HERE, "..", "results", "m1_plaintext_baseline.json")
OUT_DIR = os.path.join(_HERE, "..", "data", "m1_real_ckks_samples")

# --- fixed by docs/real_ckks_multiseed_protocol.md ------------------------------------------------
LABEL = "label_hypertension_uncomplicated"
SAMPLING_SUBSEED = 20260921
N_POSITIVE, N_NEGATIVE = 12, 13
# ---------------------------------------------------------------------------------------------------


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def main() -> int:
    cohort = pd.read_parquet(COHORT_PATH)
    features = pd.read_parquet(FEATURES_PATH)
    # NOTE: only the id columns of the cohort are merged; no label column is read into the frame.
    merged = features.merge(cohort[["stay_id", "subject_id"]], on="stay_id", how="inner")
    feature_cols = [c for c in features.columns if c != "stay_id"]
    _, _, test_ids = patient_level_split(merged)  # same subject set as the other evaluators (id columns only)
    test_df = merged[merged["subject_id"].isin(test_ids)].reset_index(drop=True)

    with open(BASELINE_PATH, "r", encoding="utf-8") as f:
        baseline = json.load(f)
    label_cols = [c for c in cohort.columns if c.startswith("label_")]
    label_index = label_cols.index(LABEL)

    os.makedirs(OUT_DIR, exist_ok=True)
    manifest = {"protocol": "docs/real_ckks_multiseed_protocol.md", "label": LABEL,
                "sampling_subseed": SAMPLING_SUBSEED, "n_positive": N_POSITIVE, "n_negative": N_NEGATIVE,
                "population": "TEST split (fidelity measurement only; nothing selected from it)",
                "stratum_rule": "plaintext margin >= frozen validation-tuned margin threshold", "seeds": {}}

    for seed in MODEL_SEEDS:
        thr = next(r for r in baseline["per_seed_results"] if r["seed"] == seed)["per_label"][label_index]["tuned_threshold_margin"]
        clf = xgb.XGBClassifier()
        clf.load_model(os.path.join(MODEL_DIR, f"xgb_float_seed{seed}_{LABEL}.json"))
        booster = clf.get_booster()
        booster.feature_names = feature_cols
        margin = booster.predict(xgb.DMatrix(test_df[feature_cols], feature_names=feature_cols), output_margin=True)
        pred_pos = margin >= thr

        rng = np.random.default_rng([SAMPLING_SUBSEED, seed])
        pos_pool, neg_pool = np.flatnonzero(pred_pos), np.flatnonzero(~pred_pos)
        assert len(pos_pool) >= N_POSITIVE and len(neg_pool) >= N_NEGATIVE, f"seed {seed}: stratum too small"
        pos_pick = np.sort(rng.choice(pos_pool, size=N_POSITIVE, replace=False))
        neg_pick = np.sort(rng.choice(neg_pool, size=N_NEGATIVE, replace=False))
        idx = np.concatenate([pos_pick, neg_pick])

        out = pd.DataFrame({
            "stay_id": test_df["stay_id"].to_numpy()[idx],
            "subject_id": test_df["subject_id"].to_numpy()[idx],
            "test_row_index": idx,
            "plaintext_margin": margin[idx],
            "stratum": ["predicted_positive"] * N_POSITIVE + ["predicted_negative"] * N_NEGATIVE,
        })
        path = os.path.join(OUT_DIR, f"seed{seed}.csv")
        out.to_csv(path, index=False)
        manifest["seeds"][str(seed)] = {
            "file": os.path.basename(path), "sha256": sha256(path), "frozen_margin_threshold": thr,
            "n_test_rows": int(len(test_df)), "n_predicted_positive_in_test": int(pred_pos.sum()),
            "n_sampled": int(len(out)), "distinct_subjects": int(out["subject_id"].nunique()),
        }
        print(f"seed {seed}: thr={thr:.4f}  test pos-pool={len(pos_pool)} neg-pool={len(neg_pool)}  "
              f"sampled {len(out)} ({out['subject_id'].nunique()} distinct subjects)")

    with open(os.path.join(OUT_DIR, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
