"""
The single canonical patient-level split for M1. One definition, used by every
script that needs one.

Why this file exists: `features.py` (which selects the top-N ETC codes by
prevalence) and `train_plaintext.py` (which fits and evaluates) originally
each rolled their own split. A 2026-09-11 audit confirmed they happened to
agree -- zero subjects leaked, because both used seed 42 over an identically
ordered subject array -- but that was a coincidence of two independent
implementations, not a guarantee. Changing the seed in one file would have
silently started selecting features using test-set patients. There is now one
split function and one set of constants.

Three-way, split on `subject_id` (never `stay_id` or `hadm_id`): a patient
contributes ~1.95 ED stays on average, so a stay-level split would put the
same patient's records on both sides.

  train (60%) -- fit the model; select the ETC feature vocabulary
  val   (20%) -- tune the per-label decision threshold; NEVER used for fitting
  test  (20%) -- reported numbers only, touched once

The validation split exists specifically because the default 0.5 decision
threshold was found to be degenerate on this cohort (predicted positive rate
1.0% against 22.9% prevalence for fluid/electrolyte disorders). Tuning that
threshold on test would be optimistically biased, so it is tuned here instead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SPLIT_SEED = 42
TRAIN_FRACTION = 0.6
VAL_FRACTION = 0.2
# test takes the remainder (0.2) -- stated explicitly rather than derived, so a
# change to either constant above is an obvious edit rather than a silent shift.
TEST_FRACTION = 0.2

# The five fixed seeds used project-wide (M3/M4/M5 all use these), varying only
# model-fitting stochasticity -- never the split, which stays fixed at SPLIT_SEED.
MODEL_SEEDS = (42, 123, 7, 456, 789)


def patient_level_split(
    df: pd.DataFrame,
    seed: int = SPLIT_SEED,
    train_fraction: float = TRAIN_FRACTION,
    val_fraction: float = VAL_FRACTION,
) -> tuple[set[int], set[int], set[int]]:
    """Return (train_subject_ids, val_subject_ids, test_subject_ids).

    `df` must carry a `subject_id` column. The split is a function of the
    distinct subject ids only, so it is stable regardless of how many stays
    each caller's frame happens to contain -- `features.py` and
    `train_plaintext.py` get the same partition even though one is passed the
    cohort and the other a feature/label join of it.
    """
    if "subject_id" not in df.columns:
        raise KeyError("patient_level_split requires a 'subject_id' column")

    subject_ids = np.sort(df["subject_id"].unique())  # sort first: .unique() is
    # order-of-appearance, so sorting makes the partition independent of row order
    rng = np.random.default_rng(seed)
    rng.shuffle(subject_ids)

    n = len(subject_ids)
    n_train = int(n * train_fraction)
    n_val = int(n * val_fraction)

    return (
        set(subject_ids[:n_train]),
        set(subject_ids[n_train : n_train + n_val]),
        set(subject_ids[n_train + n_val :]),
    )
