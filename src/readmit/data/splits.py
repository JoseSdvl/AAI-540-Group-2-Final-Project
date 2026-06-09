"""Patient-grouped, time-aware 40/30/30 splitter."""

from __future__ import annotations

import numpy as np
import pandas as pd

from readmit.config import (
    ADMIT_DATE_COL,
    PATIENT_ID_COL,
    RANDOM_SEED,
    TEST_FRAC,
    TRAIN_FRAC,
    VAL_FRAC,
)


def split_train_test_val(
    df: pd.DataFrame,
    train_frac: float = TRAIN_FRAC,
    test_frac: float = TEST_FRAC,
    val_frac: float = VAL_FRAC,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split ``df`` into (train, test, val) at the required 40/30/30 ratio.

    Splitting is done at the **patient** level — a beneficiary's encounters
    never straddle two folds — and ordered by first-admission date so the
    train fold is chronologically earliest, mimicking deployment where the
    model is trained on history and scored on future patients.
    """
    if abs(train_frac + test_frac + val_frac - 1.0) > 1e-9:
        raise ValueError("Split fractions must sum to 1.0")

    # Order patients by their first admission date, then break ties with a
    # seeded shuffle so adjacent dates don't all land in one fold.
    first_admit = (
        df.groupby(PATIENT_ID_COL)[ADMIT_DATE_COL].min().sort_values()
    )
    rng = np.random.default_rng(seed)
    ordered_patients = first_admit.index.to_numpy()

    n = len(ordered_patients)
    if n < 3:
        raise ValueError("Need at least 3 patients to produce a 40/30/30 split")

    n_train = int(np.floor(n * train_frac))
    n_test = int(np.floor(n * test_frac))
    # Val gets whatever is left so we don't drop rows to rounding.
    n_val = n - n_train - n_test

    train_pat = ordered_patients[:n_train]
    test_pat = ordered_patients[n_train : n_train + n_test]
    val_pat = ordered_patients[n_train + n_test :]

    # Lightly shuffle within each fold for reproducibility / less ordering bias.
    rng.shuffle(train_pat)
    rng.shuffle(test_pat)
    rng.shuffle(val_pat)

    train_df = df[df[PATIENT_ID_COL].isin(train_pat)].copy()
    test_df = df[df[PATIENT_ID_COL].isin(test_pat)].copy()
    val_df = df[df[PATIENT_ID_COL].isin(val_pat)].copy()

    assert len(set(train_pat) & set(test_pat)) == 0
    assert len(set(train_pat) & set(val_pat)) == 0
    assert len(set(test_pat) & set(val_pat)) == 0
    assert n_val > 0, "Validation fold ended up empty"

    return train_df, test_df, val_df
