"""Programmatic 30-day readmission labelling + prior-utilisation features.

Both pieces live together because they share the same per-patient encounter
ordering: deriving them in a single pass avoids re-sorting and guarantees the
prior-utilisation features only reflect *past* events (no future leakage).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from readmit.config import (
    ADMIT_DATE_COL,
    DISCHARGE_DATE_COL,
    LABEL_COL,
    PATIENT_ID_COL,
)

READMISSION_WINDOW_DAYS = 30
PRIOR_WINDOW_DAYS = 90


def attach_labels_and_priors(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with ``readmitted_30d`` and recomputed priors.

    The label is 1 if *any* subsequent inpatient admission for the same
    beneficiary occurs within 30 days of discharge from the index admission,
    else 0. The final admission of every patient is always labelled 0 because
    there is no future encounter to define readmission for it — which matches
    the operational definition used by CMS HRRP.

    ``prior_inpatient_90d`` is overwritten with the count of *prior* inpatient
    admissions whose discharge fell in the 90 days before the current
    admission. ``prior_ed_90d`` and ``prior_outpatient_90d`` are left as
    ingested (they come from outpatient/ED files in the real CMS extract).
    """
    if df.empty:
        out = df.copy()
        out[LABEL_COL] = pd.Series(dtype=np.int8)
        return out

    out = df.sort_values([PATIENT_ID_COL, ADMIT_DATE_COL]).reset_index(drop=True).copy()

    # ---- 30-day readmission label -----------------------------------------
    next_admit = out.groupby(PATIENT_ID_COL)[ADMIT_DATE_COL].shift(-1)
    gap = (next_admit - out[DISCHARGE_DATE_COL]).dt.days
    out[LABEL_COL] = ((gap >= 0) & (gap <= READMISSION_WINDOW_DAYS)).astype(np.int8)

    # ---- Prior inpatient in 90 days (no leakage: strictly past) -----------
    out["prior_inpatient_90d"] = _rolling_prior_count(
        out, group_col=PATIENT_ID_COL, time_col=ADMIT_DATE_COL,
        window_days=PRIOR_WINDOW_DAYS,
    )
    return out


def _rolling_prior_count(
    df: pd.DataFrame, group_col: str, time_col: str, window_days: int
) -> np.ndarray:
    """For each row, count earlier rows in the same group within ``window_days``."""
    counts = np.zeros(len(df), dtype=np.int32)
    # Iterate per group; groups are typically small (<=10 encounters/patient).
    for _, idx in df.groupby(group_col, sort=False).indices.items():
        times = df[time_col].to_numpy()[idx]
        for i, t in enumerate(times):
            lo = t - np.timedelta64(window_days, "D")
            # earlier rows only (j < i) AND within window
            window_mask = (times[:i] >= lo) & (times[:i] < t)
            counts[idx[i]] = int(window_mask.sum())
    return counts
