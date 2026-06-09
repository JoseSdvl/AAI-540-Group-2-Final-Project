"""Deterministic feature engineering on top of the curated encounter frame.

Everything here is **stateless** transforms that depend only on the row in
question (or already-computed prior-utilisation columns). Stateful pieces
(one-hot encoder fit, scaler fit) live in :func:`build_preprocessor` which
returns a scikit-learn ``ColumnTransformer`` so they can be fit on train and
applied identically at inference time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from readmit.config import FEATURES, PRIMARY_DX_COL


# ---------------------------------------------------------------------------
# Bucketing / derived columns
# ---------------------------------------------------------------------------

AGE_BANDS = [(0, 64, "<65"), (65, 69, "65-69"), (70, 74, "70-74"),
             (75, 79, "75-79"), (80, 84, "80-84"), (85, 200, "85+")]

LOS_BUCKETS = [(1, 1, "1d"), (2, 3, "2-3d"), (4, 7, "4-7d"), (8, 999, "8+d")]


def _bucket(value: float, buckets: list[tuple[int, int, str]], default: str = "unknown") -> str:
    for lo, hi, label in buckets:
        if lo <= value <= hi:
            return label
    return default


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``age_band``, ``los_bucket``, ``primary_dx_chapter``.

    ``primary_dx_chapter`` is just an alias today (the ingest layer already
    groups ICD-10 codes into clinical chapters), but keeping the column name
    distinct from the raw ``primary_diagnosis`` makes the contract explicit
    and survives a future change to ICD-10-CM ingest.
    """
    out = df.copy()
    out["age_band"] = out["age"].apply(lambda v: _bucket(int(v), AGE_BANDS))
    out["los_bucket"] = out["length_of_stay"].apply(lambda v: _bucket(int(v), LOS_BUCKETS))
    out["primary_dx_chapter"] = out[PRIMARY_DX_COL].astype(str)
    return out


# ---------------------------------------------------------------------------
# sklearn preprocessor (used by both baseline LR and XGBoost wrapper)
# ---------------------------------------------------------------------------

def build_preprocessor(scale_numeric: bool = False) -> ColumnTransformer:
    """Return an unfitted ColumnTransformer for the declared feature set.

    Parameters
    ----------
    scale_numeric:
        Whether to standard-scale numeric features. Set to ``True`` for the
        logistic-regression baseline; XGBoost is scale-invariant so we leave
        it off there to keep the trained artifact slightly smaller and
        explanations more interpretable.
    """
    numeric_steps: list[tuple[str, object]] = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))
    numeric_pipeline = Pipeline(numeric_steps)

    categorical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=10),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, FEATURES.numeric),
            ("cat", categorical_pipeline, FEATURES.categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def winsorize_utilization(df: pd.DataFrame, p: float = 0.99) -> pd.DataFrame:
    """Clip extreme right tails on prior-utilisation counts.

    Real CMS data has a handful of beneficiaries with implausibly high
    utilization that act as leverage points. Clipping at the 99th percentile
    is a standard fix that keeps the model from over-weighting outliers
    without dropping any rows.
    """
    out = df.copy()
    util_cols = ["prior_inpatient_90d", "prior_ed_90d", "prior_outpatient_90d"]
    for col in util_cols:
        if col in out.columns:
            cap = np.nanpercentile(out[col], p * 100)
            out[col] = out[col].clip(upper=cap)
    return out
