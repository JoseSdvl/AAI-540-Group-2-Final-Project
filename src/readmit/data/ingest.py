"""Data ingest for CMS DE-SynPUF.

Three modes are supported:

* ``source="cms-open"`` — build the encounter frame from the public AWS Open
  Data Registry bucket ``s3://synpuf-omop/`` (OMOP CDM v5.x) and cache it as
  ``encounters.parquet`` under ``s3_uri``. This is the recommended mode for
  end-to-end SageMaker runs.
* ``source="s3"``  — read a pre-curated ``encounters.parquet`` from a project
  S3 prefix (e.g. one a teammate already wrote with ``source="cms-open"``).
* ``source="synthetic"`` — generate a deterministic synthetic claims dataset
  that exercises every column the rest of the pipeline expects. This is what
  CI uses and what local developers can run without S3 credentials.

The synthetic generator is intentionally simple: it produces patient-level
inpatient encounters with a coherent readmission rate (~17%, matching the CMS
HRRP national heart-failure benchmark) so downstream training and evaluation
behave realistically.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from readmit.config import (
    ADMIT_DATE_COL,
    DISCHARGE_DATE_COL,
    PATIENT_ID_COL,
    PRIMARY_DX_COL,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

def load_encounters(
    source: str = "synthetic",
    s3_uri: str | None = None,
    n_patients: int = 10_000,
    seed: int = 42,
    cms_tier: str = "100k",
    force_recurate: bool = False,
) -> pd.DataFrame:
    """Return a normalised encounter-level DataFrame.

    The returned frame has *one row per inpatient admission* and the columns:

        beneficiary_id, admission_date, discharge_date, age, sex,
        primary_diagnosis, discharge_disposition, payer_type, length_of_stay,
        n_chronic_conditions, charlson_index, prior_inpatient_90d,
        prior_ed_90d, prior_outpatient_90d.

    Labels are **not** attached here — see :mod:`readmit.data.labeling`.
    """
    if source == "cms-open":
        if not s3_uri:
            raise ValueError(
                "source='cms-open' requires s3_uri for the project bucket "
                "(curated parquet is cached at {s3_uri}/encounters.parquet)"
            )
        # Import lazily so the synthetic path keeps working even without s3fs.
        from readmit.data.curate_cms import curate_cms_synpuf
        df = curate_cms_synpuf(
            tier=cms_tier,
            curated_s3_uri=s3_uri,
            sample_n=n_patients if n_patients else None,
            force_recurate=force_recurate,
        )
        _validate_schema(df)
        return df
    if source == "s3":
        if not s3_uri:
            raise ValueError("source='s3' requires s3_uri (e.g. s3://bucket/prefix/)")
        return _load_from_s3(s3_uri)
    if source == "synthetic":
        return _generate_synthetic(n_patients=n_patients, seed=seed)
    raise ValueError(f"Unknown source: {source!r}")


# ---------------------------------------------------------------------------
# S3 loader (CMS DE-SynPUF in OMOP CDM Parquet)
# ---------------------------------------------------------------------------

def _load_from_s3(s3_uri: str) -> pd.DataFrame:
    """Read curated inpatient encounters from S3.

    The expected layout under ``s3_uri`` is OMOP CDM Parquet exports
    (``visit_occurrence``, ``person``, ``condition_occurrence``) that have
    already been pre-joined into a single ``encounters.parquet`` file by the
    Glue/Processing step. If the curated file does not exist, this function
    raises rather than falling back silently — silent fallbacks are how
    drift bugs ship to production.
    """
    uri = s3_uri.rstrip("/") + "/encounters.parquet"
    logger.info("Reading curated encounters from %s", uri)
    df = pd.read_parquet(uri)
    _validate_schema(df)
    return df


# ---------------------------------------------------------------------------
# Synthetic generator (CI + local dev)
# ---------------------------------------------------------------------------

_DX_CHAPTERS = [
    "circulatory",      # HRRP target: heart failure
    "respiratory",      # HRRP target: pneumonia / COPD
    "musculoskeletal",  # HRRP target: joint replacement
    "endocrine",        # diabetes, etc.
    "renal",
    "neoplasm",
    "infectious",
    "other",
]
_DISPOSITIONS = ["home", "home_with_services", "snf_rehab", "transfer", "other"]
_PAYERS = ["medicare_ffs", "medicare_advantage", "dual_eligible"]
_SEXES = ["M", "F"]


def _generate_synthetic(n_patients: int, seed: int) -> pd.DataFrame:
    """Deterministically generate inpatient encounters."""
    rng = np.random.default_rng(seed)

    # Each patient has 1..5 encounters in a 2-year window.
    encounters_per = rng.integers(low=1, high=6, size=n_patients)
    n_rows = int(encounters_per.sum())

    patient_ids = np.repeat(np.arange(n_patients, dtype=np.int64), encounters_per)

    # Demographics are patient-level, so broadcast over encounters.
    pat_age = rng.integers(low=65, high=96, size=n_patients)
    pat_sex = rng.choice(_SEXES, size=n_patients)
    pat_payer = rng.choice(_PAYERS, size=n_patients, p=[0.55, 0.30, 0.15])

    ages = pat_age[patient_ids]
    sexes = pat_sex[patient_ids]
    payers = pat_payer[patient_ids]

    # Admission dates uniformly in [2009-01-01, 2010-12-31] then sorted per pt.
    base = datetime(2009, 1, 1)
    day_offsets = rng.integers(low=0, high=730, size=n_rows)
    df = pd.DataFrame(
        {
            PATIENT_ID_COL: patient_ids,
            ADMIT_DATE_COL: [base + timedelta(days=int(d)) for d in day_offsets],
        }
    )
    df = df.sort_values([PATIENT_ID_COL, ADMIT_DATE_COL]).reset_index(drop=True)

    # Length of stay: skewed; most 1-7 days, long tail to 30.
    los = np.clip(rng.gamma(shape=2.0, scale=2.0, size=len(df)).round(), 1, 30).astype(int)
    df["length_of_stay"] = los
    df[DISCHARGE_DATE_COL] = df[ADMIT_DATE_COL] + pd.to_timedelta(los, unit="D")

    # Clinical / demographic features.
    df[PRIMARY_DX_COL] = rng.choice(
        _DX_CHAPTERS,
        size=len(df),
        p=[0.22, 0.18, 0.14, 0.12, 0.08, 0.08, 0.08, 0.10],
    )
    df["discharge_disposition"] = rng.choice(
        _DISPOSITIONS, size=len(df), p=[0.55, 0.20, 0.15, 0.05, 0.05]
    )
    df["age"] = ages
    df["sex"] = sexes
    df["payer_type"] = payers
    df["n_chronic_conditions"] = rng.integers(low=0, high=12, size=len(df))
    df["charlson_index"] = np.clip(
        (df["n_chronic_conditions"] * 0.6 + rng.normal(0, 1.0, size=len(df))).round(),
        0, 15,
    ).astype(int)

    # Prior utilization windows are deterministic from earlier rows below
    # (computed in labeling for consistency, but we add 0 placeholders here).
    df["prior_inpatient_90d"] = 0
    df["prior_ed_90d"] = rng.integers(low=0, high=4, size=len(df))
    df["prior_outpatient_90d"] = rng.integers(low=0, high=10, size=len(df))

    _validate_schema(df)
    return df


# ---------------------------------------------------------------------------
# Schema guard
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = {
    PATIENT_ID_COL,
    ADMIT_DATE_COL,
    DISCHARGE_DATE_COL,
    "length_of_stay",
    PRIMARY_DX_COL,
    "discharge_disposition",
    "age",
    "sex",
    "payer_type",
    "n_chronic_conditions",
    "charlson_index",
    "prior_inpatient_90d",
    "prior_ed_90d",
    "prior_outpatient_90d",
}


def _validate_schema(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Encounter frame missing required columns: {sorted(missing)}")
    if df[PATIENT_ID_COL].isna().any():
        raise ValueError("beneficiary_id contains nulls")
    if (df["length_of_stay"] < 1).any():
        raise ValueError("length_of_stay must be >= 1 day")
