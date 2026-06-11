"""Curate the public CMS DE-SynPUF OMOP dataset into an `encounters` frame.

The raw data lives in the AWS Open Data registry (``s3://synpuf-omop/``,
``us-east-1``, OMOP Common Data Model v5.x). It's distributed as one OMOP
table per object (``person``, ``visit_occurrence``, ``condition_occurrence``…)
at three sample sizes: 1k, 100k, and 2.3M persons.

This module performs the join that turns those OMOP tables into the
**encounter-level** frame our model consumes (one row per inpatient
admission, with patient-grouped 90-day prior-utilization counts, primary
diagnosis chapter, and discharge disposition). The output schema matches
``readmit.data.ingest._validate_schema`` so the rest of the pipeline doesn't
care whether the data came from CMS or from the synthetic generator.

The curated frame is written once to the project bucket (``encounters.parquet``)
and re-read on subsequent runs — re-curation is skipped if the file already
exists.
"""

from __future__ import annotations

import logging
from typing import Iterable

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
# Public bucket layout
# ---------------------------------------------------------------------------

CMS_OPEN_BUCKET = "synpuf-omop"
CMS_OPEN_REGION = "us-east-1"

# Only the two compressed-with-stdlib tiers are supported. The 2.3M tier ships
# as ``.lzo`` which needs an external decompressor we don't want in the runtime.
CMS_TIERS: dict[str, dict] = {
    "1k":   {"prefix": "cmsdesynpuf1k/",   "compression": "bz2",  "name_style": "upper_cdm"},
    "100k": {"prefix": "cmsdesynpuf100k/", "compression": "gzip", "name_style": "lower"},
}

# OMOP concept-id vocabulary (just what we need for encounters).
_VISIT_INPATIENT  = 9201
_VISIT_OUTPATIENT = 9202
_VISIT_ER         = 9203

_GENDER_MALE   = 8507
_GENDER_FEMALE = 8532

# discharge_to_concept_id → bucketed disposition. IDs come from the OMOP Visit
# vocabulary; everything not listed (including 0 / "Unknown") falls into "home".
_DISPOSITION_MAP = {
    8536: "home",                 # Home / Self Care
    8546: "home_with_services",   # Hospice
    8676: "home_with_services",   # Hospice / Medical Facility
    8717: "transfer",             # Inpatient Hospital
    8920: "snf_rehab",            # Skilled Nursing Facility
    8863: "snf_rehab",            # Intermediate Medical Care / SNF
    8844: "home_with_services",   # Home Health
    8870: "other",                # Expired
}

# 8 chapters used everywhere downstream — keep ordering stable so the modulo
# bucket below is reproducible across runs / kernels.
_DX_CHAPTERS = [
    "circulatory",
    "respiratory",
    "musculoskeletal",
    "endocrine",
    "renal",
    "neoplasm",
    "infectious",
    "other",
]


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

def curate_cms_synpuf(
    tier: str = "100k",
    curated_s3_uri: str | None = None,
    *,
    sample_n: int | None = None,
    force_recurate: bool = False,
    storage_options_curated: dict | None = None,
) -> pd.DataFrame:
    """Build the encounter-level frame from CMS DE-SynPUF OMOP tables.

    Parameters
    ----------
    tier
        Which sample tier to read from the public bucket (``"1k"`` or ``"100k"``).
    curated_s3_uri
        If given, the curated frame is cached under
        ``{curated_s3_uri}/encounters.parquet``. On subsequent calls the file
        is read back instead of re-running the join.
    sample_n
        Optionally cap the number of inpatient encounters returned (useful for
        a quick demo cell). Sampling is applied *after* curation so the cached
        parquet always contains the full curated set.
    force_recurate
        If True, ignore any cached parquet and rebuild from raw.
    storage_options_curated
        Passed to ``pandas.to_parquet`` / ``read_parquet`` when writing/reading
        the curated file. Use this if the project bucket needs explicit
        credentials.
    """
    if tier not in CMS_TIERS:
        raise ValueError(f"Unknown tier {tier!r} — must be one of {list(CMS_TIERS)}")

    if curated_s3_uri and not force_recurate:
        cached = _try_read_cached(curated_s3_uri, storage_options_curated)
        if cached is not None:
            logger.info("Loaded cached curated encounters (%d rows) from %s",
                        len(cached), curated_s3_uri)
            return _maybe_sample(cached, sample_n)

    logger.info("Curating CMS DE-SynPUF (%s tier) from s3://%s/%s",
                tier, CMS_OPEN_BUCKET, CMS_TIERS[tier]["prefix"])

    persons    = _read_omop_table(tier, "person", [
        "person_id", "gender_concept_id", "year_of_birth",
    ])
    visits     = _read_omop_table(tier, "visit_occurrence", [
        "visit_occurrence_id", "person_id", "visit_concept_id",
        "visit_start_date", "visit_end_date", "discharge_to_concept_id",
    ], parse_dates=["visit_start_date", "visit_end_date"])
    conditions = _read_omop_table(tier, "condition_occurrence", [
        "person_id", "visit_occurrence_id", "condition_concept_id",
        "condition_start_date",
    ], parse_dates=["condition_start_date"])

    encounters = _build_encounters(persons, visits, conditions)

    if curated_s3_uri:
        out = curated_s3_uri.rstrip("/") + "/encounters.parquet"
        logger.info("Writing curated encounters (%d rows) to %s", len(encounters), out)
        # Only forward storage_options when caller actually provided one.
        # Older pandas (SKLearn 1.2-1 container) forwards storage_options=None
        # straight to pyarrow.parquet.write_table(), which rejects the kwarg.
        write_kwargs = {"index": False}
        if storage_options_curated:
            write_kwargs["storage_options"] = storage_options_curated
        encounters.to_parquet(out, **write_kwargs)

    return _maybe_sample(encounters, sample_n)


# ---------------------------------------------------------------------------
# Raw OMOP readers (anonymous S3)
# ---------------------------------------------------------------------------

def _omop_filename(tier: str, table: str) -> str:
    cfg = CMS_TIERS[tier]
    if cfg["name_style"] == "upper_cdm":
        return f"CDM_{table.upper()}.csv.bz2"
    return f"{table}.csv.gz"


def _read_omop_table(
    tier: str,
    table: str,
    columns: Iterable[str],
    parse_dates: Iterable[str] | None = None,
) -> pd.DataFrame:
    uri = f"s3://{CMS_OPEN_BUCKET}/{CMS_TIERS[tier]['prefix']}{_omop_filename(tier, table)}"
    logger.info("Reading OMOP %s from %s", table, uri)
    # ``anon=True`` so we don't need AWS creds for the public bucket; pandas
    # auto-detects gzip/bz2 from the suffix.
    df = pd.read_csv(
        uri,
        storage_options={"anon": True, "client_kwargs": {"region_name": CMS_OPEN_REGION}},
        low_memory=False,
    )
    df.columns = [c.lower() for c in df.columns]
    keep = [c for c in columns if c in df.columns]
    df = df[keep]
    if parse_dates:
        for col in parse_dates:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Join + feature derivation
# ---------------------------------------------------------------------------

def _build_encounters(
    persons: pd.DataFrame,
    visits: pd.DataFrame,
    conditions: pd.DataFrame,
) -> pd.DataFrame:
    # ---- 1. Inpatient visits become the encounter rows -----------------
    inpat = visits.loc[visits["visit_concept_id"] == _VISIT_INPATIENT].copy()
    inpat = inpat.dropna(subset=["visit_start_date", "visit_end_date"])

    # length_of_stay: at least 1 day, capped at 30 to match training-time priors.
    los = (inpat["visit_end_date"] - inpat["visit_start_date"]).dt.days.fillna(0) + 1
    inpat["length_of_stay"] = los.clip(lower=1, upper=30).astype(int)

    # ---- 2. Demographics from person -----------------------------------
    inpat = inpat.merge(persons, on="person_id", how="left")
    inpat["age"] = (
        inpat["visit_start_date"].dt.year - inpat["year_of_birth"].fillna(1940).astype(int)
    ).clip(lower=18, upper=110).astype(int)
    inpat["sex"] = np.where(inpat["gender_concept_id"] == _GENDER_MALE, "M",
                    np.where(inpat["gender_concept_id"] == _GENDER_FEMALE, "F", "F"))

    # ---- 3. Discharge disposition (bucketed) ---------------------------
    inpat["discharge_disposition"] = (
        inpat["discharge_to_concept_id"].map(_DISPOSITION_MAP).fillna("home")
    )

    # ---- 4. Payer type (synthetic dataset has no payer detail in 1k/100k)
    # All DE-SynPUF beneficiaries are Medicare; we mark the FFS vs MA split
    # deterministically from person_id so subgroup analysis still works.
    payer_choices = np.array(["medicare_ffs", "medicare_advantage", "dual_eligible"])
    payer_idx = (inpat["person_id"].astype(np.int64) % 100)
    inpat["payer_type"] = np.where(
        payer_idx < 55, payer_choices[0],
        np.where(payer_idx < 85, payer_choices[1], payer_choices[2]),
    )

    # ---- 5. Prior-utilization windows ----------------------------------
    inpat = _attach_prior_utilization(inpat, visits)

    # ---- 6. Primary diagnosis ------------------------------------------
    inpat = _attach_primary_diagnosis(inpat, conditions)

    # ---- 7. Chronic-condition burden -----------------------------------
    inpat = _attach_chronic_conditions(inpat, conditions)

    # ---- 8. Final column rename + schema ordering ----------------------
    encounters = pd.DataFrame({
        PATIENT_ID_COL:        inpat["person_id"].astype(np.int64),
        ADMIT_DATE_COL:        inpat["visit_start_date"],
        DISCHARGE_DATE_COL:    inpat["visit_end_date"],
        "length_of_stay":      inpat["length_of_stay"],
        PRIMARY_DX_COL:        inpat["primary_dx_chapter"],
        "discharge_disposition": inpat["discharge_disposition"],
        "age":                 inpat["age"],
        "sex":                 inpat["sex"],
        "payer_type":          inpat["payer_type"],
        "n_chronic_conditions": inpat["n_chronic_conditions"],
        "charlson_index":      inpat["charlson_index"],
        "prior_inpatient_90d": inpat["prior_inpatient_90d"],
        "prior_ed_90d":        inpat["prior_ed_90d"],
        "prior_outpatient_90d": inpat["prior_outpatient_90d"],
    })
    encounters = encounters.sort_values([PATIENT_ID_COL, ADMIT_DATE_COL]).reset_index(drop=True)
    return encounters


def _attach_prior_utilization(inpat: pd.DataFrame, visits: pd.DataFrame) -> pd.DataFrame:
    """Count prior inpatient / ER / outpatient visits in the 90 days
    preceding each inpatient admission.

    Vectorised: cross-joins inpatient encounters with the patient's full
    visit history on ``person_id``, filters to the 90-day pre-admission
    window, and pivots a single ``groupby`` to get per-type counts.
    Replaces the previous per-row Python loop (~5 min on 100k tier) with a
    couple of pandas ops (~2-3 s on the same data).
    """
    type_map = {
        _VISIT_INPATIENT:  "prior_inpatient_90d",
        _VISIT_ER:         "prior_ed_90d",
        _VISIT_OUTPATIENT: "prior_outpatient_90d",
    }
    typed = visits.loc[visits["visit_concept_id"].isin(type_map)].copy()
    typed["visit_type"] = typed["visit_concept_id"].map(type_map)
    typed = typed.dropna(subset=["visit_start_date"])[
        ["person_id", "visit_start_date", "visit_type"]
    ].rename(columns={"visit_start_date": "prior_date"})

    out = inpat.copy()
    out["_enc_row"] = np.arange(len(out), dtype=np.int64)

    # Cross-merge encounter rows with the patient's full visit history.
    merged = out[["_enc_row", "person_id", "visit_start_date"]].merge(
        typed, on="person_id", how="left",
    )
    merged = merged.dropna(subset=["prior_date"])

    # Strictly *before* admission, within the 90-day window.
    days_before = (merged["visit_start_date"] - merged["prior_date"]).dt.days
    in_window = merged[(days_before > 0) & (days_before <= 90)]

    # One groupby + unstack gives counts for all three types at once.
    counts = (
        in_window.groupby(["_enc_row", "visit_type"]).size().unstack(fill_value=0)
    )

    for col in type_map.values():
        out[col] = (
            out["_enc_row"].map(counts[col] if col in counts.columns else {})
            .fillna(0).astype(np.int32)
        )

    return out.drop(columns=["_enc_row"])


def _attach_primary_diagnosis(inpat: pd.DataFrame, conditions: pd.DataFrame) -> pd.DataFrame:
    """Pick the earliest condition recorded against each inpatient visit and
    bucket its concept-id deterministically into one of the 8 chapter labels
    the rest of the pipeline understands."""
    cond = conditions.dropna(subset=["visit_occurrence_id", "condition_concept_id"]).copy()
    cond = cond.sort_values(["visit_occurrence_id", "condition_start_date"])
    primary = cond.drop_duplicates("visit_occurrence_id", keep="first")[
        ["visit_occurrence_id", "condition_concept_id"]
    ]
    primary["primary_dx_chapter"] = _chapter_for(primary["condition_concept_id"].to_numpy())

    merged = inpat.merge(primary, on="visit_occurrence_id", how="left")
    merged["primary_dx_chapter"] = merged["primary_dx_chapter"].fillna("other")
    return merged


def _chapter_for(concept_ids: np.ndarray) -> np.ndarray:
    """Stable hash → chapter. We use a modulo bucket rather than a real
    SNOMED→chapter map because the OMOP concept table isn't published in
    this open dataset; the modulo is deterministic across runs."""
    # Skew slightly toward circulatory/respiratory to match the HRRP focus.
    weights = np.array([0.22, 0.18, 0.14, 0.12, 0.08, 0.08, 0.08, 0.10])
    cum = np.cumsum(weights)
    bucket_floats = ((concept_ids.astype(np.int64) * 2654435761) & 0xFFFFFFFF) / 2**32
    idx = np.searchsorted(cum, bucket_floats, side="right")
    idx = np.clip(idx, 0, len(_DX_CHAPTERS) - 1)
    return np.array(_DX_CHAPTERS)[idx]


def _attach_chronic_conditions(inpat: pd.DataFrame, conditions: pd.DataFrame) -> pd.DataFrame:
    """For each inpatient encounter, count distinct condition concepts in the
    365 days preceding admission and approximate a Charlson-style burden.

    Vectorised replacement for the original per-row Python loop. The merge
    creates one row per (encounter, prior condition) pair which is then
    filtered to the 365-day window and reduced via a single
    ``groupby.nunique()`` to get distinct concept counts per encounter.
    Runs in seconds on the 100k tier vs ~10 min for the loop version.
    """
    cond = conditions.dropna(subset=["condition_start_date", "condition_concept_id"])[
        ["person_id", "condition_concept_id", "condition_start_date"]
    ].rename(columns={"condition_start_date": "prior_date"})

    out = inpat.copy()
    out["_enc_row"] = np.arange(len(out), dtype=np.int64)

    merged = out[["_enc_row", "person_id", "visit_start_date"]].merge(
        cond, on="person_id", how="left",
    )
    merged = merged.dropna(subset=["prior_date"])

    days_before = (merged["visit_start_date"] - merged["prior_date"]).dt.days
    in_window = merged[(days_before > 0) & (days_before <= 365)]

    n_chronic_series = (
        in_window.groupby("_enc_row")["condition_concept_id"].nunique()
    )
    n_chronic = (
        out["_enc_row"].map(n_chronic_series).fillna(0).astype(np.int32).to_numpy()
    )
    out["n_chronic_conditions"] = n_chronic
    out["charlson_index"] = np.clip(np.round(n_chronic * 0.6).astype(np.int32), 0, 15)
    return out.drop(columns=["_enc_row"])


# ---------------------------------------------------------------------------
# Caching helpers
# ---------------------------------------------------------------------------

def _try_read_cached(curated_s3_uri: str, storage_options: dict | None) -> pd.DataFrame | None:
    uri = curated_s3_uri.rstrip("/") + "/encounters.parquet"
    # Only forward storage_options when caller actually provided one — older
    # pandas (SKLearn 1.2-1 container) forwards storage_options=None straight
    # to pyarrow.parquet.read_table(), which rejects the kwarg.
    read_kwargs = {"storage_options": storage_options} if storage_options else {}
    try:
        return pd.read_parquet(uri, **read_kwargs)
    except (FileNotFoundError, OSError) as exc:  # pragma: no cover - depends on s3fs runtime
        logger.info("No cached encounters at %s (%s); will re-curate.", uri, exc)
        return None


def _maybe_sample(df: pd.DataFrame, sample_n: int | None) -> pd.DataFrame:
    if sample_n is None or sample_n >= len(df):
        return df
    # Stratified-by-patient down-sample so the split helper still works.
    chosen = (
        df[PATIENT_ID_COL]
        .drop_duplicates()
        .sample(n=min(sample_n, df[PATIENT_ID_COL].nunique()), random_state=42)
    )
    return df[df[PATIENT_ID_COL].isin(chosen)].reset_index(drop=True)
