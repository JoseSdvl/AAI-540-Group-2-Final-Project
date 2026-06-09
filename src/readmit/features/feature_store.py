"""SageMaker Feature Store integration for the readmission model.

Why this module exists
----------------------
Training, batch scoring, and real-time inference must see *identical* feature
definitions and identical values for the same `beneficiary_id` + `event_time`.
SageMaker Feature Store is the system of record for that contract:

* **Online store** — low-latency lookup keyed by `beneficiary_id`, used at
  inference time when the clinical decision-support UI needs the latest
  features for a patient.
* **Offline store** — partitioned Parquet in S3 + a Glue / Athena table, used
  for training-set assembly, backfills, and audits.

The helpers below are small wrappers around `sagemaker.feature_store` that
encode our project conventions:

* `beneficiary_id` (string)  -> record identifier
* `event_time`     (float)   -> seconds since epoch (Feature Store requirement)
* Numeric columns become `Fractional`, integer columns `Integral`, everything
  else `String`. Booleans are cast to `Integral` (0/1).

Public API
----------
* :func:`prepare_records`              - DataFrame -> Feature-Store-ready frame
* :func:`build_feature_definitions`    - DataFrame -> list[FeatureDefinition]
* :func:`create_or_update_feature_group` - idempotent FeatureGroup create
* :func:`ingest_features`              - parallel ingest of a DataFrame
* :func:`get_latest_features`          - online lookup for one or more IDs
* :func:`query_offline_features`       - Athena query against the offline store
"""

from __future__ import annotations

import logging
import time
from typing import Iterable

import pandas as pd

from readmit.config import (
    FEATURE_GROUP_NAME,
    FEATURE_STORE_S3_PREFIX,
    PATIENT_ID_COL,
)

logger = logging.getLogger(__name__)

RECORD_ID_COL = PATIENT_ID_COL
EVENT_TIME_COL = "event_time"


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

def prepare_records(
    df: pd.DataFrame,
    record_id_col: str = RECORD_ID_COL,
    event_time_col: str = EVENT_TIME_COL,
    event_time_source: str | None = "discharge_date",
) -> pd.DataFrame:
    """Return a copy of ``df`` shaped for Feature Store ingestion.

    Ensures:
    * ``record_id_col`` is present and cast to ``str``.
    * ``event_time_col`` is present and is a float (seconds since epoch).
    * Boolean columns are converted to ``int`` (Feature Store has no Bool).
    * Object columns are coerced to ``str`` so ``String`` features ingest
      cleanly.

    Parameters
    ----------
    event_time_source
        Column in ``df`` to derive ``event_time`` from when ``event_time_col``
        is not already populated. Defaults to ``discharge_date``. If neither is
        available, the current UTC time is used.
    """
    out = df.copy()

    if record_id_col not in out.columns:
        raise ValueError(f"record id column '{record_id_col}' missing from frame")
    out[record_id_col] = out[record_id_col].astype(str)

    if event_time_col not in out.columns:
        if event_time_source and event_time_source in out.columns:
            ts = pd.to_datetime(out[event_time_source], utc=True, errors="coerce")
            out[event_time_col] = ts.astype("int64") // 10**9
        else:
            out[event_time_col] = int(time.time())
    out[event_time_col] = out[event_time_col].astype("float64")

    # Booleans -> int (Feature Store rejects bool dtype)
    for col in out.select_dtypes(include=["bool"]).columns:
        out[col] = out[col].astype("int64")

    # Datetime columns -> ISO 8601 string (Feature Store has no datetime type;
    # the canonical pattern is to store them as Strings so they remain queryable
    # via Athena on the offline store).
    for col in out.select_dtypes(include=["datetime", "datetimetz"]).columns:
        if col == event_time_col:
            continue  # already cast to float epoch above
        out[col] = (
            pd.to_datetime(out[col], utc=True, errors="coerce")
            .dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            .fillna("")
        )

    # Object -> string (Feature Store rejects mixed objects / NaN object)
    for col in out.select_dtypes(include=["object"]).columns:
        out[col] = out[col].astype("string").fillna("")

    return out


def build_feature_definitions(df: pd.DataFrame):
    """Infer ``FeatureDefinition`` objects from a prepared DataFrame.

    Mapping:
        * float / decimal -> Fractional
        * int             -> Integral
        * anything else   -> String
    """
    from sagemaker.feature_store.feature_definition import (
        FeatureDefinition,
        FeatureTypeEnum,
    )

    defs: list[FeatureDefinition] = []
    for col, dtype in df.dtypes.items():
        if pd.api.types.is_float_dtype(dtype):
            ftype = FeatureTypeEnum.FRACTIONAL
        elif pd.api.types.is_integer_dtype(dtype):
            ftype = FeatureTypeEnum.INTEGRAL
        else:
            ftype = FeatureTypeEnum.STRING
        defs.append(FeatureDefinition(feature_name=col, feature_type=ftype))
    return defs


# ---------------------------------------------------------------------------
# Feature Group lifecycle
# ---------------------------------------------------------------------------

def _wait_for_status(feature_group, target: str = "Created", timeout_s: int = 600) -> None:
    """Poll until the FeatureGroup reaches ``target`` or we time out."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        status = feature_group.describe().get("FeatureGroupStatus")
        if status == target:
            return
        if status == "CreateFailed":
            raise RuntimeError(
                f"FeatureGroup {feature_group.name} create failed: "
                f"{feature_group.describe().get('FailureReason')}"
            )
        time.sleep(5)
    raise TimeoutError(
        f"FeatureGroup {feature_group.name} did not reach {target} in {timeout_s}s"
    )


def create_or_update_feature_group(
    df: pd.DataFrame,
    role_arn: str,
    name: str = FEATURE_GROUP_NAME,
    s3_uri_prefix: str = FEATURE_STORE_S3_PREFIX,
    region: str = "us-east-1",
    enable_online_store: bool = True,
    description: str | None = None,
):
    """Create the FeatureGroup if it does not exist, or return the existing one.

    Returns the :class:`sagemaker.feature_store.feature_group.FeatureGroup`.
    Safe to call repeatedly — existence is checked first.
    """
    import boto3
    from sagemaker.session import Session
    from sagemaker.feature_store.feature_group import FeatureGroup

    boto_sess = boto3.Session(region_name=region)
    sess = Session(boto_session=boto_sess)

    fg = FeatureGroup(name=name, sagemaker_session=sess)

    sm = boto_sess.client("sagemaker")
    try:
        sm.describe_feature_group(FeatureGroupName=name)
        logger.info("FeatureGroup %s already exists — reusing.", name)
        return fg
    except sm.exceptions.ResourceNotFound:
        pass

    prepared = prepare_records(df.head(1))  # only need a schema sample
    fg.load_feature_definitions(data_frame=prepared)

    fg.create(
        s3_uri=s3_uri_prefix,
        record_identifier_name=RECORD_ID_COL,
        event_time_feature_name=EVENT_TIME_COL,
        role_arn=role_arn,
        enable_online_store=enable_online_store,
        description=description
        or "30-day readmission features (AAI-540 Group 2).",
    )
    _wait_for_status(fg, "Created")
    logger.info("FeatureGroup %s created at %s", name, s3_uri_prefix)
    return fg


def ingest_features(
    df: pd.DataFrame,
    name: str = FEATURE_GROUP_NAME,
    region: str = "us-east-1",
    max_workers: int = 4,
    max_processes: int = 1,
    wait: bool = True,
):
    """Ingest a DataFrame into an existing FeatureGroup.

    The DataFrame is passed through :func:`prepare_records` first so callers
    don't have to remember the dtype rules.
    """
    import boto3
    from sagemaker.session import Session
    from sagemaker.feature_store.feature_group import FeatureGroup

    sess = Session(boto_session=boto3.Session(region_name=region))
    fg = FeatureGroup(name=name, sagemaker_session=sess)

    prepared = prepare_records(df)
    response = fg.ingest(
        data_frame=prepared,
        max_workers=max_workers,
        max_processes=max_processes,
        wait=wait,
    )
    logger.info("Ingested %d rows into FeatureGroup %s", len(prepared), name)
    return response


# ---------------------------------------------------------------------------
# Read paths
# ---------------------------------------------------------------------------

def get_latest_features(
    record_ids: Iterable[str],
    name: str = FEATURE_GROUP_NAME,
    feature_names: list[str] | None = None,
    region: str = "us-east-1",
) -> pd.DataFrame:
    """Read the latest record(s) from the *online* store.

    Returns a DataFrame indexed by ``beneficiary_id``.
    """
    import boto3

    runtime = boto3.client("sagemaker-featurestore-runtime", region_name=region)
    identifiers = [{"FeatureGroupName": name, "RecordIdentifiersValueAsString": [str(r) for r in record_ids]}]
    if feature_names:
        identifiers[0]["FeatureNames"] = feature_names

    resp = runtime.batch_get_record(Identifiers=identifiers)
    rows = []
    for rec in resp.get("Records", []):
        flat = {f["FeatureName"]: f["ValueAsString"] for f in rec["Record"]}
        rows.append(flat)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).set_index(RECORD_ID_COL, drop=True)
    return df


def query_offline_features(
    query: str,
    name: str = FEATURE_GROUP_NAME,
    region: str = "us-east-1",
    output_location: str | None = None,
) -> pd.DataFrame:
    """Run an Athena query against the *offline* store and return a DataFrame.

    ``query`` must reference the Athena table produced by the FeatureGroup
    (see ``fg.athena_query().table_name``).
    """
    import boto3
    from sagemaker.session import Session
    from sagemaker.feature_store.feature_group import FeatureGroup

    sess = Session(boto_session=boto3.Session(region_name=region))
    fg = FeatureGroup(name=name, sagemaker_session=sess)
    q = fg.athena_query()
    if output_location is None:
        output_location = f"{FEATURE_STORE_S3_PREFIX}/athena-results/"
    q.run(query_string=query, output_location=output_location)
    q.wait()
    return q.as_dataframe()
