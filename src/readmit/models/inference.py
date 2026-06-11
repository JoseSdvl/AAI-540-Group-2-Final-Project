"""SageMaker real-time inference entrypoint.

SageMaker's scikit-learn / XGBoost framework containers look for the four
functions ``model_fn``, ``input_fn``, ``predict_fn``, ``output_fn`` in the
script provided via ``source_dir`` + ``entry_point``. We implement all four
here so the deploy step can simply point at ``readmit/models/inference.py``.

Supported input content types:
    * ``application/json``  — ``{"instances": [{...feature_dict...}]}``
    * ``text/csv``          — header row required, columns must match
                              :data:`readmit.config.FEATURES.all`
"""

from __future__ import annotations

import io
import json
import os
import sys
from typing import Any

# SageMaker launches this entry point directly, so sys.path[0] is the script's
# own dir (.../code/readmit/models/), not the source_dir root. Add the root so
# `from readmit.*` imports resolve in-container.
_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import joblib
import numpy as np
import pandas as pd

from readmit.config import FEATURES
from readmit.features.engineering import add_derived_columns


# ---------------------------------------------------------------------------
# SageMaker hook functions
# ---------------------------------------------------------------------------

def model_fn(model_dir: str):
    """Load the joblib pipeline written by training."""
    return joblib.load(os.path.join(model_dir, "model.joblib"))


def input_fn(request_body: str | bytes, request_content_type: str):
    """Deserialize the incoming request into a DataFrame of feature rows."""
    ct = (request_content_type or "").lower()
    if "json" in ct:
        body = json.loads(request_body)
        instances = body.get("instances") if isinstance(body, dict) else body
        if instances is None:
            raise ValueError("JSON payload must include 'instances' list")
        df = pd.DataFrame(instances)
    elif "csv" in ct:
        df = pd.read_csv(io.StringIO(_to_text(request_body)))
    else:
        raise ValueError(f"Unsupported content type: {request_content_type!r}")
    return _ensure_features(df)


def predict_fn(input_df: pd.DataFrame, model) -> dict[str, Any]:
    """Score the input frame; return both probability and binary flag."""
    proba = model.predict_proba(input_df)[:, 1]
    threshold = float(os.environ.get("READMIT_THRESHOLD", "0.5"))
    flag = (proba >= threshold).astype(np.int8)
    return {
        "risk_score": proba.tolist(),
        "high_risk_flag": flag.tolist(),
        "threshold": threshold,
    }


def output_fn(prediction: dict, accept: str) -> tuple[str, str]:
    """Serialize predictions; default to JSON."""
    accept = (accept or "application/json").lower()
    if "csv" in accept:
        df = pd.DataFrame(
            {"risk_score": prediction["risk_score"],
             "high_risk_flag": prediction["high_risk_flag"]}
        )
        return df.to_csv(index=False), "text/csv"
    return json.dumps(prediction), "application/json"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_text(payload: str | bytes) -> str:
    return payload.decode("utf-8") if isinstance(payload, (bytes, bytearray)) else payload


def _ensure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same derived columns used in training and order the columns."""
    df = add_derived_columns(df)
    missing = [c for c in FEATURES.all if c not in df.columns]
    if missing:
        raise ValueError(f"Inference payload missing required features: {missing}")
    return df[FEATURES.all]
