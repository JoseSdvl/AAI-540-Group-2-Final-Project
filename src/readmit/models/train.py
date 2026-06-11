"""SageMaker-compatible training entry point.

This script is what SageMaker invokes inside the training container. It is
*also* runnable locally:

    python -m readmit.models.train \
        --train /tmp/train.parquet --val /tmp/val.parquet \
        --model-dir /tmp/model --output-dir /tmp/out

The SageMaker convention is:

* hyperparameters arrive as CLI flags (``--learning-rate``, etc.)
* input channels arrive as environment variables (``SM_CHANNEL_TRAIN``)
* the model artifact is written to ``SM_MODEL_DIR`` (default ``/opt/ml/model``)
* evaluation metrics are written to ``SM_OUTPUT_DATA_DIR``
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# SageMaker launches this entry point directly, so sys.path[0] is the script's
# own dir (.../code/readmit/models/), not the source_dir root. Add the root so
# `from readmit.*` imports resolve in-container.
_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from readmit.config import FEATURES, LABEL_COL, RANDOM_SEED
from readmit.features.engineering import (
    add_derived_columns,
    build_preprocessor,
    winsorize_utilization,
)
from readmit.models.evaluate import compute_metrics

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # Data channels (SageMaker injects these env vars; we fall back for local runs)
    p.add_argument("--train", default=os.environ.get("SM_CHANNEL_TRAIN", "data/train"))
    p.add_argument("--val", default=os.environ.get("SM_CHANNEL_VALIDATION", "data/val"))
    p.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "artifacts/model"))
    p.add_argument("--output-dir", default=os.environ.get("SM_OUTPUT_DATA_DIR", "artifacts/out"))

    # Hyperparameters
    p.add_argument("--model-kind", choices=["xgboost", "logreg"], default="xgboost")
    p.add_argument("--learning-rate", type=float, default=0.08)
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--n-estimators", type=int, default=400)
    p.add_argument("--subsample", type=float, default=0.9)
    p.add_argument("--colsample-bytree", type=float, default=0.9)
    p.add_argument("--reg-lambda", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    return p.parse_args()


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def _read_channel(path: str) -> pd.DataFrame:
    """Read either a directory of parquet files or a single parquet/csv."""
    p = Path(path)
    if p.is_dir():
        files = sorted([*p.glob("*.parquet"), *p.glob("*.csv")])
        if not files:
            raise FileNotFoundError(f"No parquet/csv files in {p}")
        frames = [pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f)
                  for f in files]
        return pd.concat(frames, ignore_index=True)
    if p.suffix == ".csv":
        return pd.read_csv(p)
    return pd.read_parquet(p)


def _split_x_y(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    df = add_derived_columns(winsorize_utilization(df))
    y = df[LABEL_COL].to_numpy().astype(np.int8)
    x = df[FEATURES.all]
    return x, y


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------

def build_xgb_pipeline(args: argparse.Namespace, scale_pos_weight: float) -> Pipeline:
    pre = build_preprocessor(scale_numeric=False)
    clf = XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_lambda=args.reg_lambda,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        random_state=args.seed,
        n_jobs=-1,
    )
    return Pipeline([("preprocessor", pre), ("classifier", clf)])


def build_logreg_pipeline(args: argparse.Namespace) -> Pipeline:
    pre = build_preprocessor(scale_numeric=True)
    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=args.seed,
        n_jobs=-1,
    )
    return Pipeline([("preprocessor", pre), ("classifier", clf)])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = _parse_args()
    logger.info("Args: %s", vars(args))

    train_df = _read_channel(args.train)
    val_df = _read_channel(args.val)
    logger.info("Train rows=%d, Val rows=%d", len(train_df), len(val_df))

    x_train, y_train = _split_x_y(train_df)
    x_val, y_val = _split_x_y(val_df)

    # scale_pos_weight = N_neg / N_pos handles class imbalance for XGBoost.
    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    scale_pos_weight = max(neg / max(pos, 1), 1.0)
    logger.info("Class balance: pos=%d neg=%d scale_pos_weight=%.3f",
                pos, neg, scale_pos_weight)

    if args.model_kind == "xgboost":
        pipeline = build_xgb_pipeline(args, scale_pos_weight=scale_pos_weight)
    else:
        pipeline = build_logreg_pipeline(args)

    pipeline.fit(x_train, y_train)

    val_proba = pipeline.predict_proba(x_val)[:, 1]
    metrics = compute_metrics(y_val, val_proba)
    logger.info("Validation metrics: %s", metrics)

    # ---- Save artifacts -----------------------------------------------------
    model_dir = Path(args.model_dir); model_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, model_dir / "model.joblib")
    (model_dir / "feature_spec.json").write_text(
        json.dumps({"numeric": FEATURES.numeric, "categorical": FEATURES.categorical}, indent=2)
    )

    # SageMaker Model Monitor & the Pipelines step both look for this exact
    # filename when deciding whether the model passes the deploy gate.
    (output_dir / "evaluation.json").write_text(
        json.dumps(
            {
                "binary_classification_metrics": {
                    "auc": {"value": metrics["auc"]},
                    "pr_auc": {"value": metrics["pr_auc"]},
                    "f1": {"value": metrics["f1"]},
                    "precision": {"value": metrics["precision"]},
                    "recall": {"value": metrics["recall"]},
                    "brier_score": {"value": metrics["brier"]},
                }
            },
            indent=2,
        )
    )
    logger.info("Wrote model to %s and metrics to %s", model_dir, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
