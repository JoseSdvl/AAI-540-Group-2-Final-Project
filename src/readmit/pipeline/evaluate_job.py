"""SageMaker Processing entrypoint: evaluate the trained model on the test split.

Inputs (mounted by the pipeline):
    /opt/ml/processing/model/model.tar.gz   — model artifact from training
    /opt/ml/processing/test/test.parquet    — held-out test split

Output:
    /opt/ml/processing/evaluation/evaluation.json   — read by ConditionStep
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tarfile
from pathlib import Path

import joblib
import pandas as pd

# When SageMaker FrameworkProcessor runs this script, sys.path[0] is the script's
# own directory (.../code/readmit/pipeline/), NOT the source_dir root (.../code/),
# so `from readmit.*` cannot resolve. Add the source_dir root to sys.path.
_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from readmit.config import FEATURES, LABEL_COL
from readmit.features.engineering import add_derived_columns, winsorize_utilization
from readmit.models.evaluate import compute_metrics, recall_at_k, subgroup_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _load_model(model_dir: Path):
    """Untar (if needed) and load the joblib pipeline."""
    tar = model_dir / "model.tar.gz"
    if tar.exists():
        with tarfile.open(tar) as t:
            t.extractall(model_dir)
    return joblib.load(model_dir / "model.joblib")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default="/opt/ml/processing/model")
    p.add_argument("--test-dir", default="/opt/ml/processing/test")
    p.add_argument("--output-dir", default="/opt/ml/processing/evaluation")
    args = p.parse_args()

    model = _load_model(Path(args.model_dir))

    test_files = sorted(Path(args.test_dir).glob("*.parquet"))
    if not test_files:
        raise FileNotFoundError(f"No parquet in {args.test_dir}")
    df = pd.concat([pd.read_parquet(f) for f in test_files], ignore_index=True)
    df = add_derived_columns(winsorize_utilization(df))

    y = df[LABEL_COL].to_numpy()
    x = df[FEATURES.all]
    proba = model.predict_proba(x)[:, 1]

    metrics = compute_metrics(y, proba)
    metrics["recall_at_top_10pct"] = recall_at_k(y, proba, 0.10)
    metrics["recall_at_top_20pct"] = recall_at_k(y, proba, 0.20)
    subgroup = subgroup_metrics(df, y, proba, group_cols=["age_band", "sex",
                                                          "primary_dx_chapter"])

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "binary_classification_metrics": {
            "auc": {"value": metrics["auc"]},
            "pr_auc": {"value": metrics["pr_auc"]},
            "f1": {"value": metrics["f1"]},
            "precision": {"value": metrics["precision"]},
            "recall": {"value": metrics["recall"]},
            "brier_score": {"value": metrics["brier"]},
            "recall_at_top_10pct": {"value": metrics["recall_at_top_10pct"]},
            "recall_at_top_20pct": {"value": metrics["recall_at_top_20pct"]},
        },
        "summary": metrics,
    }
    (out_dir / "evaluation.json").write_text(json.dumps(report, indent=2))
    subgroup.to_csv(out_dir / "subgroup_metrics.csv", index=False)
    logger.info("Wrote evaluation report to %s (AUC=%.4f)", out_dir, metrics["auc"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
