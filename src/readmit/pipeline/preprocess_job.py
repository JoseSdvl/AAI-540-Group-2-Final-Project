"""SageMaker Processing entrypoint: build curated + featurised splits.

Runs inside an ``SKLearnProcessor`` job. Inputs/outputs follow the SageMaker
Processing convention of ``/opt/ml/processing/{input,output}``.

Outputs (parquet):
    /opt/ml/processing/train/train.parquet
    /opt/ml/processing/test/test.parquet
    /opt/ml/processing/validation/val.parquet
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from readmit.data.ingest import load_encounters
from readmit.data.labeling import attach_labels_and_priors
from readmit.data.splits import split_train_test_val

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["synthetic", "s3"], default="synthetic")
    p.add_argument("--s3-uri", default=None,
                   help="Required when --source s3, e.g. s3://cms-readmit-curated/")
    p.add_argument("--n-patients", type=int, default=50_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-out", default="/opt/ml/processing/train")
    p.add_argument("--test-out", default="/opt/ml/processing/test")
    p.add_argument("--val-out", default="/opt/ml/processing/validation")
    args = p.parse_args()

    logger.info("Loading encounters (source=%s, n=%d)", args.source, args.n_patients)
    encounters = load_encounters(
        source=args.source, s3_uri=args.s3_uri,
        n_patients=args.n_patients, seed=args.seed,
    )

    logger.info("Attaching 30-day readmission labels + priors")
    labeled = attach_labels_and_priors(encounters)

    logger.info("Splitting 40/30/30 (train/test/val) at the patient level")
    train_df, test_df, val_df = split_train_test_val(labeled, seed=args.seed)

    for out_dir, df, name in [
        (args.train_out, train_df, "train"),
        (args.test_out, test_df, "test"),
        (args.val_out, val_df, "val"),
    ]:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        path = Path(out_dir) / f"{name}.parquet"
        df.to_parquet(path, index=False)
        logger.info("Wrote %s rows=%d positive_rate=%.4f -> %s",
                    name, len(df), df["readmitted_30d"].mean(), path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
