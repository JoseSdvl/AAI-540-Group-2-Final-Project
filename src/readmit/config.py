"""Project-wide configuration constants.

Centralising bucket names, instance types, and split ratios here keeps the
training script, pipeline, and notebook in lockstep.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---- Split ratios (per AAI-540 final-project requirement) -------------------
TRAIN_FRAC: float = 0.40
TEST_FRAC: float = 0.30
VAL_FRAC: float = 0.30
RANDOM_SEED: int = 42

assert abs(TRAIN_FRAC + TEST_FRAC + VAL_FRAC - 1.0) < 1e-9


# ---- AWS resource defaults --------------------------------------------------
DEFAULT_REGION: str = "us-east-1"

# Buckets are intentionally placeholders; override via env or notebook config.
RAW_BUCKET: str = "cms-readmit-raw"
CURATED_BUCKET: str = "cms-readmit-curated"
FEATURES_BUCKET: str = "cms-readmit-features"
MODEL_BUCKET: str = "cms-readmit-model-artifacts"
MONITOR_BUCKET: str = "cms-readmit-monitoring"


# ---- Compute defaults -------------------------------------------------------
TRAINING_INSTANCE: str = "ml.m5.xlarge"
ENDPOINT_INSTANCE: str = "ml.m5.large"
BATCH_INSTANCE: str = "ml.m5.xlarge"


# ---- Modeling defaults ------------------------------------------------------
PRIMARY_METRIC: str = "auc"
AUC_THRESHOLD_DEPLOY: float = 0.75
AUC_THRESHOLD_ALERT: float = 0.70  # below this -> CloudWatch alarm fires


# ---- Feature Store + Model Registry ----------------------------------------
FEATURE_GROUP_NAME: str = "readmit-encounter-features"
FEATURE_STORE_S3_PREFIX: str = f"s3://{FEATURES_BUCKET}/feature-store"
MODEL_PACKAGE_GROUP: str = "ReadmitRiskModels"
MODEL_PACKAGE_GROUP_DESCRIPTION: str = (
    "Readmission-risk model versions (XGBoost on CMS DE-SynPUF). "
    "AAI-540 Group 2 / ClearPath Health Analytics."
)


# ---- Schemas / labels -------------------------------------------------------
LABEL_COL: str = "readmitted_30d"
PATIENT_ID_COL: str = "beneficiary_id"
ADMIT_DATE_COL: str = "admission_date"
DISCHARGE_DATE_COL: str = "discharge_date"
PRIMARY_DX_COL: str = "primary_diagnosis"


@dataclass(frozen=True)
class FeatureSpec:
    """Declared feature set, kept in one place so train/inference agree."""

    numeric: list[str] = field(
        default_factory=lambda: [
            "length_of_stay",
            "prior_inpatient_90d",
            "prior_ed_90d",
            "prior_outpatient_90d",
            "n_chronic_conditions",
            "charlson_index",
        ]
    )
    categorical: list[str] = field(
        default_factory=lambda: [
            "age_band",
            "sex",
            "primary_dx_chapter",
            "discharge_disposition",
            "payer_type",
            "los_bucket",
        ]
    )

    @property
    def all(self) -> list[str]:
        return self.numeric + self.categorical


FEATURES = FeatureSpec()
