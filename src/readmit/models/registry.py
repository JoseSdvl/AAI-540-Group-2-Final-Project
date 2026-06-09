"""SageMaker Model Registry (a.k.a. Model Store) helpers.

The registry is our system of record for production-eligible models. Each
training run lands as a *ModelPackage* inside the
``ReadmitRiskModels`` ModelPackageGroup with:

* `ModelDataUrl`              -> S3 location of the model artifact (.tar.gz)
* `ModelMetrics`              -> S3 location of evaluation.json
* `ModelApprovalStatus`       -> PendingManualApproval | Approved | Rejected
* `InferenceSpecification`    -> the SKLearn image + supported MIME types

The CD pipeline only deploys versions that are ``Approved``. Manual approval
happens either in SageMaker Studio or via :func:`transition_approval` below.

Public API
----------
* :func:`ensure_model_package_group`
* :func:`register_model_version`
* :func:`list_versions`
* :func:`get_latest_approved`
* :func:`transition_approval`
* :func:`deploy_from_registry`
"""

from __future__ import annotations

import logging
from typing import Any

from readmit.config import (
    AUC_THRESHOLD_DEPLOY,
    DEFAULT_REGION,
    ENDPOINT_INSTANCE,
    MODEL_PACKAGE_GROUP,
    MODEL_PACKAGE_GROUP_DESCRIPTION,
)

logger = logging.getLogger(__name__)

APPROVED = "Approved"
REJECTED = "Rejected"
PENDING = "PendingManualApproval"


# ---------------------------------------------------------------------------
# Group lifecycle
# ---------------------------------------------------------------------------

def ensure_model_package_group(
    group_name: str = MODEL_PACKAGE_GROUP,
    description: str = MODEL_PACKAGE_GROUP_DESCRIPTION,
    region: str = DEFAULT_REGION,
) -> str:
    """Create the ModelPackageGroup if missing; return its ARN."""
    import boto3

    sm = boto3.client("sagemaker", region_name=region)
    try:
        resp = sm.describe_model_package_group(ModelPackageGroupName=group_name)
        logger.info("ModelPackageGroup %s already exists.", group_name)
        return resp["ModelPackageGroupArn"]
    except sm.exceptions.ClientError as exc:
        if "does not exist" not in str(exc) and "ValidationException" not in str(exc):
            raise

    resp = sm.create_model_package_group(
        ModelPackageGroupName=group_name,
        ModelPackageGroupDescription=description,
    )
    logger.info("ModelPackageGroup %s created.", group_name)
    return resp["ModelPackageGroupArn"]


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def _sklearn_inference_image(region: str, framework_version: str = "1.2-1") -> str:
    """Return the official SKLearn inference image URI for ``region``.

    ``framework_version`` must be one of the tags published by AWS
    (e.g. ``"0.23-1"``, ``"1.0-1"``, ``"1.2-1"``). The full tag —
    including the SageMaker container build suffix (``-1``) — is required;
    do *not* strip it. As of mid-2025 the newest sklearn inference image
    is ``1.2-1``; pickles produced by sklearn 1.4+ may still load (joblib
    is forward-compatible for simple estimators), but pin training to
    sklearn 1.2 if you hit unpickling errors at inference time.
    """
    from sagemaker.image_uris import retrieve

    return retrieve(
        framework="sklearn",
        region=region,
        version=framework_version,
        py_version="py3",
        instance_type=ENDPOINT_INSTANCE,
        image_scope="inference",
    )


def register_model_version(
    model_data_s3_uri: str,
    group_name: str = MODEL_PACKAGE_GROUP,
    region: str = DEFAULT_REGION,
    approval_status: str = PENDING,
    image_uri: str | None = None,
    metrics_s3_uri: str | None = None,
    description: str | None = None,
    customer_metadata: dict[str, str] | None = None,
) -> str:
    """Register one model version. Returns the ``ModelPackageArn``."""
    import boto3

    ensure_model_package_group(group_name, region=region)
    sm = boto3.client("sagemaker", region_name=region)

    image_uri = image_uri or _sklearn_inference_image(region)

    inference_spec: dict[str, Any] = {
        "Containers": [
            {
                "Image": image_uri,
                "ModelDataUrl": model_data_s3_uri,
            }
        ],
        "SupportedContentTypes": ["application/json", "text/csv"],
        "SupportedResponseMIMETypes": ["application/json"],
        "SupportedRealtimeInferenceInstanceTypes": [
            ENDPOINT_INSTANCE,
            "ml.m5.xlarge",
            "ml.c5.large",
            "ml.c5.xlarge",
        ],
        "SupportedTransformInstanceTypes": ["ml.m5.xlarge"],
    }

    kwargs: dict[str, Any] = {
        "ModelPackageGroupName": group_name,
        "ModelPackageDescription": description
        or "Readmission-risk XGBoost — registered from training pipeline.",
        "InferenceSpecification": inference_spec,
        "ModelApprovalStatus": approval_status,
    }
    if metrics_s3_uri:
        kwargs["ModelMetrics"] = {
            "ModelQuality": {
                "Statistics": {
                    "ContentType": "application/json",
                    "S3Uri": metrics_s3_uri,
                }
            }
        }
    if customer_metadata:
        kwargs["CustomerMetadataProperties"] = {
            k: str(v) for k, v in customer_metadata.items()
        }

    resp = sm.create_model_package(**kwargs)
    arn = resp["ModelPackageArn"]
    logger.info("Registered ModelPackage %s (status=%s)", arn, approval_status)
    return arn


# ---------------------------------------------------------------------------
# Discovery / approval
# ---------------------------------------------------------------------------

def list_versions(
    group_name: str = MODEL_PACKAGE_GROUP,
    status_filter: str | None = None,
    region: str = DEFAULT_REGION,
    max_results: int = 50,
) -> list[dict]:
    """List model versions in a group, newest first."""
    import boto3

    sm = boto3.client("sagemaker", region_name=region)
    kwargs: dict[str, Any] = {
        "ModelPackageGroupName": group_name,
        "SortBy": "CreationTime",
        "SortOrder": "Descending",
        "MaxResults": max_results,
    }
    if status_filter:
        kwargs["ModelApprovalStatus"] = status_filter
    return sm.list_model_packages(**kwargs).get("ModelPackageSummaryList", [])


def get_latest_approved(
    group_name: str = MODEL_PACKAGE_GROUP,
    region: str = DEFAULT_REGION,
) -> str | None:
    """Return the ARN of the most recent ``Approved`` version, or ``None``."""
    versions = list_versions(group_name=group_name, status_filter=APPROVED, region=region)
    if not versions:
        return None
    return versions[0]["ModelPackageArn"]


def transition_approval(
    model_package_arn: str,
    new_status: str,
    region: str = DEFAULT_REGION,
    reason: str | None = None,
) -> None:
    """Move a ModelPackage to ``Approved`` / ``Rejected`` / ``PendingManualApproval``."""
    if new_status not in {APPROVED, REJECTED, PENDING}:
        raise ValueError(f"new_status must be one of Approved/Rejected/PendingManualApproval, got {new_status}")
    import boto3

    sm = boto3.client("sagemaker", region_name=region)
    kwargs: dict[str, Any] = {
        "ModelPackageArn": model_package_arn,
        "ModelApprovalStatus": new_status,
    }
    if reason:
        kwargs["ApprovalDescription"] = reason
    sm.update_model_package(**kwargs)
    logger.info("ModelPackage %s -> %s", model_package_arn, new_status)


# ---------------------------------------------------------------------------
# Deploy from registry
# ---------------------------------------------------------------------------

def deploy_from_registry(
    model_package_arn: str,
    endpoint_name: str,
    role_arn: str,
    region: str = DEFAULT_REGION,
    instance_type: str = ENDPOINT_INSTANCE,
    initial_instance_count: int = 1,
    data_capture_config: Any | None = None,
):
    """Deploy a registered ModelPackage to a real-time endpoint."""
    import boto3
    from sagemaker import ModelPackage, Session

    sess = Session(boto_session=boto3.Session(region_name=region))
    model = ModelPackage(
        role=role_arn,
        model_package_arn=model_package_arn,
        sagemaker_session=sess,
    )
    predictor = model.deploy(
        initial_instance_count=initial_instance_count,
        instance_type=instance_type,
        endpoint_name=endpoint_name,
        data_capture_config=data_capture_config,
    )
    logger.info("Deployed %s to endpoint %s", model_package_arn, endpoint_name)
    return predictor


# ---------------------------------------------------------------------------
# Convenience: gate by metric before promotion
# ---------------------------------------------------------------------------

def auto_approve_if_above_threshold(
    model_package_arn: str,
    auc: float,
    threshold: float = AUC_THRESHOLD_DEPLOY,
    region: str = DEFAULT_REGION,
) -> bool:
    """Approve a model only when its AUC clears the deploy threshold."""
    if auc >= threshold:
        transition_approval(
            model_package_arn,
            APPROVED,
            region=region,
            reason=f"Automated approval — AUC {auc:.4f} >= {threshold}",
        )
        return True
    transition_approval(
        model_package_arn,
        REJECTED,
        region=region,
        reason=f"Automated rejection — AUC {auc:.4f} < {threshold}",
    )
    return False
