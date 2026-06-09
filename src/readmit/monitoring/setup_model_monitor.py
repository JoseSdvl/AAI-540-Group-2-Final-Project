"""Configure SageMaker Model Monitor for the readmission endpoint.

Two monitors are wired up:

1. **DataQualityMonitor** (`DefaultModelMonitor`) — baselines numeric +
   categorical feature distributions against the training data, then runs
   hourly against captured inference traffic to detect feature drift.

2. **ModelQualityMonitor** — scheduled job that joins ground-truth labels
   (uploaded daily/weekly by the care-team workflow) with the captured
   predictions and computes AUC / accuracy / precision / recall. The custom
   ``ReadmitModel/AUC`` metric it publishes is what the CloudWatch alarm in
   :mod:`readmit.monitoring.alerts` watches.

Run order::

    enable_data_capture(endpoint_name)        # once
    baseline = create_data_quality_baseline() # once per training
    create_data_quality_schedule(...)         # once
    baseline_q = create_model_quality_baseline()
    create_model_quality_schedule(...)
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def enable_data_capture(
    endpoint_name: str,
    capture_s3_uri: str,
    sampling_pct: int = 100,
    region: str = "us-east-1",
) -> None:
    """Enable DataCapture on an existing endpoint in-place."""
    import boto3
    from sagemaker.model_monitor import DataCaptureConfig
    from sagemaker.predictor import Predictor
    from sagemaker.session import Session

    boto_session = boto3.Session(region_name=region)
    sm_session = Session(boto_session=boto_session)

    capture = DataCaptureConfig(
        enable_capture=True,
        sampling_percentage=sampling_pct,
        destination_s3_uri=capture_s3_uri,
        capture_options=["REQUEST", "RESPONSE"],
        sagemaker_session=sm_session,
    )
    Predictor(endpoint_name=endpoint_name,
              sagemaker_session=sm_session).update_data_capture_config(capture)
    logger.info("Data capture enabled on %s -> %s", endpoint_name, capture_s3_uri)


def create_data_quality_baseline(
    role_arn: str,
    baseline_dataset_s3_uri: str,
    output_s3_uri: str,
    region: str = "us-east-1",
    instance_type: str = "ml.m5.xlarge",
):
    """Compute statistics + constraints from the training data."""
    import boto3
    from sagemaker.model_monitor import DefaultModelMonitor
    from sagemaker.model_monitor.dataset_format import DatasetFormat
    from sagemaker.session import Session

    session = Session(boto_session=boto3.Session(region_name=region))
    monitor = DefaultModelMonitor(
        role=role_arn, instance_count=1, instance_type=instance_type,
        volume_size_in_gb=20, max_runtime_in_seconds=3600,
        sagemaker_session=session,
    )
    monitor.suggest_baseline(
        baseline_dataset=baseline_dataset_s3_uri,
        dataset_format=DatasetFormat.csv(header=True),
        output_s3_uri=output_s3_uri,
        wait=True,
    )
    logger.info("Data-quality baseline written to %s", output_s3_uri)
    return monitor


def create_data_quality_schedule(
    monitor,
    endpoint_name: str,
    schedule_name: str = "readmit-data-quality-hourly",
    output_s3_uri: Optional[str] = None,
):
    """Attach an hourly drift-check schedule to the endpoint."""
    from sagemaker.model_monitor import CronExpressionGenerator

    monitor.create_monitoring_schedule(
        monitor_schedule_name=schedule_name,
        endpoint_input=endpoint_name,
        output_s3_uri=output_s3_uri,
        statistics=monitor.baseline_statistics(),
        constraints=monitor.suggested_constraints(),
        schedule_cron_expression=CronExpressionGenerator.hourly(),
        enable_cloudwatch_metrics=True,
    )
    logger.info("Data-quality monitoring schedule created: %s", schedule_name)


def create_model_quality_baseline(
    role_arn: str,
    baseline_dataset_s3_uri: str,
    output_s3_uri: str,
    ground_truth_attribute: str = "readmitted_30d",
    inference_attribute: str = "high_risk_flag",
    probability_attribute: str = "risk_score",
    region: str = "us-east-1",
    instance_type: str = "ml.m5.xlarge",
):
    """Baseline AUC/F1/precision/recall on a labelled validation set."""
    import boto3
    from sagemaker.model_monitor import ModelQualityMonitor
    from sagemaker.model_monitor.dataset_format import DatasetFormat
    from sagemaker.session import Session

    session = Session(boto_session=boto3.Session(region_name=region))
    monitor = ModelQualityMonitor(
        role=role_arn, instance_count=1, instance_type=instance_type,
        volume_size_in_gb=20, max_runtime_in_seconds=3600,
        sagemaker_session=session,
    )
    monitor.suggest_baseline(
        baseline_dataset=baseline_dataset_s3_uri,
        dataset_format=DatasetFormat.csv(header=True),
        problem_type="BinaryClassification",
        inference_attribute=inference_attribute,
        probability_attribute=probability_attribute,
        ground_truth_attribute=ground_truth_attribute,
        output_s3_uri=output_s3_uri,
        wait=True,
    )
    logger.info("Model-quality baseline written to %s", output_s3_uri)
    return monitor


def create_model_quality_schedule(
    monitor,
    endpoint_name: str,
    ground_truth_input_s3_uri: str,
    schedule_name: str = "readmit-model-quality-daily",
    output_s3_uri: Optional[str] = None,
):
    """Daily model-quality monitor; joins captured predictions to GT labels."""
    from sagemaker.model_monitor import CronExpressionGenerator, EndpointInput

    monitor.create_monitoring_schedule(
        monitor_schedule_name=schedule_name,
        endpoint_input=EndpointInput(
            endpoint_name=endpoint_name,
            destination="/opt/ml/processing/input_data",
            probability_attribute="risk_score",
            probability_threshold_attribute=0.5,
            inference_attribute="high_risk_flag",
        ),
        problem_type="BinaryClassification",
        ground_truth_input=ground_truth_input_s3_uri,
        output_s3_uri=output_s3_uri,
        constraints=monitor.suggested_constraints(),
        schedule_cron_expression=CronExpressionGenerator.daily(),
        enable_cloudwatch_metrics=True,
    )
    logger.info("Model-quality monitoring schedule created: %s", schedule_name)
