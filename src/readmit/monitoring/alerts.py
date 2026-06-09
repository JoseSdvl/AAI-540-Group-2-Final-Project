"""CloudWatch alarms + dashboard for the readmission endpoint.

This module is idempotent: re-running it just updates the existing
alarms/dashboard/topic in place. It is called once at deploy time from CI/CD.

Alarms wired up:

* ``Readmit-Endpoint-Latency-p95``      — SageMaker/InvocationLatency p95 > 500ms
* ``Readmit-Endpoint-5xx``              — SageMaker invocation 5xx > 0 in 5 min
* ``Readmit-Endpoint-4xx``              — SageMaker invocation 4xx > 10 in 5 min
* ``Readmit-Endpoint-CPU``              — CPUUtilization > 80% for 15 min
* ``Readmit-DataDrift-Violations``      — Model Monitor drift > 0
* ``Readmit-ModelQuality-AUC``          — SageMaker Model Monitor AUC < floor
* ``Readmit-Custom-AUC-Floor``          — custom ReadmitModel/AUC < 0.70

All alarms publish to an SNS topic so the alarm state transitions are visible
in the AWS console and persisted in CloudWatch — no external subscribers
(email / Slack / PagerDuty) are configured here. A CloudWatch **dashboard**
is also created so operators have a single AWS-native view of latency,
invocation errors, drift, and AUC.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from readmit.config import AUC_THRESHOLD_ALERT

logger = logging.getLogger(__name__)

ALERT_TOPIC_NAME = "readmit-model-alerts"
CUSTOM_NAMESPACE = "ReadmitModel"
DASHBOARD_NAME_TEMPLATE = "Readmit-{endpoint}"


# ---------------------------------------------------------------------------
# SNS topic (alarm action target only — no external subscribers)
# ---------------------------------------------------------------------------

def ensure_sns_topic(
    topic_name: str = ALERT_TOPIC_NAME,
    region: str = "us-east-1",
) -> str:
    """Create (or fetch) the SNS topic that alarms publish to.

    The topic is used purely as the alarm-action target so alarm state
    transitions are recorded and visible in the AWS console. No subscribers
    are configured.

    Returns the topic ARN.
    """
    import boto3

    sns = boto3.client("sns", region_name=region)
    topic_arn = sns.create_topic(Name=topic_name)["TopicArn"]
    logger.info("SNS topic ready: %s (no subscribers configured)", topic_arn)
    return topic_arn


# ---------------------------------------------------------------------------
# CloudWatch alarms
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AlarmSpec:
    name: str
    namespace: str
    metric_name: str
    dimensions: list[dict]
    statistic: str
    threshold: float
    comparison: str  # e.g. "GreaterThanThreshold"
    period: int = 300
    evaluation_periods: int = 1
    description: str = ""
    extended_statistic: str | None = None  # for "p95", "p99"


def _put_alarm(cw, spec: AlarmSpec, topic_arn: str) -> None:
    kwargs: dict = {
        "AlarmName": spec.name,
        "AlarmDescription": spec.description,
        "Namespace": spec.namespace,
        "MetricName": spec.metric_name,
        "Dimensions": spec.dimensions,
        "Period": spec.period,
        "EvaluationPeriods": spec.evaluation_periods,
        "Threshold": spec.threshold,
        "ComparisonOperator": spec.comparison,
        "TreatMissingData": "notBreaching",
        "ActionsEnabled": True,
        "AlarmActions": [topic_arn],
        "OKActions": [topic_arn],
    }
    if spec.extended_statistic:
        kwargs["ExtendedStatistic"] = spec.extended_statistic
    else:
        kwargs["Statistic"] = spec.statistic
    cw.put_metric_alarm(**kwargs)
    logger.info("Upserted alarm %s", spec.name)


def configure_alarms(
    endpoint_name: str,
    variant_name: str = "AllTraffic",
    topic_arn: str | None = None,
    region: str = "us-east-1",
    auc_floor: float = AUC_THRESHOLD_ALERT,
) -> str:
    """Create/refresh all CloudWatch alarms for ``endpoint_name``.

    Returns the SNS topic ARN that the alarms publish to. The topic has no
    external subscribers — alarm state is intended to be reviewed in the
    AWS CloudWatch console / dashboard.
    """
    import boto3

    if topic_arn is None:
        topic_arn = ensure_sns_topic(region=region)

    cw = boto3.client("cloudwatch", region_name=region)

    endpoint_dim = [
        {"Name": "EndpointName", "Value": endpoint_name},
        {"Name": "VariantName", "Value": variant_name},
    ]
    monitor_dim = [{"Name": "EndpointName", "Value": endpoint_name}]
    model_dim = [{"Name": "EndpointName", "Value": endpoint_name}]

    specs = [
        AlarmSpec(
            name=f"Readmit-{endpoint_name}-Latency-p95",
            namespace="AWS/SageMaker",
            metric_name="ModelLatency",
            dimensions=endpoint_dim,
            statistic="",  # using extended_statistic
            extended_statistic="p95",
            threshold=500_000.0,  # microseconds → 500 ms
            comparison="GreaterThanThreshold",
            period=300, evaluation_periods=3,
            description="p95 model latency > 500ms for 15 minutes",
        ),
        AlarmSpec(
            name=f"Readmit-{endpoint_name}-5xx",
            namespace="AWS/SageMaker",
            metric_name="Invocation5XXErrors",
            dimensions=endpoint_dim,
            statistic="Sum",
            threshold=0,
            comparison="GreaterThanThreshold",
            period=300, evaluation_periods=1,
            description="Any 5xx errors in the last 5 minutes",
        ),
        AlarmSpec(
            name=f"Readmit-{endpoint_name}-4xx",
            namespace="AWS/SageMaker",
            metric_name="Invocation4XXErrors",
            dimensions=endpoint_dim,
            statistic="Sum",
            threshold=10,
            comparison="GreaterThanThreshold",
            period=300, evaluation_periods=1,
            description="More than 10 4xx errors in 5 minutes",
        ),
        AlarmSpec(
            name=f"Readmit-{endpoint_name}-CPU",
            namespace="/aws/sagemaker/Endpoints",
            metric_name="CPUUtilization",
            dimensions=endpoint_dim,
            statistic="Average",
            threshold=80.0,
            comparison="GreaterThanThreshold",
            period=300, evaluation_periods=3,
            description="Average CPU > 80% for 15 minutes",
        ),
        AlarmSpec(
            name=f"Readmit-{endpoint_name}-DataDrift",
            namespace="aws/sagemaker/Endpoints/data-metrics",
            metric_name="feature_baseline_drift_check_violations",
            dimensions=monitor_dim,
            statistic="Sum",
            threshold=0,
            comparison="GreaterThanThreshold",
            period=3600, evaluation_periods=1,
            description="Any feature baseline-drift violations in the last hour",
        ),
        AlarmSpec(
            name=f"Readmit-{endpoint_name}-ModelQuality-AUC",
            namespace="aws/sagemaker/Endpoints/model-metrics",
            metric_name="auc",
            dimensions=model_dim,
            statistic="Average",
            threshold=auc_floor,
            comparison="LessThanThreshold",
            period=86400, evaluation_periods=1,
            description=f"Model-quality AUC dropped below floor of {auc_floor}",
        ),
        AlarmSpec(
            name=f"Readmit-{endpoint_name}-Custom-AUC-Floor",
            namespace=CUSTOM_NAMESPACE,
            metric_name="AUC",
            dimensions=[{"Name": "Endpoint", "Value": endpoint_name}],
            statistic="Average",
            threshold=auc_floor,
            comparison="LessThanThreshold",
            period=86400, evaluation_periods=1,
            description="Custom-published AUC below threshold (offline eval lambda)",
        ),
    ]

    for spec in specs:
        _put_alarm(cw, spec, topic_arn)
    return topic_arn


# ---------------------------------------------------------------------------
# Custom-metric publisher (called from an offline scheduled Lambda)
# ---------------------------------------------------------------------------

def publish_offline_auc(
    endpoint_name: str,
    auc: float,
    region: str = "us-east-1",
    namespace: str = CUSTOM_NAMESPACE,
) -> None:
    """Publish a custom AUC metric so an alarm can react to offline eval runs.

    Typical flow: every night a Lambda joins captured predictions with the
    care-team's confirmed readmission outcomes, computes AUC, and calls this
    function. If AUC < floor, the alarm above fires and notifies SNS.
    """
    import boto3

    cw = boto3.client("cloudwatch", region_name=region)
    cw.put_metric_data(
        Namespace=namespace,
        MetricData=[
            {
                "MetricName": "AUC",
                "Dimensions": [{"Name": "Endpoint", "Value": endpoint_name}],
                "Value": float(auc),
                "Unit": "None",
            }
        ],
    )
    logger.info("Published %s/AUC=%.4f for %s", namespace, auc, endpoint_name)


# ---------------------------------------------------------------------------
# CloudWatch dashboard
# ---------------------------------------------------------------------------

def build_dashboard(
    endpoint_name: str,
    variant_name: str = "AllTraffic",
    region: str = "us-east-1",
    dashboard_name: str | None = None,
) -> str:
    """Create / update a CloudWatch dashboard for the endpoint.

    The dashboard puts the model-ops view in one place:

    * Invocation volume + p50/p95/p99 latency
    * 4xx / 5xx invocation errors
    * Endpoint CPU / Memory utilisation
    * Model Monitor data-drift violation count
    * SageMaker Model Monitor AUC (joined ground truth)
    * Custom ReadmitModel/AUC (offline eval)

    Returns the dashboard name.
    """
    import boto3

    name = dashboard_name or DASHBOARD_NAME_TEMPLATE.format(endpoint=endpoint_name)
    cw = boto3.client("cloudwatch", region_name=region)

    endpoint_dim = [
        ["EndpointName", endpoint_name],
        ["VariantName", variant_name],
    ]

    body = {
        "widgets": [
            {
                "type": "text",
                "x": 0, "y": 0, "width": 24, "height": 2,
                "properties": {
                    "markdown": (
                        f"# Readmit Risk — {endpoint_name}\n"
                        "AAI-540 Group 2 · ClearPath Health Analytics\n"
                        "_Operational view: latency, errors, drift, AUC._"
                    )
                },
            },
            {
                "type": "metric",
                "x": 0, "y": 2, "width": 12, "height": 6,
                "properties": {
                    "title": "Latency (ms)",
                    "region": region,
                    "view": "timeSeries", "stacked": False,
                    "stat": "Average", "period": 60,
                    "metrics": [
                        ["AWS/SageMaker", "ModelLatency", *sum(endpoint_dim, []),
                         {"stat": "p50", "label": "p50"}],
                        ["...", {"stat": "p95", "label": "p95"}],
                        ["...", {"stat": "p99", "label": "p99"}],
                    ],
                },
            },
            {
                "type": "metric",
                "x": 12, "y": 2, "width": 12, "height": 6,
                "properties": {
                    "title": "Invocations + Errors",
                    "region": region,
                    "view": "timeSeries", "stacked": False,
                    "stat": "Sum", "period": 300,
                    "metrics": [
                        ["AWS/SageMaker", "Invocations", *sum(endpoint_dim, [])],
                        ["AWS/SageMaker", "Invocation4XXErrors", *sum(endpoint_dim, [])],
                        ["AWS/SageMaker", "Invocation5XXErrors", *sum(endpoint_dim, [])],
                    ],
                },
            },
            {
                "type": "metric",
                "x": 0, "y": 8, "width": 12, "height": 6,
                "properties": {
                    "title": "Endpoint CPU / Memory",
                    "region": region,
                    "view": "timeSeries", "stacked": False,
                    "stat": "Average", "period": 300,
                    "metrics": [
                        ["/aws/sagemaker/Endpoints", "CPUUtilization", *sum(endpoint_dim, [])],
                        ["/aws/sagemaker/Endpoints", "MemoryUtilization", *sum(endpoint_dim, [])],
                    ],
                },
            },
            {
                "type": "metric",
                "x": 12, "y": 8, "width": 12, "height": 6,
                "properties": {
                    "title": "Data-quality drift violations (hourly)",
                    "region": region,
                    "view": "timeSeries", "stacked": False,
                    "stat": "Sum", "period": 3600,
                    "metrics": [
                        [
                            "aws/sagemaker/Endpoints/data-metrics",
                            "feature_baseline_drift_check_violations",
                            "EndpointName", endpoint_name,
                        ],
                    ],
                },
            },
            {
                "type": "metric",
                "x": 0, "y": 14, "width": 12, "height": 6,
                "properties": {
                    "title": "Model Monitor — AUC (joined GT)",
                    "region": region,
                    "view": "timeSeries", "stacked": False,
                    "stat": "Average", "period": 86400,
                    "metrics": [
                        [
                            "aws/sagemaker/Endpoints/model-metrics",
                            "auc", "EndpointName", endpoint_name,
                        ],
                    ],
                    "yAxis": {"left": {"min": 0, "max": 1}},
                    "annotations": {
                        "horizontal": [
                            {"label": "AUC alert floor",
                             "value": AUC_THRESHOLD_ALERT, "color": "#d62728"}
                        ]
                    },
                },
            },
            {
                "type": "metric",
                "x": 12, "y": 14, "width": 12, "height": 6,
                "properties": {
                    "title": "Custom ReadmitModel/AUC (offline eval)",
                    "region": region,
                    "view": "timeSeries", "stacked": False,
                    "stat": "Average", "period": 86400,
                    "metrics": [
                        [CUSTOM_NAMESPACE, "AUC", "Endpoint", endpoint_name],
                    ],
                    "yAxis": {"left": {"min": 0, "max": 1}},
                    "annotations": {
                        "horizontal": [
                            {"label": "AUC alert floor",
                             "value": AUC_THRESHOLD_ALERT, "color": "#d62728"}
                        ]
                    },
                },
            },
        ]
    }

    cw.put_dashboard(DashboardName=name, DashboardBody=json.dumps(body))
    logger.info("Dashboard upserted: %s", name)
    return name
