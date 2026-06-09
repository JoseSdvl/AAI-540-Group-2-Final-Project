"""SageMaker Pipelines definition for the readmission model.

Builds a 5-step pipeline:

    1. Preprocess  (SKLearnProcessor) — emits train/test/val parquet
    2. Train       (SKLearn estimator running readmit/models/train.py)
    3. Evaluate    (SKLearnProcessor scoring the model on the *test* split)
    4. Condition   — gate on AUC >= AUC_THRESHOLD_DEPLOY
    5. Register    — register the model in the SageMaker Model Registry
                     with status PendingManualApproval (production gate)

Usage (from a notebook or CI):

    from readmit.pipeline.sagemaker_pipeline import build_pipeline
    pipeline = build_pipeline(role_arn="arn:aws:iam::...:role/ReadmitRole")
    pipeline.upsert(role_arn="arn:aws:iam::...:role/ReadmitRole")
    pipeline.start()
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Import locally inside the builder so the rest of the package (tests, ingest,
# training script) does not require the sagemaker SDK to be installed.


def build_pipeline(
    role_arn: str,
    *,
    pipeline_name: str = "ReadmitRiskPipeline",
    model_package_group: str = "ReadmitRiskModels",
    source_dir: str = "src",
    data_source: str = "synthetic",
    s3_data_uri: Optional[str] = None,
    n_patients: int = 100_000,
    region: str = "us-east-1",
    training_instance: str = "ml.m5.xlarge",
    processing_instance: str = "ml.m5.large",
    auc_threshold: float = 0.75,
):
    """Construct (but do not run) the SageMaker Pipeline."""
    import sagemaker
    from sagemaker.processing import ProcessingInput, ProcessingOutput
    from sagemaker.sklearn.estimator import SKLearn
    from sagemaker.sklearn.processing import SKLearnProcessor
    from sagemaker.workflow.condition_step import ConditionStep
    from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
    from sagemaker.workflow.functions import JsonGet
    from sagemaker.workflow.model_step import ModelStep
    from sagemaker.workflow.parameters import ParameterInteger, ParameterString
    from sagemaker.workflow.pipeline import Pipeline
    from sagemaker.workflow.pipeline_context import PipelineSession
    from sagemaker.workflow.properties import PropertyFile
    from sagemaker.workflow.step_collections import RegisterModel
    from sagemaker.workflow.steps import ProcessingStep, TrainingStep

    session = PipelineSession()

    # ---- Pipeline parameters --------------------------------------------
    p_data_source = ParameterString(name="DataSource", default_value=data_source)
    p_s3_uri = ParameterString(name="S3DataUri", default_value=s3_data_uri or "")
    p_n_patients = ParameterInteger(name="NPatients", default_value=n_patients)
    p_model_approval = ParameterString(
        name="ModelApprovalStatus", default_value="PendingManualApproval"
    )

    # ---- 1) Preprocess --------------------------------------------------
    sklearn_processor = SKLearnProcessor(
        framework_version="1.2-1",
        role=role_arn,
        instance_type=processing_instance,
        instance_count=1,
        base_job_name="readmit-preprocess",
        sagemaker_session=session,
    )

    preprocess_args = [
        "--source", p_data_source,
        "--s3-uri", p_s3_uri,
        "--n-patients", p_n_patients.to_string(),
    ]

    preprocess_step = ProcessingStep(
        name="Preprocess",
        processor=sklearn_processor,
        code=f"{source_dir}/readmit/pipeline/preprocess_job.py",
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
            ProcessingOutput(output_name="test", source="/opt/ml/processing/test"),
            ProcessingOutput(output_name="validation",
                             source="/opt/ml/processing/validation"),
        ],
        job_arguments=preprocess_args,
    )

    # ---- 2) Train -------------------------------------------------------
    estimator = SKLearn(
        entry_point="readmit/models/train.py",
        source_dir=source_dir,
        role=role_arn,
        instance_type=training_instance,
        instance_count=1,
        framework_version="1.2-1",
        py_version="py3",
        base_job_name="readmit-train",
        hyperparameters={
            "model-kind": "xgboost",
            "learning-rate": 0.08,
            "max-depth": 6,
            "n-estimators": 400,
        },
        sagemaker_session=session,
    )

    train_step = TrainingStep(
        name="Train",
        estimator=estimator,
        inputs={
            "train": sagemaker.inputs.TrainingInput(
                s3_data=preprocess_step.properties.ProcessingOutputConfig.Outputs[
                    "train"
                ].S3Output.S3Uri,
                content_type="application/x-parquet",
            ),
            "validation": sagemaker.inputs.TrainingInput(
                s3_data=preprocess_step.properties.ProcessingOutputConfig.Outputs[
                    "validation"
                ].S3Output.S3Uri,
                content_type="application/x-parquet",
            ),
        },
    )

    # ---- 3) Evaluate ----------------------------------------------------
    evaluation_report = PropertyFile(
        name="EvaluationReport", output_name="evaluation", path="evaluation.json"
    )

    evaluate_step = ProcessingStep(
        name="Evaluate",
        processor=sklearn_processor,
        code=f"{source_dir}/readmit/pipeline/evaluate_job.py",
        inputs=[
            ProcessingInput(
                source=train_step.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=preprocess_step.properties.ProcessingOutputConfig.Outputs[
                    "test"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=[
            ProcessingOutput(output_name="evaluation",
                             source="/opt/ml/processing/evaluation"),
        ],
        property_files=[evaluation_report],
    )

    # ---- 4) Register (under condition) ----------------------------------
    register_step = RegisterModel(
        name="RegisterModel",
        estimator=estimator,
        model_data=train_step.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["application/json", "text/csv"],
        response_types=["application/json", "text/csv"],
        inference_instances=["ml.m5.large", "ml.m5.xlarge"],
        transform_instances=["ml.m5.xlarge"],
        model_package_group_name=model_package_group,
        approval_status=p_model_approval,
    )

    auc_condition = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=evaluate_step.name,
            property_file=evaluation_report,
            json_path="binary_classification_metrics.auc.value",
        ),
        right=auc_threshold,
    )

    gate_step = ConditionStep(
        name="AUCGate",
        conditions=[auc_condition],
        if_steps=[register_step],
        else_steps=[],
    )

    # ---- Assemble -------------------------------------------------------
    return Pipeline(
        name=pipeline_name,
        parameters=[p_data_source, p_s3_uri, p_n_patients, p_model_approval],
        steps=[preprocess_step, train_step, evaluate_step, gate_step],
        sagemaker_session=session,
    )
