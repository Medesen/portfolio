"""
MLflow model logging and optional registry registration.

Separated from the pipeline orchestrator because it owns MLflow specifics:
inferring the model signature from validation data, choosing the pyfunc
predict function, and registering the model when MLFLOW_REGISTER_MODEL is set.
"""

import os
from pathlib import Path
from typing import Any, Optional

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.pipeline import Pipeline

from ..config import TrainingConfig
from ..utils.logger import get_logger

logger = get_logger("churn_training")


def log_to_mlflow(
    best_model: Pipeline,
    X_val: pd.DataFrame,
    y_val_proba: "np.ndarray[Any, Any]",
    metadata_path: str,
    run_config_path: "str | Path",
    config: TrainingConfig,
) -> Optional[str]:
    """
    Log model and artifacts to MLflow.

    Args:
        best_model: Trained model pipeline
        X_val: Validation features (for signature and input example)
        y_val_proba: Validation predicted probabilities (for signature)
        metadata_path: Path to metadata JSON file
        run_config_path: Path to run configuration YAML file
        config: Training configuration

    Returns:
        MLflow run ID (if active run exists)
    """
    logger.info("Logging model to MLflow")

    # Create model signature (input/output schema)
    # Output should match predict_proba output: 2D array with shape (n_samples, 2)
    # This represents [prob_class_0, prob_class_1] for each sample
    # Note: The API layer (src/api) converts this to PredictionResponse with additional fields
    y_val_proba_full = best_model.predict_proba(X_val)  # Shape: (n_samples, 2)
    logger.debug(
        f"Creating signature: input shape {X_val.shape}, output shape {y_val_proba_full.shape}"
    )
    signature = infer_signature(X_val, y_val_proba_full)
    logger.info(
        f"Model signature created: {len(signature.inputs.inputs)} inputs, output schema with 2 probability columns"
    )

    # Determine if we should register the model
    register_model = os.getenv("MLFLOW_REGISTER_MODEL", "false").lower() == "true"
    registered_model_name = "churn_prediction_model" if register_model else None

    # Log model (with optional registration). pyfunc_predict_fn="predict_proba"
    # makes the pyfunc flavor serve probabilities, matching the signature
    # inferred above — without it, pyfunc predict() would return class labels
    # while the logged signature promises two probability columns.
    if config.mlflow.log_models:
        model_info = mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            signature=signature,
            input_example=X_val.iloc[:5],
            registered_model_name=registered_model_name,  # Register if flag set
            pyfunc_predict_fn="predict_proba",
        )
        logger.info("Model logged to MLflow")

        # If registered, log registration details
        if register_model and hasattr(model_info, "registered_model_version"):
            logger.info("=" * 60)
            logger.info("MODEL REGISTRY")
            logger.info("=" * 60)
            logger.info(f"Model registered: {registered_model_name}")
            logger.info(f"Version: {model_info.registered_model_version}")
            logger.info("Stage: None (use promotion script to set stage)")
            logger.info("=" * 60)

    # Log artifacts to MLflow
    if config.mlflow.log_artifacts:
        mlflow.log_artifact(metadata_path, "metadata")
        mlflow.log_artifact(str(run_config_path), "config")
        logger.info("Configuration and metadata logged to MLflow")

    # Get MLflow run ID
    mlflow_run_id = None
    if mlflow.active_run():
        mlflow_run_id = mlflow.active_run().info.run_id
        logger.info(f"MLflow run ID: {mlflow_run_id}")

    return mlflow_run_id
