"""
Training pipeline orchestration.

The workflow: load the Telco churn dataset (ARFF), preprocess, 3-way split,
grid-search with cross-validation, evaluate on the validation set, tune the
classification threshold, evaluate the held-out test set at that threshold,
compute drift-detection reference statistics, and save the versioned model,
metadata, config and diagnostics — with every run tracked in MLflow.

This module owns configuration loading, data preparation, and model fitting,
and delegates the rest: evaluation/threshold tuning/diagnostics to
``reporting``, artifact publication to ``artifacts``, and MLflow model
logging to ``mlflow_logging``. The ``train.py`` CLI shim calls ``main()``.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from ..config import TrainingConfig
from ..data import create_three_way_split, load_data, preprocess_data
from ..evaluation.metrics import compute_metrics
from ..models import build_pipeline
from ..models.model_factory import get_model, get_model_display_name, get_param_grid
from ..utils.logger import setup_logger
from . import perform_grid_search
from .artifacts import compute_reference_statistics, save_training_artifacts
from .mlflow_logging import log_to_mlflow
from .reporting import (
    evaluate_and_analyze,
    evaluate_test_set,
    print_training_summary,
    tune_thresholds,
)

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logger = setup_logger(name="churn_training", log_level=LOG_LEVEL, log_file="logs/training.log")


def load_configuration(config_path: Optional[str] = None, use_env: bool = False) -> TrainingConfig:
    """
    Load and validate training configuration.

    Args:
        config_path: Path to config file (YAML or JSON). If None, uses defaults.
        use_env: Load configuration from environment variables

    Returns:
        Validated TrainingConfig object

    Raises:
        ValueError: If config file has unsupported extension
    """
    if config_path:
        logger.info(f"Loading configuration from: {config_path}")
        if config_path.endswith((".yaml", ".yml")):
            config = TrainingConfig.from_yaml(config_path)
        elif config_path.endswith(".json"):
            config = TrainingConfig.from_json(config_path)
        else:
            raise ValueError("Config file must be .yaml, .yml, or .json")
    elif use_env:
        logger.info("Loading configuration from environment variables")
        config = TrainingConfig.from_env()
    else:
        logger.info("Using default configuration")
        config = TrainingConfig()

    logger.info("Configuration loaded successfully")
    logger.debug(f"Config: {config.to_dict()}")

    return config


def setup_experiment(config: TrainingConfig, run_id: str, config_source: str) -> None:
    """
    Configure MLflow experiment and start tracking run.

    Args:
        config: Training configuration
        run_id: Unique run identifier (timestamp-based)
        config_source: Source of config ('file', 'env', or 'default')
    """
    # Configure MLflow (always enabled)
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment(config.mlflow.experiment_name)
    logger.info(f"MLflow tracking enabled: {config.mlflow.tracking_uri}")
    logger.info(f"Experiment: {config.mlflow.experiment_name}")

    # Start MLflow run
    mlflow.start_run()

    # Log tags
    mlflow.set_tag("run_id", run_id)
    mlflow.set_tag("stage", "validation")
    mlflow.set_tag("model_type", config.model.model_type)
    mlflow.set_tag("config_source", config_source)

    logger.info("MLflow run started")


def load_and_prepare_data(
    config: TrainingConfig,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
    pd.DataFrame,
    pd.Series,
]:
    """
    Load dataset, preprocess, and create train/validation/test splits.

    Args:
        config: Training configuration with data settings

    Returns:
        Tuple: (X_train, X_val, X_test, y_train, y_val, y_test, X, y)
    """
    # Load dataset
    logger.info(f"Loading dataset from {config.data.data_path}")
    df = load_data(config.data.data_path)
    logger.info(f"Loaded {len(df)} records")
    logger.debug(f"Dataset columns: {df.columns.tolist()}")

    # Preprocess
    logger.info("Preprocessing data")
    X, y = preprocess_data(df)
    logger.info(f"Preprocessed features: {X.shape[1]} columns, {X.shape[0]} rows")

    # Create 3-way split
    logger.info("Creating train/validation/test splits")
    X_train, X_val, X_test, y_train, y_val, y_test = create_three_way_split(
        X,
        y,
        test_size=config.data.test_size,
        val_size=config.data.val_size,
        random_state=config.data.random_state,
        stratify=config.data.stratify,
    )
    logger.info(f"Train: {len(X_train)} samples")
    logger.info(f"Validation: {len(X_val)} samples")
    logger.info(f"Test: {len(X_test)} samples")
    logger.debug(f"Class distribution - Train: {y_train.value_counts().to_dict()}")

    # Log data parameters to MLflow
    mlflow.log_param("data_path", config.data.data_path)
    mlflow.log_param("n_total_samples", len(X))
    mlflow.log_param("n_features", X.shape[1])
    mlflow.log_param("n_train", len(X_train))
    mlflow.log_param("n_val", len(X_val))
    mlflow.log_param("n_test", len(X_test))
    mlflow.log_param("test_size", config.data.test_size)
    mlflow.log_param("val_size", config.data.val_size)
    mlflow.log_param("random_state", config.data.random_state)
    mlflow.log_param("stratify", config.data.stratify)

    return X_train, X_val, X_test, y_train, y_val, y_test, X, y


def resolve_param_grid(config: TrainingConfig) -> tuple[dict[str, Any], str]:
    """
    Select the hyperparameter grid for the configured model type.

    Random Forest grids come from the training config (the config schema exposes
    RF grid fields, so ``--quick`` and custom configs are honored). XGBoost and
    Logistic Regression have no config grid fields, so their model-specific
    factory grids are used.

    Args:
        config: Training configuration with the resolved model type.

    Returns:
        Tuple of (param_grid, grid_source) where grid_source is "config" or
        "factory".
    """
    if config.model.model_type == "random_forest":
        return config.model.to_param_grid(), "config"
    return get_param_grid(config.model.model_type), "factory"


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: ColumnTransformer,
    numeric_features: list[str],
    categorical_features: list[str],
    config: TrainingConfig,
) -> tuple[Pipeline, dict[str, Any], float, float]:
    """
    Perform hyperparameter tuning via grid search and train best model.

    Args:
        X_train: Training features
        y_train: Training target
        preprocessor: Preprocessing pipeline
        numeric_features: List of numeric feature names
        categorical_features: List of categorical feature names
        config: Training configuration with model settings

    Returns:
        Tuple: (best_model, best_params, search_time, best_score)
    """
    logger.info("Building preprocessing pipeline")
    logger.info(f"{len(numeric_features)} numeric features")
    logger.info(f"{len(categorical_features)} categorical features")
    logger.debug(f"Numeric features: {numeric_features}")
    logger.debug(f"Categorical features: {categorical_features}")

    # Log feature counts and model type
    mlflow.log_param("n_numeric_features", len(numeric_features))
    mlflow.log_param("n_categorical_features", len(categorical_features))
    mlflow.log_param("model_type", config.model.model_type)

    # Instantiate the model and resolve its hyperparameter grid (config-driven
    # for random_forest, factory grid for xgboost/logistic_regression)
    model = get_model(config.model.model_type, random_state=config.data.random_state)
    param_grid, grid_source = resolve_param_grid(config)
    n_combinations = int(np.prod([len(v) for v in param_grid.values()]))

    model_display_name = get_model_display_name(config.model.model_type)
    logger.info(f"Training {model_display_name} model")
    logger.info(f"Hyperparameter grid source: {grid_source} ({n_combinations} combinations)")
    logger.info(f"Hyperparameter grid: {param_grid}")
    mlflow.log_param("grid_source", grid_source)
    mlflow.log_param("grid_combinations", n_combinations)

    # Hyperparameter tuning with grid search
    logger.info(
        f"Starting hyperparameter grid search with {config.model.cv_folds}-fold cross-validation"
    )
    logger.info("This may take several minutes...")
    grid_search, best_params, search_time = perform_grid_search(
        X_train,
        y_train,
        preprocessor,
        model=model,
        param_grid=param_grid,
        cv_folds=config.model.cv_folds,
        scoring=config.model.scoring,
        random_state=config.data.random_state,
        n_jobs=config.model.n_jobs,
    )
    best_model = grid_search.best_estimator_
    logger.info(f"Grid search complete in {search_time:.2f} seconds")
    logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    logger.info(f"Best parameters: {best_params}")

    # Log hyperparameters and CV score
    mlflow.log_param("cv_folds", config.model.cv_folds)
    mlflow.log_param("scoring", config.model.scoring)
    mlflow.log_param("n_jobs", config.model.n_jobs)
    mlflow.log_param("search_time_seconds", search_time)
    for param, value in best_params.items():
        mlflow.log_param(param, value)
    mlflow.log_metric("cv_roc_auc", grid_search.best_score_)
    logger.info("Best hyperparameters and CV score logged to MLflow")

    return best_model, best_params, search_time, grid_search.best_score_


def main(
    config_path: Optional[str] = None, use_env: bool = False, model_type: Optional[str] = None
) -> None:
    """
    Main training workflow orchestrator.

    Args:
        config_path: Path to config file (YAML or JSON). If None, uses defaults.
        use_env: Load configuration from environment variables
        model_type: Model type to train (random_forest, xgboost, logistic_regression). Overrides config.
    """
    logger.info("=" * 70)
    logger.info("CHURN PREDICTION MODEL TRAINING")
    logger.info("=" * 70)

    # Setup directories
    models_dir = Path("models")
    diagnostics_dir = Path("diagnostics")
    configs_dir = Path("configs")
    models_dir.mkdir(exist_ok=True)
    diagnostics_dir.mkdir(exist_ok=True)
    configs_dir.mkdir(exist_ok=True)

    # Generate run ID
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"Run ID: {run_id}")

    # Determine config source for tagging
    config_source = "file" if config_path else ("env" if use_env else "default")

    try:
        # 1. Load configuration
        config = load_configuration(config_path, use_env)

        # Override model_type if provided via CLI
        if model_type:
            logger.info(f"Overriding model type from CLI: {model_type}")
            config.model.model_type = model_type

        # 2. Setup MLflow experiment and start tracking
        setup_experiment(config, run_id, config_source)

        # 3. Load and prepare data (includes MLflow logging)
        X_train, X_val, X_test, y_train, y_val, y_test, X, y = load_and_prepare_data(config)

        # 4. Build preprocessing pipeline
        preprocessor, numeric_features, categorical_features = build_pipeline()

        # 5. Train model (grid search + hyperparameter tuning)
        best_model, best_params, search_time, best_score = train_model(
            X_train, y_train, preprocessor, numeric_features, categorical_features, config
        )

        # 6. Evaluate and analyze (metrics, plots, feature importance)
        metrics, y_val_pred, y_val_proba, feature_importance_df = evaluate_and_analyze(
            best_model, X_train, X_val, y_val, numeric_features, categorical_features, run_id
        )

        # 7. Tune classification thresholds
        threshold_info = tune_thresholds(y_val, y_val_proba, run_id)
        tuned_threshold = threshold_info["chosen_threshold"]

        # 7b. Re-evaluate the validation set at the tuned threshold. The step-6
        # metrics use the default 0.5 cut, so the two sets of numbers must be
        # labeled apart wherever they are reported.
        y_val_pred_tuned = (y_val_proba >= tuned_threshold).astype(int)
        metrics_val_tuned = compute_metrics(y_val, y_val_pred_tuned, y_val_proba)
        logger.info(f"\nValidation metrics at tuned threshold ({tuned_threshold:.4f}):")
        logger.info(f"  Precision: {metrics_val_tuned['precision']:.4f}")
        logger.info(f"  Recall:    {metrics_val_tuned['recall']:.4f}")
        logger.info(f"  F1 Score:  {metrics_val_tuned['f1']:.4f}")
        for metric_name, metric_value in metrics_val_tuned.items():
            if isinstance(metric_value, (int, float)):
                mlflow.log_metric(f"val_tuned_{metric_name}", metric_value)
        logger.info("Tuned-threshold validation metrics logged to MLflow")

        # 8. Evaluate on held-out test set
        test_metrics = evaluate_test_set(best_model, X_test, y_test, tuned_threshold)

        # 9. Compute reference statistics for drift detection
        logger.info("\nComputing reference statistics for drift detection")
        reference_stats = compute_reference_statistics(
            X_train,
            y_train,
            numeric_features,
            categorical_features,
            model=best_model,
            threshold=tuned_threshold,
        )
        logger.info(
            f"Reference statistics computed ({len(reference_stats['numeric'])} numeric, "
            f"{len(reference_stats['categorical'])} categorical features)"
        )

        # 10. Save training artifacts (model, metadata, config)
        versioned_model_path, latest_model_path, metadata_path, run_config_path = (
            save_training_artifacts(
                best_model,
                best_params,
                metrics,
                test_metrics,
                search_time,
                numeric_features,
                categorical_features,
                reference_stats,
                threshold_info,
                config,
                run_id,
            )
        )

        # 11. Log to MLflow (model, signature, artifacts)
        mlflow_run_id = log_to_mlflow(
            best_model, X_val, y_val_proba, metadata_path, run_config_path, config
        )

        # 12. Print training summary
        print_training_summary(
            metrics,
            test_metrics,
            threshold_info,
            feature_importance_df,
            config,
            run_id,
            mlflow_run_id,
            metrics_val_tuned=metrics_val_tuned,
        )

    except Exception:
        # Record the run as FAILED — the default end_run() status would mark a
        # crashed training run FINISHED in MLflow.
        if mlflow.active_run():
            mlflow.end_run(status="FAILED")
            logger.info("MLflow run ended (FAILED)")
        raise
    else:
        if mlflow.active_run():
            mlflow.end_run()
            logger.info("MLflow run ended")
