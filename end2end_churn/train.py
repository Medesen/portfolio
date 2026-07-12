"""
Training script for customer churn prediction model.

Usage (via Docker/Make - RECOMMENDED):
    make train                  # Train with default config
    make train-quick            # Train with quick config (fast)
    make train-prod             # Train with production config (thorough)
    make train-rf               # Train Random Forest specifically
    make train-xgboost          # Train XGBoost specifically
    make train-logreg           # Train Logistic Regression specifically
    make train-register         # Train and register in MLflow Registry

Direct usage (for development only):
    python train.py [--config CONFIG_PATH] [--model-type MODEL_TYPE]

This script orchestrates the complete training workflow:
1. Loads the Telco Customer Churn dataset (ARFF format)
2. Preprocesses numeric and categorical features
3. Creates 3-way split (train/validation/test)
4. Performs hyperparameter tuning with grid search + cross-validation
5. Evaluates best model comprehensively on validation set
6. Generates diagnostic visualizations and feature importances
7. Saves model with versioning and metadata

Features:
- Hyperparameter Tuning & Model Versioning
  * Grid search over model parameters
  * 5-fold cross-validation on training set
  * Model versioning with timestamps and run IDs
  * Feature importance analysis and visualization
  * Metadata tracking (params, metrics, timings)
  * Modular code structure with src/ package

- Structured Logging
  * Structured logging to console and file (logs/training.log)
  * Different log levels for better observability

- Drift Detection
  * Compute reference statistics from training data
  * Save baseline distributions for monitoring
  * Enable drift detection in production API

- MLflow Experiment Tracking
  * Track all training experiments with MLflow
  * Log parameters, metrics, and artifacts
  * Model signatures for input/output validation
  * Compare runs in MLflow UI
  * Model lineage and reproducibility

- Configuration Management
  * Pydantic-based type-safe configuration
  * Load configs from YAML, JSON, or environment variables
  * Separate configs for dev, quick testing, and production
  * Save config with each training run for reproducibility

- Threshold Tuning Strategies
  * Tune classification threshold beyond default 0.5
  * Multiple strategies: F1 maximization, precision-constrained, top-k, cost-sensitive
  * Threshold analysis visualizations
  * Save threshold strategies in model metadata

- Model Registry & Versioning
  * Register models in MLflow Registry
  * Model stages: None → Staging → Production → Archived
  * Enable loading models from registry by stage
  * Model lifecycle management

- Code Quality
  * Decomposed main() function into focused sub-functions
  * Each function has single responsibility
  * Improved testability and maintainability
  * Clear separation of concerns
"""

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.config import TrainingConfig

# Import from our modular src package
from src.data import create_three_way_split, load_data, preprocess_data
from src.evaluation import (
    plot_confusion_matrix,
    plot_feature_importances,
    plot_precision_recall_curve,
    plot_roc_curve,
)
from src.evaluation.threshold import (
    evaluate_threshold,
    tune_threshold_cost_sensitive,
    tune_threshold_f1,
    tune_threshold_precision_constrained_recall,
    tune_threshold_top_k,
)
from src.models import build_pipeline
from src.training import evaluate_model, extract_feature_importances, perform_grid_search
from src.utils import save_diagnostics_report, save_metadata, save_model
from src.utils.logger import setup_logger

# Initialize logger
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logger = setup_logger(name="churn_training", log_level=LOG_LEVEL, log_file="logs/training.log")


# ==============================================================================
# TRAINING WORKFLOW SUB-FUNCTIONS
# ==============================================================================


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
    logger.info(f"✓ MLflow tracking enabled: {config.mlflow.tracking_uri}")
    logger.info(f"✓ Experiment: {config.mlflow.experiment_name}")

    # Start MLflow run
    mlflow.start_run()

    # Log tags
    mlflow.set_tag("run_id", run_id)
    mlflow.set_tag("stage", "validation")
    mlflow.set_tag("model_type", config.model.model_type)
    mlflow.set_tag("config_source", config_source)

    logger.info("✓ MLflow run started")


def load_and_prepare_data(
    config: TrainingConfig,
) -> Tuple[
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


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: ColumnTransformer,
    numeric_features: List[str],
    categorical_features: List[str],
    config: TrainingConfig,
) -> Tuple[Pipeline, Dict, float]:
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
    from src.models.model_factory import get_model, get_model_display_name, get_param_grid

    logger.info("Building preprocessing pipeline")
    logger.info(f"✓ {len(numeric_features)} numeric features")
    logger.info(f"✓ {len(categorical_features)} categorical features")
    logger.debug(f"Numeric features: {numeric_features}")
    logger.debug(f"Categorical features: {categorical_features}")

    # Log feature counts and model type
    mlflow.log_param("n_numeric_features", len(numeric_features))
    mlflow.log_param("n_categorical_features", len(categorical_features))
    mlflow.log_param("model_type", config.model.model_type)

    # Get model and param grid from factory
    model = get_model(config.model.model_type, random_state=config.data.random_state)
    param_grid = get_param_grid(config.model.model_type)

    model_display_name = get_model_display_name(config.model.model_type)
    logger.info(f"Training {model_display_name} model")
    logger.info(f"Hyperparameter grid: {param_grid}")

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
    logger.info("✓ Best hyperparameters and CV score logged to MLflow")

    return best_model, best_params, search_time, grid_search.best_score_


def evaluate_and_analyze(
    best_model: Pipeline,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    numeric_features: List[str],
    categorical_features: List[str],
    run_id: str,
) -> Tuple[Dict, np.ndarray, pd.DataFrame]:
    """
    Evaluate model on validation set and generate diagnostic artifacts.

    Args:
        best_model: Trained model pipeline
        X_train: Training features (for feature importance)
        X_val: Validation features
        y_val: Validation target
        numeric_features: List of numeric feature names
        categorical_features: List of categorical feature names
        run_id: Unique run identifier for file naming

    Returns:
        Tuple: (metrics, y_val_pred, y_val_proba, feature_importance_df)
    """
    diagnostics_dir = Path("diagnostics")

    # Evaluate on validation set
    logger.info("Evaluating best model on validation set")
    metrics, y_val_pred, y_val_proba = evaluate_model(best_model, X_val, y_val)
    logger.info(f"Validation ROC AUC: {metrics['roc_auc']:.4f}")
    logger.info(f"Validation F1 Score: {metrics['f1']:.4f}")
    logger.debug(f"All metrics: {metrics}")

    # Log validation metrics
    for metric_name, metric_value in metrics.items():
        # Only log scalar values (skip nested dicts like confusion_matrix)
        if isinstance(metric_value, (int, float)):
            mlflow.log_metric(f"val_{metric_name}", metric_value)
    logger.info("✓ Validation metrics logged to MLflow")

    # Extract feature importances
    logger.info("Extracting feature importances")
    feature_importance_df = extract_feature_importances(
        best_model, numeric_features, categorical_features, X_train
    )
    logger.debug(f"Top 5 features: {feature_importance_df.head(5)['feature'].tolist()}")

    # Generate diagnostics
    logger.info("Generating diagnostic visualizations")

    cm_path = str(diagnostics_dir / f"confusion_matrix_{run_id}.png")
    plot_confusion_matrix(
        y_val, y_val_pred, cm_path, title="Confusion Matrix (Validation Set, threshold = 0.5)"
    )
    logger.info("✓ Confusion matrix saved")

    roc_path = str(diagnostics_dir / f"roc_curve_{run_id}.png")
    plot_roc_curve(y_val, y_val_proba, roc_path, title="ROC Curve (Validation Set)")
    logger.info("✓ ROC curve saved")

    pr_path = str(diagnostics_dir / f"precision_recall_curve_{run_id}.png")
    plot_precision_recall_curve(
        y_val, y_val_proba, pr_path, title="Precision-Recall Curve (Validation Set)"
    )
    logger.info("✓ PR curve saved")

    fi_path = str(diagnostics_dir / f"feature_importances_{run_id}.png")
    plot_feature_importances(feature_importance_df, fi_path, top_n=20)
    logger.info("✓ Feature importances plot saved")

    # Save diagnostics report
    logger.info("Saving diagnostics report")
    diag_path = str(diagnostics_dir / f"evaluation_report_{run_id}.txt")
    save_diagnostics_report(metrics, diag_path, set_name="Validation (default 0.5 threshold)")
    logger.info("✓ Evaluation report saved")

    # Save feature importances CSV
    fi_csv_path = diagnostics_dir / f"feature_importances_{run_id}.csv"
    feature_importance_df.to_csv(fi_csv_path, index=False)
    logger.info("✓ Feature importances CSV saved")

    # Log artifacts to MLflow
    mlflow.log_artifact(cm_path, "plots")
    mlflow.log_artifact(roc_path, "plots")
    mlflow.log_artifact(pr_path, "plots")
    mlflow.log_artifact(fi_path, "plots")
    mlflow.log_artifact(str(fi_csv_path), "data")
    mlflow.log_artifact(diag_path, "reports")
    logger.info("✓ Artifacts logged to MLflow")

    return metrics, y_val_pred, y_val_proba, feature_importance_df


def tune_thresholds(y_val: pd.Series, y_val_proba: np.ndarray, run_id: str) -> Dict[str, Any]:
    """
    Tune classification threshold using multiple strategies.

    Args:
        y_val: Validation target
        y_val_proba: Validation predicted probabilities
        run_id: Unique run identifier for file naming

    Returns:
        Dict with threshold information for all strategies
    """
    diagnostics_dir = Path("diagnostics")

    logger.info("=" * 60)
    logger.info("THRESHOLD TUNING")
    logger.info("=" * 60)

    # Strategy 1: F1 Maximization (default strategy)
    logger.info("\nStrategy 1: F1 Maximization")
    logger.info("  Goal: Balance precision and recall equally")
    threshold_f1, metrics_f1 = tune_threshold_f1(y_val, y_val_proba)
    logger.info(f"  Best F1 threshold: {threshold_f1:.4f}")
    logger.info(f"  Precision: {metrics_f1['precision']:.4f}")
    logger.info(f"  Recall:    {metrics_f1['recall']:.4f}")
    logger.info(f"  F1 Score:  {metrics_f1['f1']:.4f}")

    # Strategy 2: Precision-Constrained Recall
    logger.info("\nStrategy 2: Precision-Constrained Recall")
    logger.info("  Goal: Maximize recall while maintaining min precision (70%)")
    try:
        threshold_pcr, metrics_pcr = tune_threshold_precision_constrained_recall(
            y_val, y_val_proba, min_precision=0.70
        )
        logger.info(f"  Threshold (≥70% precision): {threshold_pcr:.4f}")
        logger.info(f"  Precision: {metrics_pcr['precision']:.4f}")
        logger.info(f"  Recall:    {metrics_pcr['recall']:.4f}")
        logger.info(f"  F1 Score:  {metrics_pcr['f1']:.4f}")
    except ValueError as e:
        logger.warning(f"  Could not achieve precision constraint: {e}")
        threshold_pcr, metrics_pcr = None, None

    # Strategy 3: Top-K Selection
    logger.info("\nStrategy 3: Top-K Selection")
    logger.info("  Goal: Target top 20% highest-risk customers")
    threshold_topk, metrics_topk = tune_threshold_top_k(y_val_proba, ratio=0.20)
    logger.info(f"  Threshold (top 20%): {threshold_topk:.4f}")
    logger.info(f"  Will flag: {metrics_topk['k']} customers ({metrics_topk['ratio']:.1%})")
    logger.info(f"  Actual flagged: {metrics_topk['n_flagged']}")

    # Strategy 4: Cost-Sensitive (example: FN cost 10x FP cost)
    logger.info("\nStrategy 4: Cost-Sensitive Optimization")
    logger.info("  Goal: Minimize expected cost (FN cost = 100, FP cost = 10)")
    logger.info("  Interpretation: Missing a churner costs 10x more than false alarm")
    threshold_cost, metrics_cost = tune_threshold_cost_sensitive(
        y_val, y_val_proba, cost_fn=100, cost_fp=10
    )
    logger.info(f"  Optimal threshold: {threshold_cost:.4f}")
    logger.info(f"  Expected cost: ${metrics_cost['expected_cost']:.2f}")
    logger.info(f"  Precision: {metrics_cost['precision']:.4f}")
    logger.info(f"  Recall:    {metrics_cost['recall']:.4f}")

    # Compare all strategies
    logger.info("\n" + "=" * 60)
    logger.info("THRESHOLD STRATEGY COMPARISON")
    logger.info("=" * 60)

    # Default 0.5
    logger.info("\nDefault (0.5):")
    metrics_default = evaluate_threshold(y_val, y_val_proba, 0.5)
    logger.info(
        f"  Precision: {metrics_default['precision']:.4f}, "
        + f"Recall: {metrics_default['recall']:.4f}, "
        + f"F1: {metrics_default['f1']:.4f}"
    )

    logger.info("\nF1 Maximization:")
    logger.info(f"  Threshold: {threshold_f1:.4f}")
    logger.info(
        f"  Precision: {metrics_f1['precision']:.4f}, "
        + f"Recall: {metrics_f1['recall']:.4f}, "
        + f"F1: {metrics_f1['f1']:.4f}"
    )

    if threshold_pcr:
        logger.info("\nPrecision-Constrained Recall:")
        logger.info(f"  Threshold: {threshold_pcr:.4f}")
        logger.info(
            f"  Precision: {metrics_pcr['precision']:.4f}, "
            + f"Recall: {metrics_pcr['recall']:.4f}, "
            + f"F1: {metrics_pcr['f1']:.4f}"
        )

    logger.info("\nTop-K (20%):")
    logger.info(f"  Threshold: {threshold_topk:.4f}")
    logger.info(f"  Flags {metrics_topk['k']} customers")

    logger.info("\nCost-Sensitive:")
    logger.info(f"  Threshold: {threshold_cost:.4f}")
    logger.info(f"  Expected cost: ${metrics_cost['expected_cost']:.2f}")

    # Choose default strategy: F1 maximization
    chosen_threshold = threshold_f1
    chosen_strategy = "f1_maximization"

    logger.info("\n" + "-" * 60)
    logger.info(f"CHOSEN STRATEGY: {chosen_strategy}")
    logger.info(f"CHOSEN THRESHOLD: {chosen_threshold:.4f}")
    logger.info("-" * 60)

    # Create threshold analysis plot
    threshold_plot_path = str(diagnostics_dir / f"threshold_analysis_{run_id}.png")
    plot_threshold_analysis(y_val, y_val_proba, threshold_plot_path, run_id)

    # Log threshold plot to MLflow
    mlflow.log_artifact(threshold_plot_path, "plots")
    logger.info("✓ Threshold analysis logged to MLflow")

    # Prepare threshold info for metadata
    threshold_info = {
        "default": 0.5,
        "strategies": {
            "f1_maximization": {
                "threshold": float(threshold_f1),
                "metrics": {
                    k: float(v) if isinstance(v, (int, float, np.number)) else v
                    for k, v in metrics_f1.items()
                },
            },
            "precision_constrained_recall": {
                "threshold": float(threshold_pcr) if threshold_pcr else None,
                "metrics": (
                    {
                        k: float(v) if isinstance(v, (int, float, np.number)) else v
                        for k, v in metrics_pcr.items()
                    }
                    if metrics_pcr
                    else None
                ),
            },
            "top_k": {
                "threshold": float(threshold_topk),
                "metrics": {
                    k: float(v) if isinstance(v, (int, float, np.number)) else v
                    for k, v in metrics_topk.items()
                },
            },
            "cost_sensitive": {
                "threshold": float(threshold_cost),
                "metrics": {
                    k: float(v) if isinstance(v, (int, float, np.number)) else v
                    for k, v in metrics_cost.items()
                },
            },
        },
        "chosen_strategy": chosen_strategy,
        "chosen_threshold": float(chosen_threshold),
    }

    logger.info("✓ Threshold strategies computed and saved")

    return threshold_info


def compute_reference_statistics(
    X: pd.DataFrame, y: pd.Series, numeric_features: List[str], categorical_features: List[str]
) -> Dict[str, Any]:
    """
    Compute reference statistics for drift detection.

    These statistics serve as the baseline for monitoring data drift
    in production. They are saved in the model metadata and used by
    the /drift endpoint to detect distribution shifts.

    Args:
        X: Training features
        y: Training target
        numeric_features: List of numeric feature names
        categorical_features: List of categorical feature names

    Returns:
        Dictionary with reference statistics
    """
    stats = {"numeric": {}, "categorical": {}, "target": {}}

    # Numeric feature statistics
    for col in numeric_features:
        if col in X.columns:
            values = X[col].dropna()
            if len(values) > 0:
                stats["numeric"][col] = {
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "median": float(values.median()),
                    "q25": float(values.quantile(0.25)),
                    "q75": float(values.quantile(0.75)),
                    "missing_rate": float(X[col].isna().mean()),
                    # Reference sample for the KS test in detect_numeric_drift
                    # (which is skipped without 'samples'); capped at 1000
                    # values to keep the model metadata small.
                    "samples": values.sample(min(len(values), 1000), random_state=42).tolist(),
                }

    # Categorical feature distributions (for PSI)
    for col in categorical_features:
        if col in X.columns:
            # Value counts as proportions
            value_counts = X[col].value_counts(normalize=True, dropna=False)
            stats["categorical"][col] = {
                "distribution": value_counts.to_dict(),
                "n_unique": int(X[col].nunique()),
                "missing_rate": float(X[col].isna().mean()),
            }

    # Target distribution
    stats["target"] = {"positive_rate": float(y.mean()), "n_samples": len(y)}

    return stats


def evaluate_test_set(
    best_model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, threshold: float
) -> Dict:
    """
    Evaluate model on held-out test set for final performance assessment.

    This represents true generalization performance on completely unseen data.
    The test set should only be evaluated once at the end of model development.

    Args:
        best_model: Trained model pipeline
        X_test: Test features
        y_test: Test target labels
        threshold: Tuned classification threshold from validation set

    Returns:
        Dictionary of test metrics
    """
    logger.info("\n" + "=" * 70)
    logger.info("EVALUATING ON HELD-OUT TEST SET")
    logger.info("=" * 70)

    # Get predictions and probabilities
    y_test_proba = best_model.predict_proba(X_test)[:, 1]

    # Apply tuned threshold (from validation set)
    y_test_pred = (y_test_proba >= threshold).astype(int)

    # Compute metrics using the same function as validation
    from src.evaluation.metrics import compute_metrics

    test_metrics = compute_metrics(y_test, y_test_pred, y_test_proba)

    logger.info(f"Test ROC AUC:       {test_metrics['roc_auc']:.4f}")
    logger.info(f"Test Accuracy:      {test_metrics['accuracy']:.4f}")
    logger.info(f"Test Precision:     {test_metrics['precision']:.4f}")
    logger.info(f"Test Recall:        {test_metrics['recall']:.4f}")
    logger.info(f"Test F1 Score:      {test_metrics['f1']:.4f}")
    logger.info(f"Test Avg Precision: {test_metrics['avg_precision']:.4f}")

    cm = test_metrics["confusion_matrix"]
    logger.info(f"\nTest Confusion Matrix:")
    logger.info(f"  TN={cm['tn']}, FP={cm['fp']}")
    logger.info(f"  FN={cm['fn']}, TP={cm['tp']}")

    # Log to MLflow
    for metric_name, metric_value in test_metrics.items():
        if isinstance(metric_value, (int, float)):
            mlflow.log_metric(f"test_{metric_name}", metric_value)
    logger.info("✓ Test metrics logged to MLflow")

    return test_metrics


def save_training_artifacts(
    best_model: Pipeline,
    best_params: Dict,
    metrics: Dict,
    test_metrics: Dict,
    search_time: float,
    numeric_features: List[str],
    categorical_features: List[str],
    reference_stats: Dict,
    threshold_info: Dict,
    config: TrainingConfig,
    run_id: str,
) -> Tuple[str, str, str]:
    """
    Save model, metadata, and configuration files.

    Includes both validation and test set metrics for comprehensive evaluation.

    Args:
        best_model: Trained model pipeline
        best_params: Best hyperparameters from grid search
        metrics: Validation metrics dictionary
        test_metrics: Test set metrics dictionary
        search_time: Grid search duration in seconds
        numeric_features: List of numeric feature names
        categorical_features: List of categorical feature names
        reference_stats: Reference statistics for drift detection
        threshold_info: Threshold tuning results
        config: Training configuration
        run_id: Unique run identifier

    Returns:
        Tuple: (versioned_model_path, latest_model_path, metadata_path, run_config_path)
    """
    models_dir = Path("models")
    configs_dir = Path("configs")

    logger.info("Saving model artifacts")

    # Save model with version - Returns checksum for integrity verification
    versioned_model_path = models_dir / f"churn_model_{run_id}.joblib"
    versioned_checksum = save_model(best_model, str(versioned_model_path))
    logger.info(f"✓ Versioned model saved: {versioned_model_path}")

    # Also save as "latest" for easy loading
    latest_model_path = models_dir / "churn_model_latest.joblib"
    latest_checksum = save_model(best_model, str(latest_model_path))
    logger.info(f"✓ Latest model saved: {latest_model_path}")

    # Save configuration for this run
    logger.info("Saving run configuration")
    run_config_path = configs_dir / f"run_config_{run_id}.yaml"
    config.save(str(run_config_path))
    logger.info(f"✓ Configuration saved: {run_config_path}")

    # Save metadata (includes validation, test metrics, reference statistics, threshold info, checksum)
    logger.info("Saving metadata")
    metadata_path = str(models_dir / f"metadata_{run_id}.json")
    save_metadata(
        run_id,
        best_model,
        best_params,
        metrics,
        test_metrics,
        search_time,
        metadata_path,
        config,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        reference_statistics=reference_stats,
        threshold=threshold_info,
        model_checksum=versioned_checksum,  # Include checksum in metadata
    )
    logger.info(
        f"✓ Metadata saved (includes validation, test metrics, threshold strategies, and checksum)"
    )

    return versioned_model_path, latest_model_path, metadata_path, run_config_path


def log_to_mlflow(
    best_model: Pipeline,
    X_val: pd.DataFrame,
    y_val_proba: np.ndarray,
    metadata_path: str,
    run_config_path: str,
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
    # Note: The API layer (serve.py) converts this to PredictionResponse with additional fields
    y_val_proba_full = best_model.predict_proba(X_val)  # Shape: (n_samples, 2)
    logger.debug(
        f"Creating signature: input shape {X_val.shape}, output shape {y_val_proba_full.shape}"
    )
    signature = infer_signature(X_val, y_val_proba_full)
    logger.info(
        f"✓ Model signature created: {len(signature.inputs.inputs)} inputs, output schema with 2 probability columns"
    )

    # Determine if we should register the model
    register_model = os.getenv("MLFLOW_REGISTER_MODEL", "false").lower() == "true"
    registered_model_name = "churn_prediction_model" if register_model else None

    # Log model (with optional registration)
    if config.mlflow.log_models:
        model_info = mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            signature=signature,
            input_example=X_val.iloc[:5],
            registered_model_name=registered_model_name,  # Register if flag set
        )
        logger.info("✓ Model logged to MLflow")

        # If registered, log registration details
        if register_model and hasattr(model_info, "registered_model_version"):
            logger.info("=" * 60)
            logger.info("MODEL REGISTRY")
            logger.info("=" * 60)
            logger.info(f"✓ Model registered: {registered_model_name}")
            logger.info(f"✓ Version: {model_info.registered_model_version}")
            logger.info(f"✓ Stage: None (use promotion script to set stage)")
            logger.info("=" * 60)

    # Log artifacts to MLflow
    if config.mlflow.log_artifacts:
        mlflow.log_artifact(metadata_path, "metadata")
        mlflow.log_artifact(str(run_config_path), "config")
        logger.info("✓ Configuration and metadata logged to MLflow")

    # Get MLflow run ID
    mlflow_run_id = None
    if mlflow.active_run():
        mlflow_run_id = mlflow.active_run().info.run_id
        logger.info(f"✓ MLflow run ID: {mlflow_run_id}")

    return mlflow_run_id


def print_training_summary(
    metrics: Dict,
    test_metrics: Dict,
    threshold_info: Dict,
    feature_importance_df: pd.DataFrame,
    config: TrainingConfig,
    run_id: str,
    mlflow_run_id: Optional[str],
    metrics_val_tuned: Optional[Dict] = None,
) -> None:
    """
    Print comprehensive training summary to console.

    Args:
        metrics: Validation metrics dictionary (default 0.5 threshold)
        test_metrics: Test metrics dictionary (tuned threshold)
        threshold_info: Threshold tuning results
        feature_importance_df: Feature importance DataFrame
        config: Training configuration
        run_id: Unique run identifier
        mlflow_run_id: MLflow run ID (if available)
        metrics_val_tuned: Validation metrics at the tuned threshold
    """
    tuned = threshold_info["chosen_threshold"]
    logger.info("=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Run ID: {run_id}")
    logger.info("")
    logger.info("Best Model Performance:")
    logger.info("  Validation Set (default 0.5 threshold):")
    logger.info(f"    ROC AUC:   {metrics['roc_auc']:.4f}")
    logger.info(f"    Precision: {metrics['precision']:.4f}")
    logger.info(f"    Recall:    {metrics['recall']:.4f}")
    logger.info(f"    F1 Score:  {metrics['f1']:.4f}")
    logger.info("")
    if metrics_val_tuned:
        logger.info(f"  Validation Set (tuned threshold {tuned:.4f}):")
        logger.info(f"    ROC AUC:   {metrics_val_tuned['roc_auc']:.4f}")
        logger.info(f"    Precision: {metrics_val_tuned['precision']:.4f}")
        logger.info(f"    Recall:    {metrics_val_tuned['recall']:.4f}")
        logger.info(f"    F1 Score:  {metrics_val_tuned['f1']:.4f}")
        logger.info("")
    logger.info(f"  Test Set (Held-Out, tuned threshold {tuned:.4f}):")
    logger.info(f"    ROC AUC:   {test_metrics['roc_auc']:.4f}")
    logger.info(f"    Precision: {test_metrics['precision']:.4f}")
    logger.info(f"    Recall:    {test_metrics['recall']:.4f}")
    logger.info(f"    F1 Score:  {test_metrics['f1']:.4f}")
    logger.info("")
    logger.info("Threshold:")
    logger.info(f"  Strategy:  {threshold_info['chosen_strategy']}")
    logger.info(f"  Threshold: {threshold_info['chosen_threshold']:.4f}")
    logger.info("")
    logger.info("Top 5 Features:")
    for idx, row in feature_importance_df.head(5).iterrows():
        logger.info(f"  {idx+1}. {row['feature']:40s} ({row['importance']:.4f})")
    logger.info("")
    logger.info("Deliverables:")
    logger.info(f"  ✓ Versioned model: models/churn_model_{run_id}.joblib")
    logger.info(f"  ✓ Latest model:    models/churn_model_latest.joblib")
    logger.info(f"  ✓ Configuration:   configs/run_config_{run_id}.yaml")
    logger.info(f"  ✓ Metadata:        models/metadata_{run_id}.json")
    logger.info(f"  ✓ Diagnostics:     diagnostics/evaluation_report_{run_id}.txt")
    logger.info(f"  ✓ Visualizations:  diagnostics/*_{run_id}.png (5 files)")
    logger.info(f"  ✓ Feature data:    diagnostics/feature_importances_{run_id}.csv")
    logger.info(f"  ✓ Training log:    logs/training.log")
    if mlflow_run_id:
        logger.info(f"  ✓ MLflow tracking: {config.mlflow.tracking_uri}/#{mlflow_run_id}")
    logger.info("")
    logger.info("Training workflow complete!")


def main(
    config_path: Optional[str] = None, use_env: bool = False, model_type: Optional[str] = None
) -> None:
    """
    Main training workflow orchestrator.

    This function orchestrates the complete training pipeline by calling focused
    sub-functions, each with a single responsibility. This decomposition improves
    testability, maintainability, and readability.

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
        from src.evaluation.metrics import compute_metrics

        y_val_pred_tuned = (y_val_proba >= tuned_threshold).astype(int)
        metrics_val_tuned = compute_metrics(y_val, y_val_pred_tuned, y_val_proba)
        logger.info(f"\nValidation metrics at tuned threshold ({tuned_threshold:.4f}):")
        logger.info(f"  Precision: {metrics_val_tuned['precision']:.4f}")
        logger.info(f"  Recall:    {metrics_val_tuned['recall']:.4f}")
        logger.info(f"  F1 Score:  {metrics_val_tuned['f1']:.4f}")
        for metric_name, metric_value in metrics_val_tuned.items():
            if isinstance(metric_value, (int, float)):
                mlflow.log_metric(f"val_tuned_{metric_name}", metric_value)
        logger.info("✓ Tuned-threshold validation metrics logged to MLflow")

        # 8. Evaluate on held-out test set
        test_metrics = evaluate_test_set(best_model, X_test, y_test, tuned_threshold)

        # 9. Compute reference statistics for drift detection
        logger.info("\nComputing reference statistics for drift detection")
        reference_stats = compute_reference_statistics(
            X_train, y_train, numeric_features, categorical_features
        )
        logger.info(
            f"✓ Reference statistics computed ({len(reference_stats['numeric'])} numeric, "
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

        # 10. Log to MLflow (model, signature, artifacts)
        mlflow_run_id = log_to_mlflow(
            best_model, X_val, y_val_proba, metadata_path, run_config_path, config
        )

        # 11. Print training summary
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

    finally:
        # End MLflow run
        if mlflow.active_run():
            mlflow.end_run()
            logger.info("✓ MLflow run ended")


# ==============================================================================
# VISUALIZATION HELPER (Kept for threshold analysis)
# ==============================================================================


def plot_threshold_analysis(
    y_true: pd.Series, y_proba: np.ndarray, output_path: str, run_id: str
) -> None:
    """
    Create comprehensive threshold analysis visualization.

    Generates 4 plots:
    1. Precision and Recall vs Threshold
    2. F1 Score vs Threshold
    3. Precision-Recall Curve
    4. Number of Positive Predictions vs Threshold
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import precision_recall_curve

    logger.info(f"Creating threshold analysis plots: {output_path}")

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Threshold Analysis - Run {run_id}", fontsize=16, fontweight="bold")

    # Plot 1: Precision-Recall vs Threshold
    axes[0, 0].plot(thresholds, precisions[:-1], label="Precision", linewidth=2, color="blue")
    axes[0, 0].plot(thresholds, recalls[:-1], label="Recall", linewidth=2, color="green")
    axes[0, 0].axvline(0.5, color="red", linestyle="--", alpha=0.7, label="Default (0.5)")

    # Find best F1 threshold
    best_f1_idx = np.argmax(f1_scores[:-1])
    best_threshold = thresholds[best_f1_idx]
    axes[0, 0].axvline(
        best_threshold,
        color="purple",
        linestyle="--",
        alpha=0.7,
        label=f"Best F1 ({best_threshold:.3f})",
    )

    axes[0, 0].set_xlabel("Threshold", fontsize=11)
    axes[0, 0].set_ylabel("Score", fontsize=11)
    axes[0, 0].set_title("Precision and Recall vs Threshold", fontsize=12, fontweight="bold")
    axes[0, 0].legend(loc="best")
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].set_xlim([0, 1])
    axes[0, 0].set_ylim([0, 1])

    # Plot 2: F1 Score vs Threshold
    axes[0, 1].plot(thresholds, f1_scores[:-1], linewidth=2, color="green")
    axes[0, 1].axvline(
        best_threshold,
        color="purple",
        linestyle="--",
        alpha=0.7,
        label=f"Best F1 ({best_threshold:.3f})",
    )
    axes[0, 1].axvline(0.5, color="red", linestyle="--", alpha=0.7, label="Default (0.5)")
    axes[0, 1].scatter(
        [best_threshold],
        [f1_scores[best_f1_idx]],
        color="purple",
        s=100,
        zorder=5,
        label=f"Max F1={f1_scores[best_f1_idx]:.3f}",
    )

    axes[0, 1].set_xlabel("Threshold", fontsize=11)
    axes[0, 1].set_ylabel("F1 Score", fontsize=11)
    axes[0, 1].set_title("F1 Score vs Threshold", fontsize=12, fontweight="bold")
    axes[0, 1].legend(loc="best")
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].set_xlim([0, 1])
    axes[0, 1].set_ylim([0, 1])

    # Plot 3: Precision-Recall Curve
    axes[1, 0].plot(recalls, precisions, linewidth=2, color="darkblue")
    axes[1, 0].fill_between(recalls, precisions, alpha=0.2, color="blue")
    axes[1, 0].set_xlabel("Recall", fontsize=11)
    axes[1, 0].set_ylabel("Precision", fontsize=11)
    axes[1, 0].set_title("Precision-Recall Curve", fontsize=12, fontweight="bold")
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].set_xlim([0, 1])
    axes[1, 0].set_ylim([0, 1])

    # Plot 4: Number of Predictions vs Threshold
    n_predictions = [(y_proba >= t).sum() for t in thresholds]
    axes[1, 1].plot(thresholds, n_predictions, linewidth=2, color="orange")
    axes[1, 1].axvline(0.5, color="red", linestyle="--", alpha=0.7, label="Default (0.5)")
    axes[1, 1].axvline(
        best_threshold,
        color="purple",
        linestyle="--",
        alpha=0.7,
        label=f"Best F1 ({best_threshold:.3f})",
    )
    axes[1, 1].axhline(
        len(y_true) * y_true.mean(),
        color="gray",
        linestyle=":",
        alpha=0.7,
        label=f"Actual positives ({int(y_true.sum())})",
    )

    axes[1, 1].set_xlabel("Threshold", fontsize=11)
    axes[1, 1].set_ylabel("Number of Positive Predictions", fontsize=11)
    axes[1, 1].set_title("Predicted Positives vs Threshold", fontsize=12, fontweight="bold")
    axes[1, 1].legend(loc="best")
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].set_xlim([0, 1])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"✓ Threshold analysis plot saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train churn prediction model with configuration management"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config file (YAML or JSON). If not provided, uses defaults.",
    )
    parser.add_argument(
        "--env", action="store_true", help="Load configuration from environment variables"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["random_forest", "xgboost", "logistic_regression"],
        help="Model type to train (overrides config file). Options: random_forest, xgboost, logistic_regression",
    )

    args = parser.parse_args()

    main(config_path=args.config, use_env=args.env, model_type=args.model_type)
