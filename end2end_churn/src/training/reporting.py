"""
Evaluation, threshold tuning, and diagnostics for training runs.

Everything here reports on a trained model without changing it: validation
and held-out test evaluation, the four threshold-tuning strategies with
their comparison, and the diagnostic plots and reports written to
``diagnostics/`` and logged to MLflow.
"""

from pathlib import Path
from typing import Any, Optional

import mlflow
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from ..config import TrainingConfig
from ..evaluation import (
    plot_confusion_matrix,
    plot_feature_importances,
    plot_precision_recall_curve,
    plot_roc_curve,
)
from ..evaluation.metrics import compute_metrics
from ..evaluation.threshold import (
    evaluate_threshold,
    tune_threshold_cost_sensitive,
    tune_threshold_f1,
    tune_threshold_precision_constrained_recall,
    tune_threshold_top_k,
)
from ..utils import save_diagnostics_report
from ..utils.logger import get_logger
from . import evaluate_model, extract_feature_importances

logger = get_logger("churn_training")


def evaluate_and_analyze(
    best_model: Pipeline,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    numeric_features: list[str],
    categorical_features: list[str],
    run_id: str,
) -> "tuple[dict[str, Any], np.ndarray[Any, Any], np.ndarray[Any, Any], pd.DataFrame]":
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
    logger.info("Validation metrics logged to MLflow")

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
    logger.info("Confusion matrix saved")

    roc_path = str(diagnostics_dir / f"roc_curve_{run_id}.png")
    plot_roc_curve(y_val, y_val_proba, roc_path, title="ROC Curve (Validation Set)")
    logger.info("ROC curve saved")

    pr_path = str(diagnostics_dir / f"precision_recall_curve_{run_id}.png")
    plot_precision_recall_curve(
        y_val, y_val_proba, pr_path, title="Precision-Recall Curve (Validation Set)"
    )
    logger.info("PR curve saved")

    fi_path = str(diagnostics_dir / f"feature_importances_{run_id}.png")
    plot_feature_importances(feature_importance_df, fi_path, top_n=20)
    logger.info("Feature importances plot saved")

    # Save diagnostics report
    logger.info("Saving diagnostics report")
    diag_path = str(diagnostics_dir / f"evaluation_report_{run_id}.txt")
    save_diagnostics_report(metrics, diag_path, set_name="Validation (default 0.5 threshold)")
    logger.info("Evaluation report saved")

    # Save feature importances CSV
    fi_csv_path = diagnostics_dir / f"feature_importances_{run_id}.csv"
    feature_importance_df.to_csv(fi_csv_path, index=False)
    logger.info("Feature importances CSV saved")

    # Log artifacts to MLflow
    mlflow.log_artifact(cm_path, "plots")
    mlflow.log_artifact(roc_path, "plots")
    mlflow.log_artifact(pr_path, "plots")
    mlflow.log_artifact(fi_path, "plots")
    mlflow.log_artifact(str(fi_csv_path), "data")
    mlflow.log_artifact(diag_path, "reports")
    logger.info("Artifacts logged to MLflow")

    return metrics, y_val_pred, y_val_proba, feature_importance_df


def tune_thresholds(
    y_val: pd.Series, y_val_proba: "np.ndarray[Any, Any]", run_id: str
) -> dict[str, Any]:
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
    y_val_arr = np.asarray(y_val)
    threshold_f1, metrics_f1 = tune_threshold_f1(y_val_arr, y_val_proba)
    logger.info(f"  Best F1 threshold: {threshold_f1:.4f}")
    logger.info(f"  Precision: {metrics_f1['precision']:.4f}")
    logger.info(f"  Recall:    {metrics_f1['recall']:.4f}")
    logger.info(f"  F1 Score:  {metrics_f1['f1']:.4f}")

    # Strategy 2: Precision-Constrained Recall
    logger.info("\nStrategy 2: Precision-Constrained Recall")
    logger.info("  Goal: Maximize recall while maintaining min precision (70%)")
    try:
        threshold_pcr, metrics_pcr = tune_threshold_precision_constrained_recall(
            y_val_arr, y_val_proba, min_precision=0.70
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
        y_val_arr, y_val_proba, cost_fn=100, cost_fp=10
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
    metrics_default = evaluate_threshold(y_val_arr, y_val_proba, 0.5)
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

    if threshold_pcr is not None and metrics_pcr is not None:
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
    logger.info("Threshold analysis logged to MLflow")

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

    logger.info("Threshold strategies computed and saved")

    return threshold_info


def evaluate_test_set(
    best_model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, threshold: float
) -> dict[str, Any]:
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
    test_metrics = compute_metrics(y_test, y_test_pred, y_test_proba)

    logger.info(f"Test ROC AUC:       {test_metrics['roc_auc']:.4f}")
    logger.info(f"Test Accuracy:      {test_metrics['accuracy']:.4f}")
    logger.info(f"Test Precision:     {test_metrics['precision']:.4f}")
    logger.info(f"Test Recall:        {test_metrics['recall']:.4f}")
    logger.info(f"Test F1 Score:      {test_metrics['f1']:.4f}")
    logger.info(f"Test Avg Precision: {test_metrics['avg_precision']:.4f}")

    cm = test_metrics["confusion_matrix"]
    logger.info("\nTest Confusion Matrix:")
    logger.info(f"  TN={cm['tn']}, FP={cm['fp']}")
    logger.info(f"  FN={cm['fn']}, TP={cm['tp']}")

    # Log to MLflow
    for metric_name, metric_value in test_metrics.items():
        if isinstance(metric_value, (int, float)):
            mlflow.log_metric(f"test_{metric_name}", metric_value)
    logger.info("Test metrics logged to MLflow")

    return test_metrics


def print_training_summary(
    metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    threshold_info: dict[str, Any],
    feature_importance_df: pd.DataFrame,
    config: TrainingConfig,
    run_id: str,
    mlflow_run_id: Optional[str],
    metrics_val_tuned: Optional[dict[str, Any]] = None,
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
    for rank, (_, row) in enumerate(feature_importance_df.head(5).iterrows(), start=1):
        logger.info(f"  {rank}. {row['feature']:40s} ({row['importance']:.4f})")
    logger.info("")
    logger.info("Deliverables:")
    logger.info(f"  Versioned model: models/churn_model_{run_id}.joblib")
    logger.info("  Latest model:    models/churn_model_latest.joblib")
    logger.info(f"  Configuration:   configs/run_config_{run_id}.yaml")
    logger.info(f"  Metadata:        models/metadata_{run_id}.json")
    logger.info(f"  Diagnostics:     diagnostics/evaluation_report_{run_id}.txt")
    logger.info(f"  Visualizations:  diagnostics/*_{run_id}.png (5 files)")
    logger.info(f"  Feature data:    diagnostics/feature_importances_{run_id}.csv")
    logger.info("  Training log:    logs/training.log")
    if mlflow_run_id:
        logger.info(f"  MLflow tracking: {config.mlflow.tracking_uri}/#{mlflow_run_id}")
    logger.info("")
    logger.info("Training workflow complete!")


def plot_threshold_analysis(
    y_true: pd.Series, y_proba: "np.ndarray[Any, Any]", output_path: str, run_id: str
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

    logger.info(f"Threshold analysis plot saved: {output_path}")
