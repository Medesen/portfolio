"""
Model Comparison Script

This script trains multiple ML models (Random Forest, XGBoost, Logistic Regression) on the
same dataset using identical train/val/test splits and compares their performance.

Usage (via Docker/Make - RECOMMENDED):
    make compare-models              # Full comparison with default config
    make compare-models-quick        # Quick comparison with minimal hyperparameters

Direct usage (for development only):
    python scripts/compare_models.py [--config CONFIG_PATH] [--quick]

Output:
    - Training logs for each model
    - Comparison report (diagnostics/model_comparison_report.txt)
    - Comparison visualization (diagnostics/model_comparison_YYYYMMDD_HHMMSS.png)
    - Individual model artifacts in models/ directory
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import TrainingConfig
from src.data.loader import load_data
from src.data.preprocessing import create_three_way_split, preprocess_data
from src.models.model_factory import get_all_model_types, get_model_display_name
from src.models.pipeline import build_pipeline
from src.training.trainer import evaluate_model
from src.utils.logger import setup_logger

# Setup logger
logger = setup_logger("model_comparison", log_file="logs/model_comparison.log")


def train_and_evaluate_model(
    model_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: TrainingConfig,
    use_quick_grid: bool = False,
) -> dict:
    """
    Train a single model and evaluate on validation set.

    Args:
        model_type: Model type identifier
        X_train, y_train: Training data
        X_val, y_val: Validation data
        config: Training configuration
        use_quick_grid: Use minimal hyperparameter grid for speed

    Returns:
        Dictionary with model type, metrics, and training time
    """
    import time

    from src.models.model_factory import get_model, get_param_grid, get_quick_param_grid
    from src.training.tuning import perform_grid_search

    logger.info("=" * 70)
    logger.info(f"Training {get_model_display_name(model_type)}")
    logger.info("=" * 70)

    # Get model and param grid
    model = get_model(model_type, random_state=config.data.random_state)

    # Use quick grid if specified, otherwise use full grid
    if use_quick_grid:
        param_grid = get_quick_param_grid(model_type)
        logger.info("Using quick parameter grid (fast iteration)")
    else:
        param_grid = get_param_grid(model_type)

    # Build preprocessing pipeline
    preprocessor, _, _ = build_pipeline()

    logger.info(f"Parameter grid: {param_grid}")
    logger.info(f"Grid size: {np.prod([len(v) for v in param_grid.values()])} combinations")

    # Train with grid search
    start_time = time.time()
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
    training_time = time.time() - start_time

    # Evaluate on validation set
    best_model = grid_search.best_estimator_
    metrics, y_pred, y_proba = evaluate_model(best_model, X_val, y_val)

    logger.info(f"{get_model_display_name(model_type)} training complete")
    logger.info(f"  Training time: {training_time:.2f}s")
    logger.info(f"  CV ROC AUC: {grid_search.best_score_:.4f}")
    logger.info(f"  Val ROC AUC: {metrics['roc_auc']:.4f}")
    logger.info(f"  Val F1: {metrics['f1']:.4f}")
    logger.info("")

    return {
        "model_type": model_type,
        "model_name": get_model_display_name(model_type),
        "cv_roc_auc": grid_search.best_score_,
        "val_roc_auc": metrics["roc_auc"],
        "val_accuracy": metrics["accuracy"],
        "val_precision": metrics["precision"],
        "val_recall": metrics["recall"],
        "val_f1": metrics["f1"],
        "val_average_precision": metrics["avg_precision"],
        "training_time": training_time,
        "search_time": search_time,
        "best_params": best_params,
        "n_combinations": np.prod([len(v) for v in param_grid.values()]),
    }


def create_comparison_visualization(results: list[dict], output_path: str):
    """
    Create bar chart comparing models across multiple metrics.

    Args:
        results: List of result dictionaries from each model
        output_path: Path to save the visualization
    """
    # Prepare data for plotting
    model_names = [r["model_name"] for r in results]
    metrics = [
        "val_roc_auc",
        "val_accuracy",
        "val_precision",
        "val_recall",
        "val_f1",
        "val_average_precision",
    ]
    metric_labels = ["ROC AUC", "Accuracy", "Precision", "Recall", "F1 Score", "Avg Precision"]

    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Model Comparison: Performance Metrics", fontsize=16, fontweight="bold")

    axes = axes.flatten()
    colors = ["#2ecc71", "#3498db", "#e74c3c"]  # Green, Blue, Red

    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx]
        values = [r[metric] for r in results]

        bars = ax.bar(
            model_names, values, color=colors, alpha=0.8, edgecolor="black", linewidth=1.5
        )

        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{val:.4f}",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=10,
            )

        # Highlight best model
        best_idx = values.index(max(values))
        bars[best_idx].set_edgecolor("gold")
        bars[best_idx].set_linewidth(3)

        ax.set_ylabel(label, fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)

        # Rotate x labels
        ax.set_xticklabels(model_names, rotation=0, ha="center")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Comparison visualization saved: {output_path}")


def generate_comparison_report(results: list[dict], output_path: str, run_id: str):
    """
    Generate detailed text report comparing all models.

    Args:
        results: List of result dictionaries from each model
        output_path: Path to save the report
        run_id: Timestamp identifier for this comparison run
    """
    with open(output_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("MODEL COMPARISON REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Run ID: {run_id}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Models Compared: {len(results)}\n")
        f.write("\n")

        # Summary table
        f.write("PERFORMANCE SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(
            f"{'Model':<15} {'ROC AUC':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}\n"
        )
        f.write("-" * 80 + "\n")

        for r in results:
            f.write(
                f"{r['model_name']:<15} "
                f"{r['val_roc_auc']:<10.4f} "
                f"{r['val_accuracy']:<10.4f} "
                f"{r['val_precision']:<10.4f} "
                f"{r['val_recall']:<10.4f} "
                f"{r['val_f1']:<10.4f}\n"
            )

        f.write("\n\n")

        # Training time comparison
        f.write("TRAINING TIME COMPARISON\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Model':<15} {'Total Time':<15} {'Grid Size':<15} {'Time per Comb':<15}\n")
        f.write("-" * 80 + "\n")

        for r in results:
            time_per_comb = r["training_time"] / r["n_combinations"]
            f.write(
                f"{r['model_name']:<15} "
                f"{r['training_time']:<15.2f} "
                f"{r['n_combinations']:<15} "
                f"{time_per_comb:<15.2f}\n"
            )

        f.write("\n\n")

        # Best model identification
        best_roc_auc = max(results, key=lambda x: x["val_roc_auc"])
        best_f1 = max(results, key=lambda x: x["val_f1"])
        fastest = min(results, key=lambda x: x["training_time"])

        f.write("WINNER ANALYSIS\n")
        f.write("-" * 80 + "\n")
        f.write(
            f"Best ROC AUC:     {best_roc_auc['model_name']} ({best_roc_auc['val_roc_auc']:.4f})\n"
        )
        f.write(f"Best F1 Score:    {best_f1['model_name']} ({best_f1['val_f1']:.4f})\n")
        f.write(f"Fastest Training: {fastest['model_name']} ({fastest['training_time']:.2f}s)\n")
        f.write("\n")

        # Detailed results
        f.write("\n\n")
        f.write("DETAILED RESULTS\n")
        f.write("=" * 80 + "\n\n")

        for r in results:
            f.write(f"{r['model_name']}\n")
            f.write("-" * 80 + "\n")
            f.write(f"  CV ROC AUC:          {r['cv_roc_auc']:.4f}\n")
            f.write(f"  Val ROC AUC:         {r['val_roc_auc']:.4f}\n")
            f.write(f"  Val Accuracy:        {r['val_accuracy']:.4f}\n")
            f.write(f"  Val Precision:       {r['val_precision']:.4f}\n")
            f.write(f"  Val Recall:          {r['val_recall']:.4f}\n")
            f.write(f"  Val F1 Score:        {r['val_f1']:.4f}\n")
            f.write(f"  Val Avg Precision:   {r['val_average_precision']:.4f}\n")
            f.write(f"  Training Time:       {r['training_time']:.2f}s\n")
            f.write(f"  Grid Search Time:    {r['search_time']:.2f}s\n")
            f.write(f"  Grid Size:           {r['n_combinations']} combinations\n")
            f.write(f"\n  Best Parameters:\n")
            for param, value in r["best_params"].items():
                f.write(f"    {param}: {value}\n")
            f.write("\n\n")

        f.write("=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")

    logger.info(f"Comparison report saved: {output_path}")


def main():
    """Main comparison workflow."""
    parser = argparse.ArgumentParser(description="Compare multiple ML models for churn prediction")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument(
        "--quick", action="store_true", help="Use quick parameter grids (fast iteration)"
    )
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("MODEL COMPARISON WORKFLOW")
    logger.info("=" * 80)

    # Generate run ID
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"Run ID: {run_id}")

    # Setup directories
    diagnostics_dir = Path("diagnostics")
    diagnostics_dir.mkdir(exist_ok=True)

    # Load configuration
    if args.config:
        config = TrainingConfig.from_yaml(args.config)
        logger.info(f"Loaded configuration from: {args.config}")
    else:
        config = TrainingConfig()
        logger.info("Using default configuration")

    # Load and prepare data (single split for fair comparison)
    logger.info("\n" + "=" * 80)
    logger.info("LOADING AND PREPARING DATA")
    logger.info("=" * 80)

    df = load_data(config.data.data_path)
    logger.info(f"Loaded {len(df)} records from {config.data.data_path}")

    X, y = preprocess_data(df)
    logger.info(f"Preprocessed data: {X.shape[0]} samples, {X.shape[1]} features")

    # Create splits (same splits for all models)
    X_train, X_val, X_test, y_train, y_val, y_test = create_three_way_split(
        X,
        y,
        test_size=config.data.test_size,
        val_size=config.data.val_size,
        random_state=config.data.random_state,
        stratify=config.data.stratify,
    )

    logger.info(f"Train set: {len(X_train)} samples")
    logger.info(f"Val set:   {len(X_val)} samples")
    logger.info(f"Test set:  {len(X_test)} samples")
    logger.info("")

    # Train and evaluate each model
    results = []
    model_types = get_all_model_types()

    for model_type in model_types:
        try:
            result = train_and_evaluate_model(
                model_type, X_train, y_train, X_val, y_val, config, use_quick_grid=args.quick
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to train {model_type}: {e}")
            logger.exception("Full traceback:")

    # Generate comparison report and visualization
    if len(results) > 0:
        logger.info("\n" + "=" * 80)
        logger.info("GENERATING COMPARISON ARTIFACTS")
        logger.info("=" * 80)

        report_path = diagnostics_dir / f"model_comparison_report_{run_id}.txt"
        generate_comparison_report(results, str(report_path), run_id)

        viz_path = diagnostics_dir / f"model_comparison_{run_id}.png"
        create_comparison_visualization(results, str(viz_path))

        # Print summary to console
        logger.info("\n" + "=" * 80)
        logger.info("COMPARISON SUMMARY")
        logger.info("=" * 80)
        logger.info(f"{'Model':<15} {'ROC AUC':<10} {'F1 Score':<10} {'Time (s)':<10}")
        logger.info("-" * 80)
        for r in results:
            logger.info(
                f"{r['model_name']:<15} {r['val_roc_auc']:<10.4f} {r['val_f1']:<10.4f} {r['training_time']:<10.2f}"
            )

        # Identify winner
        best_model = max(results, key=lambda x: x["val_roc_auc"])
        logger.info("\n" + "=" * 80)
        logger.info(
            f"WINNER: {best_model['model_name']} (ROC AUC: {best_model['val_roc_auc']:.4f})"
        )
        logger.info("=" * 80)
    else:
        logger.error("No models trained successfully!")

    logger.info("\nModel comparison complete!")


if __name__ == "__main__":
    main()
