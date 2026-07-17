"""
Artifact publication for training runs.

Owns everything a completed run leaves on disk: the drift-detection reference
statistics baked into model metadata, the versioned + latest model files with
integrity checksums, the run configuration snapshot, and the metadata pairing
that keeps ``churn_model_latest.joblib`` matched to its own metadata file.
"""

import shutil
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from ..config import TrainingConfig
from ..utils import save_metadata, save_model
from ..utils.logger import get_logger

logger = get_logger("churn_training")


def compute_reference_statistics(
    X: pd.DataFrame,
    y: pd.Series,
    numeric_features: list[str],
    categorical_features: list[str],
    model: Optional[Pipeline] = None,
    threshold: Optional[float] = None,
) -> dict[str, Any]:
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
        model: Trained pipeline. When provided with ``threshold``, a prediction
            baseline is stored (predicted-positive rate at the tuned threshold
            plus a probability histogram) so the /drift endpoint can compare
            like with like instead of label prevalence vs default-threshold
            predictions.
        threshold: Tuned classification threshold used for the prediction baseline.

    Returns:
        Dictionary with reference statistics
    """
    stats: dict[str, Any] = {"numeric": {}, "categorical": {}, "target": {}}

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

    # Target distribution (label prevalence)
    stats["target"] = {"positive_rate": float(y.mean()), "n_samples": len(y)}

    # Prediction baseline: the model's predicted-positive rate on the training
    # data AT THE TUNED THRESHOLD (not label prevalence, and not the default 0.5
    # cut), plus a fixed-bin probability histogram for distribution-level drift.
    if model is not None and threshold is not None:
        proba = model.predict_proba(X)[:, 1]
        bin_edges = np.linspace(0.0, 1.0, 11)
        counts, _ = np.histogram(proba, bins=bin_edges)
        total = int(counts.sum())
        proportions = (counts / total).tolist() if total > 0 else [0.0] * len(counts)
        stats["prediction"] = {
            "threshold": float(threshold),
            "positive_rate": float((proba >= threshold).mean()),
            "proba_mean": float(proba.mean()),
            "proba_histogram": {
                "bin_edges": bin_edges.tolist(),
                "proportions": proportions,
            },
            "n_samples": int(len(proba)),
        }

    return stats


def save_training_artifacts(
    best_model: Pipeline,
    best_params: dict[str, Any],
    metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    search_time: float,
    numeric_features: list[str],
    categorical_features: list[str],
    reference_stats: dict[str, Any],
    threshold_info: dict[str, Any],
    config: TrainingConfig,
    run_id: str,
) -> tuple[Path, Path, str, Path]:
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
    logger.info(f"Versioned model saved: {versioned_model_path}")

    # Also save as "latest" for easy loading (writes its own .sha256 sidecar)
    latest_model_path = models_dir / "churn_model_latest.joblib"
    save_model(best_model, str(latest_model_path))
    logger.info(f"Latest model saved: {latest_model_path}")

    # Save configuration for this run
    logger.info("Saving run configuration")
    run_config_path = configs_dir / f"run_config_{run_id}.yaml"
    config.save(str(run_config_path))
    logger.info(f"Configuration saved: {run_config_path}")

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
        "Metadata saved (includes validation, test metrics, threshold strategies, and checksum)"
    )

    # Pair the "latest" model with its own metadata file so loaders never glob a
    # stray or lexically-newer metadata_*.json for churn_model_latest.joblib.
    latest_metadata_path = models_dir / "metadata_latest.json"
    shutil.copyfile(metadata_path, latest_metadata_path)
    logger.info(f"Latest metadata saved: {latest_metadata_path}")

    return versioned_model_path, latest_model_path, metadata_path, run_config_path
