"""I/O utilities for saving models, metadata, and reports."""

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
from sklearn.pipeline import Pipeline

from .logger import get_logger

logger = get_logger("churn_training")


def compute_file_checksum(file_path: str, algorithm: str = "sha256") -> str:
    """
    Compute cryptographic checksum of a file.

    Args:
        file_path: Path to file
        algorithm: Hash algorithm (sha256, md5, sha1)

    Returns:
        Hexadecimal checksum string

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    hash_obj = hashlib.new(algorithm)

    with open(file_path, "rb") as f:
        # Read in chunks to handle large files efficiently
        for chunk in iter(lambda: f.read(4096), b""):
            hash_obj.update(chunk)

    return hash_obj.hexdigest()


def save_model(pipeline: Pipeline, output_path: str) -> str:
    """
    Save trained model pipeline to disk with checksum validation.

    Added integrity verification via SHA256 checksum.
    Saves both model file and checksum file (.sha256) for security.

    Args:
        pipeline: Trained model pipeline
        output_path: Path where model will be saved

    Returns:
        SHA256 checksum of the saved model
    """
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    joblib.dump(pipeline, output_path)

    model_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    logger.info(f"Model saved: {output_path}")
    logger.info(f"Size: {model_size_mb:.2f} MB")

    # Compute and save checksum for security
    checksum = compute_file_checksum(output_path, algorithm="sha256")

    checksum_path = f"{output_path}.sha256"
    checksum_data = {
        "checksum": checksum,
        "algorithm": "sha256",
        "file": os.path.basename(output_path),
        "size_bytes": Path(output_path).stat().st_size,
        "created_at": datetime.now().isoformat(),
    }

    with open(checksum_path, "w") as f:
        json.dump(checksum_data, f, indent=2)

    logger.info(f"Model checksum: {checksum[:16]}... (SHA256)")
    logger.debug(f"Checksum file: {checksum_path}")

    return checksum


def verify_model_checksum(model_path: str) -> bool:
    """
    Verify model file integrity using checksum.

    Validates model file hasn't been corrupted or tampered with.

    Args:
        model_path: Path to model file

    Returns:
        True if checksum matches, False if mismatch or checksum file missing

    Raises:
        ValueError: If checksum mismatch detected (security risk)
    """
    checksum_path = f"{model_path}.sha256"

    if not os.path.exists(checksum_path):
        logger.warning(f"No checksum file found for {model_path} - skipping verification")
        return False

    # Load expected checksum
    try:
        with open(checksum_path, "r") as f:
            checksum_data = json.load(f)
        expected_checksum = checksum_data["checksum"]
        algorithm = checksum_data.get("algorithm", "sha256")
    except Exception as e:
        logger.error(f"Failed to read checksum file: {e}")
        return False

    # Compute actual checksum
    try:
        actual_checksum = compute_file_checksum(model_path, algorithm)
    except Exception as e:
        logger.error(f"Failed to compute checksum: {e}")
        return False

    # Compare checksums
    if actual_checksum != expected_checksum:
        raise ValueError(
            f"Model checksum mismatch! File may be corrupted or tampered with.\n"
            f"Expected: {expected_checksum}\n"
            f"Actual:   {actual_checksum}\n"
            f"File:     {model_path}"
        )

    logger.info(f"Model checksum verified: {actual_checksum[:16]}...")
    return True


def save_metadata(
    run_id: str,
    best_model: Pipeline,
    best_params: dict,
    metrics: dict,
    test_metrics: dict,
    search_time: float,
    output_path: str,
    config,
    numeric_features: Optional[list[str]] = None,
    categorical_features: Optional[list[str]] = None,
    reference_statistics: Optional[dict] = None,
    threshold: Optional[dict] = None,
    model_checksum: Optional[str] = None,
) -> None:
    """
    Save model metadata including hyperparameters, validation metrics, test metrics, and run info.

    Args:
        run_id: Unique run identifier (timestamp)
        best_model: Trained model pipeline (for extracting model type)
        best_params: Best hyperparameters from grid search
        metrics: Validation set evaluation metrics dictionary
        test_metrics: Test set evaluation metrics dictionary
        search_time: Time taken for hyperparameter search
        output_path: Path where metadata JSON will be saved
        config: Training configuration (for CV folds, scoring metric, etc.)
        numeric_features: List of numeric feature names (for schema tracking)
        categorical_features: List of categorical feature names (for schema tracking)
        reference_statistics: Reference statistics for drift detection
        threshold: Threshold tuning strategies and chosen threshold
        model_checksum: SHA256 checksum of model file for integrity validation
    """
    # Extract model type dynamically from the trained pipeline
    classifier = best_model.named_steps["classifier"]
    model_type = type(classifier).__name__

    metadata = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "model_type": model_type,  # Dynamically extracted from model
        "hyperparameters": best_params,
        "validation_metrics": {
            "accuracy": float(metrics["accuracy"]),
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "f1": float(metrics["f1"]),
            "roc_auc": float(metrics["roc_auc"]),
            "avg_precision": float(metrics["avg_precision"]),
        },
        "test_metrics": {
            "accuracy": float(test_metrics["accuracy"]),
            "precision": float(test_metrics["precision"]),
            "recall": float(test_metrics["recall"]),
            "f1": float(test_metrics["f1"]),
            "roc_auc": float(test_metrics["roc_auc"]),
            "avg_precision": float(test_metrics["avg_precision"]),
        },
        "confusion_matrix": metrics["confusion_matrix"],
        "training_info": {
            "grid_search_time_seconds": float(search_time),
            "cross_validation_folds": config.model.cv_folds,  # From config
            "scoring_metric": config.model.scoring,  # From config
        },
    }

    # Add schema information if provided
    if numeric_features is not None and categorical_features is not None:
        metadata["schema"] = {
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "all_features": numeric_features + categorical_features,
            "n_features": len(numeric_features) + len(categorical_features),
        }

    # Add reference statistics for drift detection if provided
    if reference_statistics is not None:
        metadata["reference_statistics"] = reference_statistics

    # Add threshold tuning information if provided
    if threshold is not None:
        metadata["threshold"] = threshold

    # Add model checksum for integrity verification
    if model_checksum is not None:
        metadata["model_checksum"] = {"sha256": model_checksum, "algorithm": "sha256"}

    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Metadata: {output_path}")


def save_diagnostics_report(metrics: dict, output_path: str, set_name: str = "Validation") -> None:
    """
    Save comprehensive diagnostics report to text file.

    Args:
        metrics: Dictionary of computed metrics
        output_path: Where to save the report
        set_name: Name of the evaluation set (e.g., "Validation", "Test")
    """
    cm = metrics["confusion_matrix"]

    report = f"""
{'='*70}
CHURN PREDICTION MODEL - DIAGNOSTIC REPORT
{'='*70}

Evaluation Set: {set_name}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{'='*70}
CLASSIFICATION METRICS
{'='*70}

Overall Performance:
  Accuracy:           {metrics['accuracy']:.4f}
  
Probabilistic Metrics:
  ROC AUC:            {metrics['roc_auc']:.4f}  (probability ranking quality)
  Average Precision:  {metrics['avg_precision']:.4f}  (area under PR curve)

Churn Class Performance (Positive Class):
  Precision:          {metrics['precision']:.4f}  (of predicted churners, % actually churn)
  Recall:             {metrics['recall']:.4f}  (of actual churners, % we caught)
  F1 Score:           {metrics['f1']:.4f}  (harmonic mean of precision & recall)

{'='*70}
CONFUSION MATRIX
{'='*70}

                    Predicted
                No Churn    Churn
Actual  
No Churn          {cm['tn']:5d}    {cm['fp']:5d}
Churn             {cm['fn']:5d}    {cm['tp']:5d}

Interpretation:
  True Negatives  (TN): {cm['tn']:5d}  Correctly predicted "No churn"
  False Positives (FP): {cm['fp']:5d}  Predicted "Churn" but stayed (wasted retention)
  False Negatives (FN): {cm['fn']:5d}  Predicted "No churn" but left (MISSED!)
  True Positives  (TP): {cm['tp']:5d}  Correctly predicted "Churn"

{'='*70}
BUSINESS INSIGHTS
{'='*70}

Total customers evaluated: {cm['tn'] + cm['fp'] + cm['fn'] + cm['tp']}
Actual churners:          {cm['fn'] + cm['tp']} ({(cm['fn'] + cm['tp'])/(cm['tn'] + cm['fp'] + cm['fn'] + cm['tp'])*100:.1f}%)

Model Performance:
  - Catches {metrics['recall']*100:.1f}% of churners (recall)
  - {metrics['precision']*100:.1f}% of churn predictions are correct (precision)
  - Misses {cm['fn']} churners (false negatives) - these are costly!
  - Flags {cm['fp']} non-churners incorrectly (false positives) - wasted effort

Cost-Benefit Considerations:
  - False negatives (missing churners) are typically MORE costly than
    false positives (unnecessary retention efforts)
  - May want to tune threshold to increase recall at cost of precision
  - Current model is better at identifying non-churners than churners
    (typical for imbalanced datasets)

{'='*70}
MODEL INTERPRETATION
{'='*70}

ROC AUC = {metrics['roc_auc']:.3f}:
  - 0.5 = random guessing
  - 1.0 = perfect classifier
  - {metrics['roc_auc']:.3f} = {'good' if metrics['roc_auc'] > 0.8 else 'fair' if metrics['roc_auc'] > 0.7 else 'needs improvement'}

Average Precision = {metrics['avg_precision']:.3f}:
  - Baseline (random) = {cm['fn'] + cm['tp']}/{cm['tn'] + cm['fp'] + cm['fn'] + cm['tp']} = {(cm['fn'] + cm['tp'])/(cm['tn'] + cm['fp'] + cm['fn'] + cm['tp']):.3f}
  - Our model = {metrics['avg_precision']:.3f}
  - {'Significant' if metrics['avg_precision'] > 0.5 else 'Moderate'} improvement over random

{'='*70}
"""

    with open(output_path, "w") as f:
        f.write(report)

    logger.info(f"Diagnostics report: {output_path}")
