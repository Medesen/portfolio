"""
Threshold tuning strategies for classification models.

This module provides functions to optimize classification thresholds beyond
the default 0.5, aligning model decisions with business objectives.
"""

from typing import Optional

import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve, precision_score, recall_score


def tune_threshold_f1(y_true: np.ndarray, y_proba: np.ndarray) -> tuple[float, dict[str, float]]:
    """
    Find threshold that maximizes F1 score.

    Use when you want to balance precision and recall equally.
    This is a good default strategy for imbalanced datasets.

    Args:
        y_true: True labels (0/1)
        y_proba: Predicted probabilities for positive class

    Returns:
        Tuple of (best threshold, metrics at best threshold)

    Example:
        >>> threshold, metrics = tune_threshold_f1(y_val, y_val_proba)
        >>> print(f"Best F1 at threshold {threshold:.3f}: {metrics['f1']:.3f}")
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

    # Calculate F1 for each threshold
    # F1 = 2 * (precision * recall) / (precision + recall)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)

    # Find best threshold (exclude last element which corresponds to recall=0)
    best_idx = np.argmax(f1_scores[:-1])
    best_threshold = float(thresholds[best_idx])

    # Compute metrics at best threshold
    y_pred = (y_proba >= best_threshold).astype(int)

    metrics = {
        "threshold": best_threshold,
        "f1": float(f1_scores[best_idx]),
        "precision": float(precisions[best_idx]),
        "recall": float(recalls[best_idx]),
        "accuracy": float((y_pred == y_true).mean()),
    }

    return best_threshold, metrics


def tune_threshold_precision_constrained_recall(
    y_true: np.ndarray, y_proba: np.ndarray, min_precision: float = 0.75
) -> tuple[float, dict[str, float]]:
    """
    Find threshold that maximizes recall while maintaining minimum precision.

    Use when false positives are costly (e.g., expensive retention programs).
    You want to catch as many churners as possible (high recall) while
    ensuring you're not wasting money on non-churners (high precision).

    Args:
        y_true: True labels (0/1)
        y_proba: Predicted probabilities for positive class
        min_precision: Minimum acceptable precision (e.g., 0.75 = 75%)

    Returns:
        Tuple of (best threshold, metrics at best threshold)

    Raises:
        ValueError: If min_precision cannot be achieved with this model

    Example:
        >>> # Want at least 70% precision (only 30% false alarms max)
        >>> threshold, metrics = tune_threshold_precision_constrained_recall(
        ...     y_val, y_val_proba, min_precision=0.70
        ... )
        >>> print(f"Precision: {metrics['precision']:.1%}, Recall: {metrics['recall']:.1%}")
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

    # Find thresholds where precision >= min_precision
    valid_idx = np.where(precisions[:-1] >= min_precision)[0]

    if len(valid_idx) == 0:
        raise ValueError(
            f"Cannot achieve min_precision={min_precision:.3f}. "
            f"Maximum achievable precision: {precisions.max():.3f}. "
            f"Try lowering min_precision or improving the model."
        )

    # Among valid thresholds, pick the one with highest recall
    best_idx = valid_idx[np.argmax(recalls[valid_idx])]
    best_threshold = float(thresholds[best_idx])

    # Compute metrics
    y_pred = (y_proba >= best_threshold).astype(int)

    metrics = {
        "threshold": best_threshold,
        "precision": float(precisions[best_idx]),
        "recall": float(recalls[best_idx]),
        "f1": float(f1_score(y_true, y_pred)),
        "accuracy": float((y_pred == y_true).mean()),
        "min_precision_constraint": min_precision,
    }

    return best_threshold, metrics


def tune_threshold_top_k(
    y_proba: np.ndarray, k: Optional[int] = None, ratio: Optional[float] = None
) -> tuple[float, dict[str, float]]:
    """
    Find threshold that flags exactly top-k or top-k% customers.

    Use when you have a fixed budget or capacity constraint.

    Examples:
        - "We can only contact 1000 customers per month" → k=1000
        - "We want to target the top 10% riskiest customers" → ratio=0.1

    Args:
        y_proba: Predicted probabilities for positive class
        k: Exact number of customers to flag (mutually exclusive with ratio)
        ratio: Proportion of customers to flag (e.g., 0.1 for top 10%)

    Returns:
        Tuple of (threshold, info dict)

    Raises:
        ValueError: If neither or both k and ratio are provided, or if k exceeds sample size

    Example:
        >>> # Target top 20% highest-risk customers
        >>> threshold, metrics = tune_threshold_top_k(y_val_proba, ratio=0.20)
        >>> print(f"Threshold: {threshold:.3f}, will flag {metrics['k']} customers")
    """
    if k is None and ratio is None:
        raise ValueError("Must provide either k or ratio")
    if k is not None and ratio is not None:
        raise ValueError("Provide only one of k or ratio, not both")

    # Sort probabilities descending
    sorted_proba = np.sort(y_proba)[::-1]

    if ratio is not None:
        k = int(len(y_proba) * ratio)

    # Validate k (mypy type guard: k is definitely int here)
    assert k is not None, "k must be provided via k or ratio parameter"
    if k >= len(y_proba):
        raise ValueError(
            f"k={k} exceeds number of samples {len(y_proba)}. " f"Try a smaller k or ratio."
        )

    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    # Threshold is the k-th highest probability
    # Subtract small epsilon to handle ties (ensures we get at least k predictions)
    threshold = float(sorted_proba[k - 1]) - 1e-10

    metrics = {
        "threshold": threshold,
        "k": k,
        "ratio": k / len(y_proba),
        "n_flagged": int((y_proba >= threshold).sum()),
    }

    return threshold, metrics


def tune_threshold_cost_sensitive(
    y_true: np.ndarray, y_proba: np.ndarray, cost_fn: float = 1.0, cost_fp: float = 1.0
) -> tuple[float, dict[str, float]]:
    """
    Find threshold that minimizes expected cost.

    Use when you know the business costs of errors.

    Example scenarios:
        - False negative (missed churner): lose $100 customer lifetime value
        - False positive (wrong prediction): waste $10 on retention program
        → cost_fn=100, cost_fp=10 → ratio 10:1 means prioritize catching churners

    Args:
        y_true: True labels (0/1)
        y_proba: Predicted probabilities for positive class
        cost_fn: Cost of false negative (missing a positive case)
        cost_fp: Cost of false positive (false alarm)

    Returns:
        Tuple of (best threshold, metrics including expected cost)

    Example:
        >>> # Missing a churner costs $100 LTV, false alarm costs $10 campaign
        >>> threshold, metrics = tune_threshold_cost_sensitive(
        ...     y_val, y_val_proba, cost_fn=100, cost_fp=10
        ... )
        >>> print(f"Expected cost: ${metrics['expected_cost']:.2f}")
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

    n_positive = y_true.sum()
    n_negative = len(y_true) - n_positive

    best_cost = float("inf")
    best_threshold = 0.5
    best_idx = 0

    for idx, threshold in enumerate(thresholds):
        # Calculate confusion matrix components
        tp = recalls[idx] * n_positive  # True positives
        fn = n_positive - tp  # False negatives

        # precision = tp / (tp + fp) → fp = tp / precision - tp
        if precisions[idx] > 0:
            fp = tp / precisions[idx] - tp
        else:
            fp = 0

        # Calculate total cost
        cost = cost_fn * fn + cost_fp * fp

        if cost < best_cost:
            best_cost = cost
            best_threshold = float(threshold)
            best_idx = idx

    # Compute metrics at best threshold
    y_pred = (y_proba >= best_threshold).astype(int)

    metrics = {
        "threshold": best_threshold,
        "expected_cost": float(best_cost),
        "cost_fn": cost_fn,
        "cost_fp": cost_fp,
        "cost_ratio": cost_fn / cost_fp if cost_fp > 0 else float("inf"),
        "precision": float(precisions[best_idx]),
        "recall": float(recalls[best_idx]),
        "f1": float(f1_score(y_true, y_pred)),
    }

    return best_threshold, metrics


def evaluate_threshold(
    y_true: np.ndarray, y_proba: np.ndarray, threshold: float
) -> dict[str, float]:
    """
    Evaluate metrics at a specific threshold.

    Use this to compare different thresholds or to evaluate the default 0.5.

    Args:
        y_true: True labels (0/1)
        y_proba: Predicted probabilities for positive class
        threshold: Classification threshold to evaluate

    Returns:
        Dictionary of metrics at the specified threshold

    Example:
        >>> # Compare default threshold to tuned threshold
        >>> metrics_default = evaluate_threshold(y_val, y_val_proba, 0.5)
        >>> metrics_tuned = evaluate_threshold(y_val, y_val_proba, 0.35)
        >>> print(f"Default F1: {metrics_default['f1']:.3f}")
        >>> print(f"Tuned F1: {metrics_tuned['f1']:.3f}")
    """
    y_pred = (y_proba >= threshold).astype(int)

    return {
        "threshold": threshold,
        "accuracy": float((y_pred == y_true).mean()),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "n_positive_predictions": int(y_pred.sum()),
        "positive_rate": float(y_pred.mean()),
    }
