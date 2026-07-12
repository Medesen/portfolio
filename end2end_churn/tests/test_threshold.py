"""
Unit tests for threshold tuning strategies.

Tests threshold optimization functions and edge cases.

Threshold Tuning Strategies
"""

import numpy as np
import pytest

from src.evaluation.threshold import (
    evaluate_threshold,
    tune_threshold_cost_sensitive,
    tune_threshold_f1,
    tune_threshold_precision_constrained_recall,
    tune_threshold_top_k,
)

# =============================================================================
# F1 Maximization Tests
# =============================================================================


@pytest.mark.unit
def test_tune_threshold_f1_returns_valid_threshold():
    """Test F1 maximization returns a valid threshold."""
    # Create simple test data
    y_true = np.array([0, 0, 1, 1, 1])
    y_proba = np.array([0.1, 0.3, 0.6, 0.8, 0.9])

    threshold, metrics = tune_threshold_f1(y_true, y_proba)

    # Threshold should be between 0 and 1
    assert 0 < threshold < 1
    # Should return metrics
    assert "f1" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert 0 <= metrics["f1"] <= 1


@pytest.mark.unit
def test_tune_threshold_f1_improves_over_default():
    """Test F1 tuning improves over default 0.5 threshold."""
    # Imbalanced data (30% positive class)
    np.random.seed(42)
    n = 100
    y_true = np.random.choice([0, 1], n, p=[0.7, 0.3])
    # Generate probabilities with some predictive power
    y_proba = y_true * 0.6 + np.random.random(n) * 0.4

    # Tune threshold
    threshold_f1, metrics_f1 = tune_threshold_f1(y_true, y_proba)

    # Evaluate at default threshold
    metrics_default = evaluate_threshold(y_true, y_proba, 0.5)

    # F1-tuned threshold should give better or equal F1
    # Use approximate equality due to floating point precision
    assert metrics_f1["f1"] >= metrics_default["f1"] - 1e-6


# =============================================================================
# Precision-Constrained Recall Tests
# =============================================================================


@pytest.mark.unit
def test_precision_constrained_achieves_minimum():
    """Test precision-constrained returns threshold meeting constraint."""
    # Good model (high AUC)
    y_true = np.array([0] * 70 + [1] * 30)
    y_proba = np.concatenate(
        [np.random.uniform(0, 0.4, 70), np.random.uniform(0.6, 1.0, 30)]  # Negatives  # Positives
    )

    min_precision = 0.70
    threshold, metrics = tune_threshold_precision_constrained_recall(
        y_true, y_proba, min_precision=min_precision
    )

    # Should meet or exceed precision constraint
    assert metrics["precision"] >= min_precision
    assert 0 < threshold < 1


@pytest.mark.unit
def test_precision_constrained_raises_if_impossible():
    """Test precision-constrained raises ValueError if constraint impossible."""
    # Truly terrible model - all predictions point in wrong direction
    y_true = np.array([0] * 50 + [1] * 50)
    # Negatives get high probabilities, positives get low probabilities (opposite)
    y_proba = np.concatenate(
        [
            np.random.uniform(0.7, 1.0, 50),  # False positives (should be low)
            np.random.uniform(0.0, 0.3, 50),  # False negatives (should be high)
        ]
    )

    # Request very high precision (impossible with inverted predictions)
    with pytest.raises(ValueError) as exc_info:
        tune_threshold_precision_constrained_recall(y_true, y_proba, min_precision=0.95)

    assert "Cannot achieve min_precision" in str(exc_info.value)


# =============================================================================
# Top-K Selection Tests
# =============================================================================


@pytest.mark.unit
def test_top_k_with_exact_k():
    """Test top-k selection with exact k customers."""
    y_proba = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 0.95])
    k = 2

    threshold, metrics = tune_threshold_top_k(y_proba, k=k)

    # Should flag exactly k (or close due to ties)
    assert metrics["k"] == k
    assert metrics["n_flagged"] >= k  # May be slightly more due to ties


@pytest.mark.unit
def test_top_k_with_ratio():
    """Test top-k selection with ratio."""
    y_proba = np.random.uniform(0, 1, 100)
    ratio = 0.20

    threshold, metrics = tune_threshold_top_k(y_proba, ratio=ratio)

    # Should flag approximately ratio% customers
    expected_k = int(100 * ratio)
    assert metrics["k"] == expected_k
    assert abs(metrics["ratio"] - ratio) < 0.01


@pytest.mark.unit
def test_top_k_requires_k_or_ratio():
    """Test top-k raises error if neither k nor ratio provided."""
    y_proba = np.array([0.1, 0.5, 0.9])

    with pytest.raises(ValueError) as exc_info:
        tune_threshold_top_k(y_proba)

    assert "Must provide either k or ratio" in str(exc_info.value)


@pytest.mark.unit
def test_top_k_rejects_both_k_and_ratio():
    """Test top-k raises error if both k and ratio provided."""
    y_proba = np.array([0.1, 0.5, 0.9])

    with pytest.raises(ValueError) as exc_info:
        tune_threshold_top_k(y_proba, k=5, ratio=0.5)

    assert "only one of k or ratio" in str(exc_info.value)


@pytest.mark.unit
def test_top_k_validates_k_range():
    """Test top-k validates k is within valid range."""
    y_proba = np.array([0.1, 0.5, 0.9])

    # k too large
    with pytest.raises(ValueError):
        tune_threshold_top_k(y_proba, k=10)

    # k too small
    with pytest.raises(ValueError):
        tune_threshold_top_k(y_proba, k=0)


# =============================================================================
# Cost-Sensitive Tests
# =============================================================================


@pytest.mark.unit
def test_cost_sensitive_returns_valid_threshold():
    """Test cost-sensitive returns valid threshold and metrics."""
    y_true = np.array([0] * 70 + [1] * 30)
    y_proba = np.random.random(100)

    threshold, metrics = tune_threshold_cost_sensitive(y_true, y_proba, cost_fn=100, cost_fp=10)

    assert 0 < threshold < 1
    assert "expected_cost" in metrics
    assert "cost_ratio" in metrics
    assert metrics["cost_ratio"] == 10.0  # 100/10


@pytest.mark.unit
def test_cost_sensitive_prefers_recall_when_fn_costly():
    """Test cost-sensitive lowers threshold when FN cost > FP cost."""
    y_true = np.array([0] * 70 + [1] * 30)
    y_proba = np.concatenate([np.random.uniform(0, 0.5, 70), np.random.uniform(0.5, 1.0, 30)])

    # High FN cost (missing churner is expensive)
    threshold_high_fn, metrics_high_fn = tune_threshold_cost_sensitive(
        y_true, y_proba, cost_fn=100, cost_fp=1
    )

    # Balanced cost
    threshold_balanced, _ = tune_threshold_cost_sensitive(y_true, y_proba, cost_fn=1, cost_fp=1)

    # When FN is costly, should lower threshold to catch more positives
    # (higher recall, lower precision)
    assert threshold_high_fn <= threshold_balanced


# =============================================================================
# Evaluate Threshold Tests
# =============================================================================


@pytest.mark.unit
def test_evaluate_threshold_returns_all_metrics():
    """Test evaluate_threshold returns complete metrics dict."""
    y_true = np.array([0, 0, 1, 1, 1])
    y_proba = np.array([0.1, 0.3, 0.6, 0.8, 0.9])

    metrics = evaluate_threshold(y_true, y_proba, 0.5)

    assert "threshold" in metrics
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "n_positive_predictions" in metrics
    assert "positive_rate" in metrics

    # All metrics should be between 0 and 1
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["precision"] <= 1
    assert 0 <= metrics["recall"] <= 1
    assert 0 <= metrics["f1"] <= 1


@pytest.mark.unit
def test_evaluate_threshold_edge_cases():
    """Test evaluate_threshold handles edge cases."""
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.8, 0.9])

    # Very low threshold (predict all positive)
    metrics_low = evaluate_threshold(y_true, y_proba, 0.0)
    assert metrics_low["recall"] == 1.0  # Catches all positives

    # Very high threshold (predict all negative)
    metrics_high = evaluate_threshold(y_true, y_proba, 1.0)
    assert metrics_high["recall"] == 0.0  # Catches no positives


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.unit
def test_all_strategies_return_different_thresholds():
    """Test different strategies return different thresholds."""
    # Create realistic imbalanced data
    np.random.seed(42)
    y_true = np.array([0] * 700 + [1] * 300)
    y_proba = np.concatenate([np.random.uniform(0, 0.6, 700), np.random.uniform(0.4, 1.0, 300)])

    threshold_f1, _ = tune_threshold_f1(y_true, y_proba)
    threshold_pcr, _ = tune_threshold_precision_constrained_recall(
        y_true, y_proba, min_precision=0.6
    )
    threshold_topk, _ = tune_threshold_top_k(y_proba, ratio=0.3)
    threshold_cost, _ = tune_threshold_cost_sensitive(y_true, y_proba, cost_fn=10, cost_fp=1)

    # Strategies should give different results
    thresholds = [threshold_f1, threshold_pcr, threshold_topk, threshold_cost]
    assert len(set(thresholds)) >= 3  # At least 3 different thresholds


@pytest.mark.unit
def test_strategies_handle_perfect_model():
    """Test strategies handle perfect predictions."""
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.0, 0.1, 0.9, 1.0])  # Perfect separation

    # Should not raise errors
    threshold_f1, metrics_f1 = tune_threshold_f1(y_true, y_proba)
    threshold_topk, _ = tune_threshold_top_k(y_proba, k=2)

    # F1 should be 1.0 or very close
    assert metrics_f1["f1"] > 0.99
