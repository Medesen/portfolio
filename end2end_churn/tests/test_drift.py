"""
Unit tests for drift detection utilities.

Tests PSI calculation, numeric drift, categorical drift, and prediction drift.

Comprehensive Testing
"""

import numpy as np
import pandas as pd
import pytest

from src.utils.drift import (
    analyze_drift,
    calculate_psi,
    detect_categorical_drift,
    detect_numeric_drift,
    detect_prediction_drift,
)

# =============================================================================
# PSI (Population Stability Index) Tests
# =============================================================================


@pytest.mark.unit
def test_calculate_psi_no_drift():
    """Test PSI calculation with identical distributions (no drift)."""
    expected = {"A": 0.5, "B": 0.3, "C": 0.2}
    actual = {"A": 0.5, "B": 0.3, "C": 0.2}

    psi, interpretation = calculate_psi(expected, actual)

    assert psi < 0.01  # Nearly zero
    assert interpretation == "stable"


@pytest.mark.unit
def test_calculate_psi_moderate_drift():
    """Test PSI calculation with moderate distribution shift."""
    expected = {"A": 0.6, "B": 0.3, "C": 0.1}
    actual = {"A": 0.3, "B": 0.5, "C": 0.2}  # More significant shift

    psi, interpretation = calculate_psi(expected, actual)

    assert psi >= 0.1  # Should be moderate or significant
    assert interpretation in ["moderate_drift", "significant_drift"]


@pytest.mark.unit
def test_calculate_psi_significant_drift():
    """Test PSI calculation with significant distribution shift."""
    expected = {"A": 0.8, "B": 0.2}
    actual = {"A": 0.2, "B": 0.8}  # Completely flipped

    psi, interpretation = calculate_psi(expected, actual)

    assert psi >= 0.25
    assert interpretation == "significant_drift"


@pytest.mark.unit
def test_calculate_psi_handles_zero_values():
    """Test PSI calculation handles small/zero proportions gracefully."""
    expected = {"A": 0.99, "B": 0.01}
    actual = {"A": 0.98, "B": 0.02}

    psi, interpretation = calculate_psi(expected, actual)

    # Should not raise error, should return valid PSI
    assert isinstance(psi, float)
    assert psi >= 0


@pytest.mark.unit
def test_calculate_psi_handles_new_categories():
    """Test PSI handles categories that appear in actual but not expected."""
    expected = {"A": 0.5, "B": 0.5}
    actual = {"A": 0.4, "B": 0.4, "C": 0.2}  # New category 'C'

    psi, interpretation = calculate_psi(expected, actual)

    # Should handle gracefully (treat missing expected as small value)
    assert isinstance(psi, float)
    assert psi > 0


# =============================================================================
# Numeric Drift Tests
# =============================================================================


@pytest.mark.unit
def test_detect_numeric_drift_no_change():
    """Test numeric drift detection when data hasn't changed."""
    reference_stats = {"mean": 30.0, "std": 10.0, "min": 0, "max": 72}
    current_data = pd.Series(np.random.normal(30, 10, 100))

    result = detect_numeric_drift(reference_stats, current_data, threshold=0.2)

    assert result["drift_detected"] is False
    assert result["reason"] == "stable"
    assert "metrics" in result


@pytest.mark.unit
def test_detect_numeric_drift_mean_shift():
    """Test numeric drift detection catches significant mean shift."""
    reference_stats = {"mean": 30.0, "std": 10.0, "min": 0, "max": 72}
    current_data = pd.Series(np.random.normal(50, 10, 100))  # Mean shifted +20

    result = detect_numeric_drift(reference_stats, current_data, threshold=0.2)

    assert result["drift_detected"] is True
    # New reason codes: more descriptive than generic 'statistical_shift'
    assert result["reason"] in [
        "drift_detected_by_relative_change",
        "drift_detected_by_both_methods",
    ]
    assert result["metrics"]["mean_change"] > 0.2


@pytest.mark.unit
def test_detect_numeric_drift_std_shift():
    """Test numeric drift detection catches significant std shift."""
    reference_stats = {"mean": 30.0, "std": 10.0, "min": 0, "max": 72}
    current_data = pd.Series(np.random.normal(30, 25, 100))  # Std increased significantly

    result = detect_numeric_drift(reference_stats, current_data, threshold=0.2)

    assert result["drift_detected"] is True
    assert "metrics" in result
    assert "std_change" in result["metrics"]


@pytest.mark.unit
def test_detect_numeric_drift_handles_missing_values():
    """Test numeric drift handles NaN values."""
    reference_stats = {"mean": 30.0, "std": 10.0, "min": 0, "max": 72}
    current_data = pd.Series([30, 35, 40, np.nan, 25, 28])

    # Should not raise error
    result = detect_numeric_drift(reference_stats, current_data, threshold=0.2)

    assert isinstance(result, dict)
    assert "drift_detected" in result


@pytest.mark.unit
def test_detect_numeric_drift_small_sample():
    """Test numeric drift with very small sample."""
    reference_stats = {"mean": 30.0, "std": 10.0, "min": 0, "max": 72}
    current_data = pd.Series([25, 30, 35])  # Only 3 samples

    result = detect_numeric_drift(reference_stats, current_data, threshold=0.2)

    # Should handle small samples (may flag as insufficient data)
    assert isinstance(result, dict)


# =============================================================================
# Categorical Drift Tests
# =============================================================================


@pytest.mark.unit
def test_detect_categorical_drift_stable():
    """Test categorical drift detection with stable distribution."""
    reference_stats = {"distribution": {"Month-to-month": 0.5, "One year": 0.3, "Two year": 0.2}}
    current_data = pd.Series(["Month-to-month"] * 50 + ["One year"] * 30 + ["Two year"] * 20)

    result = detect_categorical_drift(reference_stats, current_data, threshold=0.25)

    assert result["drift_detected"] is False
    assert result["metrics"]["psi"] < 0.1


@pytest.mark.unit
def test_detect_categorical_drift_detected():
    """Test categorical drift detection catches significant shift."""
    reference_stats = {"distribution": {"Month-to-month": 0.5, "One year": 0.3, "Two year": 0.2}}
    current_data = pd.Series(
        ["Month-to-month"] * 20 + ["One year"] * 20 + ["Two year"] * 60  # Shifted to Two year
    )

    result = detect_categorical_drift(reference_stats, current_data, threshold=0.25)

    assert result["drift_detected"] is True
    assert result["metrics"]["psi"] >= 0.25


@pytest.mark.unit
def test_detect_categorical_drift_new_category():
    """Test categorical drift handles new categories."""
    reference_stats = {"distribution": {"A": 0.5, "B": 0.5}}
    current_data = pd.Series(["A"] * 40 + ["B"] * 40 + ["C"] * 20)  # New category 'C'

    result = detect_categorical_drift(reference_stats, current_data, threshold=0.25)

    # Should flag as drift (new category is significant change)
    assert result["drift_detected"] is True


@pytest.mark.unit
def test_detect_categorical_drift_missing_category():
    """Test categorical drift handles missing categories."""
    reference_stats = {"distribution": {"A": 0.4, "B": 0.4, "C": 0.2}}
    current_data = pd.Series(["A"] * 50 + ["B"] * 50)  # Category 'C' missing

    result = detect_categorical_drift(reference_stats, current_data, threshold=0.25)

    # Should flag as drift (missing category is significant)
    assert result["drift_detected"] is True


# =============================================================================
# Prediction Drift Tests
# =============================================================================


@pytest.mark.unit
def test_detect_prediction_drift_stable():
    """Test prediction drift detection with stable churn rate."""
    reference_rate = 0.27

    # Generate predictions with ~27% positive rate
    current_predictions = np.random.choice([0, 1], 1000, p=[0.73, 0.27])

    result = detect_prediction_drift(reference_rate, current_predictions, threshold=0.1)

    assert result["drift_detected"] is False
    assert abs(result["metrics"]["absolute_change"]) < 0.1


@pytest.mark.unit
def test_detect_prediction_drift_detected():
    """Test prediction drift detection catches significant shift."""
    reference_rate = 0.27

    # Generate predictions with ~50% positive rate (major shift)
    current_predictions = np.random.choice([0, 1], 1000, p=[0.50, 0.50])

    result = detect_prediction_drift(reference_rate, current_predictions, threshold=0.1)

    assert result["drift_detected"] is True
    assert result["metrics"]["absolute_change"] > 0.1


@pytest.mark.unit
def test_detect_prediction_drift_all_zeros():
    """Test prediction drift handles edge case of all negative predictions."""
    reference_rate = 0.27
    current_predictions = np.zeros(100)  # All 0s

    result = detect_prediction_drift(reference_rate, current_predictions, threshold=0.1)

    # Should detect drift (going from 27% to 0% is significant)
    assert result["drift_detected"] is True
    assert result["metrics"]["current_positive_rate"] == 0.0


@pytest.mark.unit
def test_detect_prediction_drift_all_ones():
    """Test prediction drift handles edge case of all positive predictions."""
    reference_rate = 0.27
    current_predictions = np.ones(100)  # All 1s

    result = detect_prediction_drift(reference_rate, current_predictions, threshold=0.1)

    # Should detect drift (going from 27% to 100% is significant)
    assert result["drift_detected"] is True
    assert result["metrics"]["current_positive_rate"] == 1.0


# =============================================================================
# Comprehensive Drift Analysis Tests
# =============================================================================


@pytest.mark.unit
def test_analyze_drift_no_drift(sample_reference_stats, sample_features):
    """Test analyze_drift with no drift detected."""
    # Use first 50 rows (should be similar to reference)
    current_data = sample_features.head(50)
    predictions = np.random.choice([0, 1], 50, p=[0.73, 0.27])

    result = analyze_drift(
        reference_stats=sample_reference_stats,
        current_data=current_data,
        predictions=predictions,
        numeric_threshold=0.3,  # Lenient for synthetic data
        categorical_threshold=0.3,
        prediction_threshold=0.3,
    )

    assert isinstance(result, dict)
    assert "overall_drift_detected" in result
    assert "summary" in result
    assert "n_features_drifted" in result["summary"]


@pytest.mark.unit
def test_analyze_drift_numeric_feature():
    """Test analyze_drift detects numeric feature drift."""
    reference_stats = {
        "numeric": {"tenure": {"mean": 30.0, "std": 10.0, "min": 0, "max": 72}},
        "categorical": {},
        "target": {"positive_rate": 0.27},
    }

    # Create data with shifted tenure
    current_data = pd.DataFrame(
        {"tenure": np.random.normal(60, 10, 100)}  # Mean shifted significantly
    )
    predictions = np.random.choice([0, 1], 100, p=[0.73, 0.27])

    result = analyze_drift(
        reference_stats=reference_stats,
        current_data=current_data,
        predictions=predictions,
        numeric_threshold=0.2,
        categorical_threshold=0.25,
        prediction_threshold=0.1,
    )

    assert result["overall_drift_detected"] is True
    assert "tenure" in result["summary"]["drifted_features"]


@pytest.mark.unit
def test_analyze_drift_summary_counts():
    """Test analyze_drift summary provides correct counts."""
    reference_stats = {
        "numeric": {
            "feature1": {"mean": 30.0, "std": 10.0},
            "feature2": {"mean": 50.0, "std": 15.0},
        },
        "categorical": {"feature3": {"distribution": {"A": 0.5, "B": 0.5}}},
        "target": {"positive_rate": 0.27},
    }

    current_data = pd.DataFrame(
        {
            "feature1": np.random.normal(30, 10, 100),  # No drift
            "feature2": np.random.normal(30, 10, 100),  # Drift (mean shifted)
            "feature3": ["A"] * 50 + ["B"] * 50,  # No drift
        }
    )
    predictions = np.random.choice([0, 1], 100, p=[0.73, 0.27])

    result = analyze_drift(
        reference_stats=reference_stats,
        current_data=current_data,
        predictions=predictions,
        numeric_threshold=0.2,
        categorical_threshold=0.25,
        prediction_threshold=0.1,
    )

    assert "summary" in result
    assert "n_features_drifted" in result["summary"]
    # At least the shifted feature2 should be detected
    assert result["summary"]["n_features_drifted"] >= 1


# =============================================================================
# Regression Tests: negative-mean denominator and KS-test activation
# =============================================================================


@pytest.mark.unit
def test_detect_numeric_drift_negative_reference_mean():
    """A shifted negative-mean feature must register relative drift.

    Regression test: with a signed denominator the relative change comes out
    negative for negative-mean references, so drift could never trigger.
    """
    reference = {"mean": -10.0, "std": 2.0}
    rng = np.random.default_rng(42)
    current = pd.Series(rng.normal(-13.0, 2.0, 500))  # 30% mean shift

    result = detect_numeric_drift(reference, current, threshold=0.2)

    assert result["metrics"]["mean_change"] > 0
    assert result["metrics"]["relative_drift_detected"] is True
    assert result["drift_detected"] is True


@pytest.mark.unit
def test_production_reference_stats_activate_ks_test():
    """The stats produced by train.compute_reference_statistics must switch on
    the KS branch of detect_numeric_drift (it is skipped without 'samples').

    Regression test: the KS path was dead code in production because the
    training pipeline never stored reference samples in the metadata.
    """
    from train import compute_reference_statistics

    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {"tenure": rng.normal(30.0, 10.0, 2000), "contract": rng.choice(["month", "year"], 2000)}
    )
    y = pd.Series(rng.integers(0, 2, 2000))

    stats = compute_reference_statistics(X, y, ["tenure"], ["contract"])

    assert "samples" in stats["numeric"]["tenure"]
    assert len(stats["numeric"]["tenure"]["samples"]) == 1000  # capped

    # Same distribution: the KS test actually runs and reports its statistic
    same = pd.Series(rng.normal(30.0, 10.0, 500))
    result = detect_numeric_drift(stats["numeric"]["tenure"], same)
    assert result["metrics"]["ks_statistic"] is not None

    # Clear distribution shift: the KS branch flags drift
    shifted = pd.Series(rng.normal(60.0, 10.0, 500))
    result = detect_numeric_drift(stats["numeric"]["tenure"], shifted)
    assert result["metrics"]["ks_drift_detected"] is True
