"""
Comprehensive tests for drift detection functionality.

Tests all drift detection methods:
- Population Stability Index (PSI) for categorical features
- Numeric drift detection (relative change + KS test)
- Prediction drift detection
- End-to-end drift analysis pipeline

Comprehensive Drift Detection Tests
"""

from typing import Any

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


class TestPSICalculation:
    """Test Population Stability Index calculation for categorical features."""

    def test_psi_identical_distributions(self):
        """PSI should be near zero for identical distributions."""
        expected = {"A": 0.5, "B": 0.3, "C": 0.2}
        actual = {"A": 0.5, "B": 0.3, "C": 0.2}

        psi, interpretation = calculate_psi(expected, actual)

        assert psi < 0.01, f"PSI should be near 0 for identical distributions, got {psi}"
        assert interpretation == "stable"

    def test_psi_small_change_stable(self):
        """PSI should classify small changes as stable."""
        expected = {"A": 0.5, "B": 0.3, "C": 0.2}
        actual = {"A": 0.52, "B": 0.28, "C": 0.2}  # Small shift

        psi, interpretation = calculate_psi(expected, actual)

        assert psi < 0.1, f"Expected stable PSI (<0.1), got {psi}"
        assert interpretation == "stable"

    def test_psi_moderate_change(self):
        """PSI should detect moderate distribution shift."""
        expected = {"A": 0.6, "B": 0.3, "C": 0.1}
        actual = {"A": 0.45, "B": 0.35, "C": 0.2}  # Larger shift to trigger moderate PSI

        psi, interpretation = calculate_psi(expected, actual)

        assert 0.1 <= psi < 0.25, f"Expected moderate PSI [0.1, 0.25), got {psi}"
        assert interpretation == "moderate_drift"

    def test_psi_significant_change(self):
        """PSI should detect significant distribution shift."""
        expected = {"A": 0.7, "B": 0.2, "C": 0.1}
        actual = {"A": 0.2, "B": 0.3, "C": 0.5}  # Large shift

        psi, interpretation = calculate_psi(expected, actual)

        assert psi >= 0.25, f"Expected significant PSI (>=0.25), got {psi}"
        assert interpretation == "significant_drift"

    def test_psi_new_category_appears(self):
        """PSI should handle new categories appearing in actual data."""
        expected = {"A": 0.6, "B": 0.4}
        actual = {"A": 0.5, "B": 0.3, "C": 0.2}  # New category C

        psi, interpretation = calculate_psi(expected, actual)

        # PSI should be non-zero due to new category
        assert psi > 0, "PSI should detect new category"
        # Could be moderate or significant depending on shift
        assert interpretation in ["moderate_drift", "significant_drift"]

    def test_psi_category_disappears(self):
        """PSI should handle categories disappearing from actual data."""
        expected = {"A": 0.5, "B": 0.3, "C": 0.2}
        actual = {"A": 0.7, "B": 0.3}  # Category C disappeared

        psi, interpretation = calculate_psi(expected, actual)

        # PSI should be non-zero due to missing category
        assert psi > 0, "PSI should detect missing category"

    def test_categorical_drift_threshold_applies_in_detector(self):
        """The drift decision threshold lives in detect_categorical_drift, not calculate_psi."""
        reference_stats = {"distribution": {"A": 0.5, "B": 0.5}}
        current = pd.Series(["A"] * 20 + ["B"] * 80)

        strict = detect_categorical_drift(reference_stats, current, threshold=0.01)
        lenient = detect_categorical_drift(reference_stats, current, threshold=10.0)

        # Same PSI value either way; only the decision changes with the threshold
        assert strict["metrics"]["psi"] == lenient["metrics"]["psi"]
        assert strict["drift_detected"] is True
        assert lenient["drift_detected"] is False


# =============================================================================
# Numeric Drift Detection Tests
# =============================================================================


class TestNumericDrift:
    """Test numeric feature drift detection using relative change and KS test."""

    def test_no_drift_identical_distribution(self):
        """No drift when distributions are identical."""
        np.random.seed(42)
        reference = {"mean": 50.0, "std": 10.0, "samples": np.random.normal(50, 10, 1000)}
        current = pd.Series(np.random.normal(50, 10, 1000))

        result = detect_numeric_drift(reference, current, threshold=0.2)

        assert not result["drift_detected"], "Should not detect drift in identical distributions"
        assert result["reason"] == "stable"
        assert result["metrics"]["mean_change"] < 0.1
        assert result["metrics"]["std_change"] < 0.15

    def test_drift_mean_shift_significant(self):
        """Detect drift when mean shifts significantly (>20%)."""
        np.random.seed(42)
        reference = {"mean": 50.0, "std": 10.0, "samples": np.random.normal(50, 10, 1000)}
        # Shift mean by 30% (50 -> 65)
        current = pd.Series(np.random.normal(65, 10, 1000))

        result = detect_numeric_drift(reference, current, threshold=0.2)

        assert result["drift_detected"], "Should detect significant mean shift"
        assert "drift_detected" in result["reason"]
        assert result["metrics"]["mean_change"] > 0.2
        assert result["metrics"]["relative_drift_detected"]

    def test_drift_std_shift_significant(self):
        """Detect drift when std deviation changes significantly."""
        np.random.seed(42)
        reference = {"mean": 50.0, "std": 10.0, "samples": np.random.normal(50, 10, 1000)}
        # Same mean, but std doubles (10 -> 20)
        current = pd.Series(np.random.normal(50, 20, 1000))

        result = detect_numeric_drift(reference, current, threshold=0.2)

        assert result["drift_detected"], "Should detect std deviation drift"
        assert result["metrics"]["std_change"] > 0.5
        assert result["metrics"]["relative_drift_detected"]

    def test_drift_both_mean_and_std_shift(self):
        """Detect drift when both mean and std shift."""
        np.random.seed(42)
        reference = {"mean": 50.0, "std": 10.0, "samples": np.random.normal(50, 10, 1000)}
        # Both mean and std change
        current = pd.Series(np.random.normal(70, 15, 1000))

        result = detect_numeric_drift(reference, current, threshold=0.2)

        assert result["drift_detected"]
        assert result["metrics"]["mean_change"] > 0.2
        assert result["metrics"]["std_change"] > 0.2

    def test_ks_test_detects_distribution_shape_change(self):
        """KS test should catch distribution shape changes even with similar mean/std."""
        np.random.seed(42)

        # Create bimodal distribution with similar mean/std
        reference_samples = np.random.normal(50, 10, 1000)
        reference = {"mean": 50.0, "std": 10.0, "samples": reference_samples}

        # Bimodal distribution (mixture of two normals)
        current_part1 = np.random.normal(40, 5, 500)
        current_part2 = np.random.normal(60, 5, 500)
        current = pd.Series(np.concatenate([current_part1, current_part2]))

        result = detect_numeric_drift(reference, current, threshold=0.2)

        # KS test should detect this even if relative change doesn't
        assert "ks_statistic" in result["metrics"]
        assert result["metrics"]["ks_statistic"] is not None
        # Drift should be detected by at least one method
        assert result["drift_detected"]

    def test_no_drift_within_threshold(self):
        """No drift when changes are within threshold."""
        np.random.seed(42)
        reference = {"mean": 100.0, "std": 20.0, "samples": np.random.normal(100, 20, 1000)}
        # Small shift: 15% change (within 20% threshold)
        current = pd.Series(np.random.normal(115, 20, 1000))

        result = detect_numeric_drift(reference, current, threshold=0.2)

        # Mean change is ~15%, should be below 20% threshold
        assert result["metrics"]["mean_change"] < 0.2
        # May still be caught by KS test, but relative drift should be false
        assert not result["metrics"]["relative_drift_detected"]

    def test_empty_current_data_handling(self):
        """Handle empty current data gracefully without errors."""
        reference = {"mean": 50.0, "std": 10.0, "samples": np.random.normal(50, 10, 100)}
        current = pd.Series([], dtype=float)

        result = detect_numeric_drift(reference, current)

        assert not result["drift_detected"]
        assert result["reason"] == "insufficient_data"
        assert "metrics" in result

    def test_single_value_current_data(self):
        """Handle single-value current data (edge case)."""
        reference = {"mean": 50.0, "std": 10.0, "samples": np.random.normal(50, 10, 100)}
        current = pd.Series([50.0])  # Single value

        result = detect_numeric_drift(reference, current)

        # Should not crash
        assert "metrics" in result
        # For single value, pandas std() returns NaN
        assert np.isnan(result["metrics"]["current_std"]) or result["metrics"]["current_std"] == 0.0

    def test_missing_ks_samples_fallback(self):
        """Fallback to relative change when KS samples not available."""
        reference = {
            "mean": 50.0,
            "std": 10.0,
            # No 'samples' key
        }
        current = pd.Series(np.random.normal(70, 10, 1000))

        result = detect_numeric_drift(reference, current, threshold=0.2)

        # Should still work with relative change only
        assert result["drift_detected"]
        assert result["metrics"]["ks_statistic"] is None
        assert result["metrics"]["ks_p_value"] is None


# =============================================================================
# Categorical Drift Detection Tests
# =============================================================================


class TestCategoricalDrift:
    """Test categorical feature drift detection using PSI."""

    def test_no_drift_stable_distribution(self):
        """No drift for stable categorical distribution."""
        reference = {"distribution": {"Month-to-month": 0.55, "One year": 0.24, "Two year": 0.21}}
        # Create similar distribution
        current = pd.Series(["Month-to-month"] * 550 + ["One year"] * 240 + ["Two year"] * 210)

        result = detect_categorical_drift(reference, current, threshold=0.25)

        assert not result["drift_detected"]
        assert result["metrics"]["psi"] < 0.1

    def test_drift_proportion_shift_moderate(self):
        """Detect moderate drift when proportions shift."""
        reference = {"distribution": {"Month-to-month": 0.55, "One year": 0.24, "Two year": 0.21}}
        # Moderate shift: Month-to-month drops to 35%, two year increases to 35%
        current = pd.Series(["Month-to-month"] * 350 + ["One year"] * 300 + ["Two year"] * 350)

        result = detect_categorical_drift(reference, current, threshold=0.25)

        # Should detect moderate drift
        psi = result["metrics"]["psi"]
        assert psi >= 0.1, f"Expected PSI >= 0.1, got {psi}"

    def test_drift_proportion_shift_significant(self):
        """Detect significant drift when proportions shift dramatically."""
        reference = {"distribution": {"Month-to-month": 0.55, "One year": 0.24, "Two year": 0.21}}
        # Significant shift: Month-to-month drops to 20%
        current = pd.Series(["Month-to-month"] * 200 + ["One year"] * 400 + ["Two year"] * 400)

        result = detect_categorical_drift(reference, current, threshold=0.25)

        assert result["drift_detected"]
        assert result["metrics"]["psi"] >= 0.25
        assert result["reason"] == "significant_drift"

    def test_new_category_appears(self):
        """Detect drift when new category appears."""
        reference = {"distribution": {"A": 0.6, "B": 0.4}}
        # New category C appears
        current = pd.Series(["A"] * 500 + ["B"] * 300 + ["C"] * 200)

        result = detect_categorical_drift(reference, current, threshold=0.25)

        # New category should trigger drift
        assert result["drift_detected"]
        assert "C" in result["metrics"]["current_distribution"]

    def test_category_disappears(self):
        """Detect drift when category disappears."""
        reference = {"distribution": {"A": 0.5, "B": 0.3, "C": 0.2}}
        # Category C disappears
        current = pd.Series(["A"] * 700 + ["B"] * 300)

        result = detect_categorical_drift(reference, current, threshold=0.25)

        # Missing category should trigger drift
        assert result["drift_detected"]
        assert "C" not in result["metrics"]["current_distribution"]

    def test_custom_psi_threshold(self):
        """Respect custom PSI threshold."""
        reference = {"distribution": {"A": 0.5, "B": 0.5}}
        current = pd.Series(["A"] * 400 + ["B"] * 600)

        # With low threshold (0.05), should detect drift
        result_low = detect_categorical_drift(reference, current, threshold=0.05)

        # With high threshold (0.5), might not detect drift
        result_high = detect_categorical_drift(reference, current, threshold=0.5)

        # Check that threshold is respected
        assert result_low["metrics"]["threshold"] == 0.05
        assert result_high["metrics"]["threshold"] == 0.5


# =============================================================================
# Prediction Drift Detection Tests
# =============================================================================


class TestPredictionDrift:
    """Test prediction drift detection."""

    def test_no_prediction_drift_stable(self):
        """No drift when prediction rate is stable."""
        reference_rate = 0.27
        # Current predictions: 27% positive (same as reference)
        predictions = np.array([1] * 270 + [0] * 730)

        result = detect_prediction_drift(reference_rate, predictions, threshold=0.1)

        assert not result["drift_detected"]
        assert result["reason"] == "stable"
        assert abs(result["metrics"]["absolute_change"]) < 0.01

    def test_prediction_drift_detected_increase(self):
        """Detect drift when prediction rate increases significantly."""
        reference_rate = 0.27
        # Current predictions: 50% positive (23% increase)
        predictions = np.array([1] * 500 + [0] * 500)

        result = detect_prediction_drift(reference_rate, predictions, threshold=0.1)

        assert result["drift_detected"]
        assert result["reason"] == "prediction_shift"
        assert result["metrics"]["absolute_change"] > 0.1
        assert result["metrics"]["current_positive_rate"] > reference_rate

    def test_prediction_drift_detected_decrease(self):
        """Detect drift when prediction rate decreases significantly."""
        reference_rate = 0.27
        # Current predictions: 10% positive (17% decrease)
        predictions = np.array([1] * 100 + [0] * 900)

        result = detect_prediction_drift(reference_rate, predictions, threshold=0.1)

        assert result["drift_detected"]
        assert result["metrics"]["absolute_change"] > 0.1
        assert result["metrics"]["current_positive_rate"] < reference_rate

    def test_prediction_drift_within_threshold(self):
        """No drift when change is within threshold."""
        reference_rate = 0.27
        # Current predictions: 32% positive (5% change, within 10% threshold)
        predictions = np.array([1] * 320 + [0] * 680)

        result = detect_prediction_drift(reference_rate, predictions, threshold=0.1)

        assert not result["drift_detected"]
        assert result["metrics"]["absolute_change"] < 0.1

    def test_prediction_drift_custom_threshold(self):
        """Respect custom threshold parameter."""
        reference_rate = 0.27
        predictions = np.array([1] * 320 + [0] * 680)  # 32% positive

        # With strict threshold (0.03), should detect
        result_strict = detect_prediction_drift(reference_rate, predictions, threshold=0.03)

        # With lenient threshold (0.1), should not detect
        result_lenient = detect_prediction_drift(reference_rate, predictions, threshold=0.1)

        assert result_strict["drift_detected"]
        assert not result_lenient["drift_detected"]

    def test_all_positive_predictions(self):
        """Handle edge case: all predictions positive."""
        reference_rate = 0.27
        predictions = np.array([1] * 1000)

        result = detect_prediction_drift(reference_rate, predictions, threshold=0.1)

        assert result["drift_detected"]
        assert result["metrics"]["current_positive_rate"] == 1.0

    def test_all_negative_predictions(self):
        """Handle edge case: all predictions negative."""
        reference_rate = 0.27
        predictions = np.array([0] * 1000)

        result = detect_prediction_drift(reference_rate, predictions, threshold=0.1)

        assert result["drift_detected"]
        assert result["metrics"]["current_positive_rate"] == 0.0


# =============================================================================
# End-to-End Drift Analysis Tests
# =============================================================================


@pytest.mark.integration
class TestDriftAnalysis:
    """Test complete drift analysis pipeline with multiple features."""

    @pytest.fixture
    def reference_statistics(self):
        """Reference statistics for comprehensive testing."""
        np.random.seed(42)
        return {
            "numeric": {
                "tenure": {"mean": 32.0, "std": 24.0, "samples": np.random.normal(32, 24, 1000)},
                "MonthlyCharges": {
                    "mean": 64.5,
                    "std": 30.0,
                    "samples": np.random.normal(64.5, 30, 1000),
                },
                "TotalCharges": {
                    "mean": 2280.0,
                    "std": 2266.0,
                    "samples": np.random.normal(2280, 2266, 1000),
                },
            },
            "categorical": {
                "Contract": {
                    "distribution": {"Month-to-month": 0.55, "One year": 0.24, "Two year": 0.21}
                },
                "InternetService": {"distribution": {"DSL": 0.24, "Fiber optic": 0.44, "No": 0.32}},
                "PaymentMethod": {
                    "distribution": {
                        "Electronic check": 0.34,
                        "Mailed check": 0.15,
                        "Bank transfer (automatic)": 0.22,
                        "Credit card (automatic)": 0.29,
                    }
                },
            },
            "target": {"positive_rate": 0.27, "n_samples": 5000},
            # Prediction baseline: predicted-positive rate at the tuned threshold
            # (drift compares current predictions against this, not label prevalence)
            "prediction": {
                "threshold": 0.4,
                "positive_rate": 0.27,
                "proba_mean": 0.27,
                "proba_histogram": {
                    "bin_edges": np.linspace(0, 1, 11).tolist(),
                    "proportions": [0.30, 0.18, 0.13, 0.10, 0.08, 0.07, 0.06, 0.04, 0.02, 0.02],
                },
                "n_samples": 5000,
            },
        }

    def test_no_drift_all_features_stable(self, reference_statistics):
        """Test full analysis with no drift in any feature."""
        np.random.seed(42)

        # Create similar data to reference
        current_data = pd.DataFrame(
            {
                "tenure": np.random.normal(32, 24, 1000),
                "MonthlyCharges": np.random.normal(64.5, 30, 1000),
                "TotalCharges": np.random.normal(2280, 2266, 1000),
                "Contract": np.random.choice(
                    ["Month-to-month", "One year", "Two year"], size=1000, p=[0.55, 0.24, 0.21]
                ),
                "InternetService": np.random.choice(
                    ["DSL", "Fiber optic", "No"], size=1000, p=[0.24, 0.44, 0.32]
                ),
                "PaymentMethod": np.random.choice(
                    [
                        "Electronic check",
                        "Mailed check",
                        "Bank transfer (automatic)",
                        "Credit card (automatic)",
                    ],
                    size=1000,
                    p=[0.34, 0.15, 0.22, 0.29],
                ),
            }
        )
        predictions = np.random.choice([0, 1], size=1000, p=[0.73, 0.27])

        report = analyze_drift(
            reference_statistics,
            current_data,
            predictions,
            numeric_threshold=0.2,
            categorical_threshold=0.25,
            prediction_threshold=0.1,
        )

        assert not report["overall_drift_detected"]
        assert report["summary"]["n_features_drifted"] == 0
        assert len(report["summary"]["drifted_features"]) == 0
        assert not report["prediction_drift"]["drift_detected"]

    def test_drift_single_numeric_feature(self, reference_statistics):
        """Detect drift when single numeric feature drifts."""
        np.random.seed(42)

        current_data = pd.DataFrame(
            {
                "tenure": np.random.normal(50, 24, 1000),  # Shifted from 32 to 50
                "MonthlyCharges": np.random.normal(64.5, 30, 1000),  # Stable
                "TotalCharges": np.random.normal(2280, 2266, 1000),  # Stable
                "Contract": np.random.choice(
                    ["Month-to-month", "One year", "Two year"], size=1000, p=[0.55, 0.24, 0.21]
                ),
                "InternetService": np.random.choice(
                    ["DSL", "Fiber optic", "No"], size=1000, p=[0.24, 0.44, 0.32]
                ),
                "PaymentMethod": np.random.choice(
                    [
                        "Electronic check",
                        "Mailed check",
                        "Bank transfer (automatic)",
                        "Credit card (automatic)",
                    ],
                    size=1000,
                    p=[0.34, 0.15, 0.22, 0.29],
                ),
            }
        )
        predictions = np.random.choice([0, 1], size=1000, p=[0.73, 0.27])

        report = analyze_drift(reference_statistics, current_data, predictions)

        assert report["overall_drift_detected"]
        assert report["summary"]["n_features_drifted"] >= 1
        assert "tenure" in report["summary"]["drifted_features"]
        assert report["numeric_features"]["tenure"]["drift_detected"]

    def test_drift_single_categorical_feature(self, reference_statistics):
        """Detect drift when single categorical feature drifts."""
        np.random.seed(42)

        current_data = pd.DataFrame(
            {
                "tenure": np.random.normal(32, 24, 1000),  # Stable
                "MonthlyCharges": np.random.normal(64.5, 30, 1000),  # Stable
                "TotalCharges": np.random.normal(2280, 2266, 1000),  # Stable
                "Contract": np.random.choice(  # Shifted distribution
                    ["Month-to-month", "One year", "Two year"],
                    size=1000,
                    p=[0.2, 0.4, 0.4],  # Changed from [0.55, 0.24, 0.21]
                ),
                "InternetService": np.random.choice(
                    ["DSL", "Fiber optic", "No"], size=1000, p=[0.24, 0.44, 0.32]
                ),
                "PaymentMethod": np.random.choice(
                    [
                        "Electronic check",
                        "Mailed check",
                        "Bank transfer (automatic)",
                        "Credit card (automatic)",
                    ],
                    size=1000,
                    p=[0.34, 0.15, 0.22, 0.29],
                ),
            }
        )
        predictions = np.random.choice([0, 1], size=1000, p=[0.73, 0.27])

        report = analyze_drift(reference_statistics, current_data, predictions)

        assert report["overall_drift_detected"]
        assert "Contract" in report["summary"]["drifted_features"]
        assert report["categorical_features"]["Contract"]["drift_detected"]

    def test_drift_multiple_features(self, reference_statistics):
        """Detect drift across multiple features simultaneously."""
        np.random.seed(42)

        current_data = pd.DataFrame(
            {
                "tenure": np.random.normal(50, 24, 1000),  # Drifted
                "MonthlyCharges": np.random.normal(80, 30, 1000),  # Drifted
                "TotalCharges": np.random.normal(2280, 2266, 1000),  # Stable
                "Contract": np.random.choice(  # Drifted
                    ["Month-to-month", "One year", "Two year"], size=1000, p=[0.2, 0.4, 0.4]
                ),
                "InternetService": np.random.choice(  # Stable
                    ["DSL", "Fiber optic", "No"], size=1000, p=[0.24, 0.44, 0.32]
                ),
                "PaymentMethod": np.random.choice(  # Stable
                    [
                        "Electronic check",
                        "Mailed check",
                        "Bank transfer (automatic)",
                        "Credit card (automatic)",
                    ],
                    size=1000,
                    p=[0.34, 0.15, 0.22, 0.29],
                ),
            }
        )
        predictions = np.random.choice([0, 1], size=1000, p=[0.5, 0.5])  # Drifted

        report = analyze_drift(reference_statistics, current_data, predictions)

        assert report["overall_drift_detected"]
        assert report["summary"]["n_features_drifted"] >= 3
        assert "tenure" in report["summary"]["drifted_features"]
        assert "MonthlyCharges" in report["summary"]["drifted_features"]
        assert "Contract" in report["summary"]["drifted_features"]
        assert report["prediction_drift"]["drift_detected"]

    def test_drift_only_predictions_shift(self, reference_statistics):
        """Detect drift when only predictions shift (not input features)."""
        np.random.seed(42)

        # All features stable
        current_data = pd.DataFrame(
            {
                "tenure": np.random.normal(32, 24, 1000),
                "MonthlyCharges": np.random.normal(64.5, 30, 1000),
                "TotalCharges": np.random.normal(2280, 2266, 1000),
                "Contract": np.random.choice(
                    ["Month-to-month", "One year", "Two year"], size=1000, p=[0.55, 0.24, 0.21]
                ),
                "InternetService": np.random.choice(
                    ["DSL", "Fiber optic", "No"], size=1000, p=[0.24, 0.44, 0.32]
                ),
                "PaymentMethod": np.random.choice(
                    [
                        "Electronic check",
                        "Mailed check",
                        "Bank transfer (automatic)",
                        "Credit card (automatic)",
                    ],
                    size=1000,
                    p=[0.34, 0.15, 0.22, 0.29],
                ),
            }
        )
        # But predictions drift significantly
        predictions = np.array([1] * 500 + [0] * 500)  # 50% instead of 27%

        report = analyze_drift(reference_statistics, current_data, predictions)

        # Overall drift should be detected due to prediction drift
        assert report["overall_drift_detected"]
        assert report["prediction_drift"]["drift_detected"]
        # Feature drift count should be low/zero
        assert report["summary"]["n_features_drifted"] <= 2  # Allow for statistical noise

    def test_missing_features_in_current_data(self, reference_statistics):
        """Handle gracefully when current data missing some features."""
        np.random.seed(42)

        # Current data only has 2 of 3 numeric features
        current_data = pd.DataFrame(
            {
                "tenure": np.random.normal(32, 24, 1000),
                "MonthlyCharges": np.random.normal(64.5, 30, 1000),
                # TotalCharges missing
                "Contract": np.random.choice(
                    ["Month-to-month", "One year", "Two year"], size=1000, p=[0.55, 0.24, 0.21]
                ),
                "InternetService": np.random.choice(
                    ["DSL", "Fiber optic", "No"], size=1000, p=[0.24, 0.44, 0.32]
                ),
                "PaymentMethod": np.random.choice(
                    [
                        "Electronic check",
                        "Mailed check",
                        "Bank transfer (automatic)",
                        "Credit card (automatic)",
                    ],
                    size=1000,
                    p=[0.34, 0.15, 0.22, 0.29],
                ),
            }
        )
        predictions = np.random.choice([0, 1], size=1000, p=[0.73, 0.27])

        # Should not crash, just skip missing feature
        report = analyze_drift(reference_statistics, current_data, predictions)

        assert "tenure" in report["numeric_features"]
        assert "MonthlyCharges" in report["numeric_features"]
        # TotalCharges not in current data, so not analyzed
        assert "TotalCharges" not in report["numeric_features"]

    def test_report_structure_complete(self, reference_statistics):
        """Verify complete report structure with all expected fields."""
        np.random.seed(42)

        current_data = pd.DataFrame(
            {
                "tenure": np.random.normal(32, 24, 500),
                "MonthlyCharges": np.random.normal(64.5, 30, 500),
                "TotalCharges": np.random.normal(2280, 2266, 500),
                "Contract": np.random.choice(
                    ["Month-to-month", "One year", "Two year"], size=500, p=[0.55, 0.24, 0.21]
                ),
                "InternetService": np.random.choice(
                    ["DSL", "Fiber optic", "No"], size=500, p=[0.24, 0.44, 0.32]
                ),
                "PaymentMethod": np.random.choice(
                    [
                        "Electronic check",
                        "Mailed check",
                        "Bank transfer (automatic)",
                        "Credit card (automatic)",
                    ],
                    size=500,
                    p=[0.34, 0.15, 0.22, 0.29],
                ),
            }
        )
        predictions = np.random.choice([0, 1], size=500, p=[0.73, 0.27])

        report = analyze_drift(reference_statistics, current_data, predictions)

        # Verify top-level structure
        assert "overall_drift_detected" in report
        assert isinstance(report["overall_drift_detected"], bool)

        assert "numeric_features" in report
        assert isinstance(report["numeric_features"], dict)

        assert "categorical_features" in report
        assert isinstance(report["categorical_features"], dict)

        assert "prediction_drift" in report
        assert isinstance(report["prediction_drift"], dict)

        assert "summary" in report
        assert "n_features_drifted" in report["summary"]
        assert "drifted_features" in report["summary"]
        assert isinstance(report["summary"]["drifted_features"], list)

        # Verify numeric feature results have expected structure
        for feature, result in report["numeric_features"].items():
            assert "drift_detected" in result
            assert "reason" in result
            assert "metrics" in result

        # Verify categorical feature results
        for feature, result in report["categorical_features"].items():
            assert "drift_detected" in result
            assert "reason" in result
            assert "metrics" in result
            assert "psi" in result["metrics"]

    def test_custom_thresholds_respected(self, reference_statistics):
        """Verify custom thresholds are respected in analysis."""
        np.random.seed(42)

        # Create data with moderate drift
        current_data = pd.DataFrame(
            {
                "tenure": np.random.normal(38, 24, 500),  # 6/32 = 18.75% change
                "MonthlyCharges": np.random.normal(64.5, 30, 500),
                "TotalCharges": np.random.normal(2280, 2266, 500),
                "Contract": np.random.choice(
                    ["Month-to-month", "One year", "Two year"], size=500, p=[0.55, 0.24, 0.21]
                ),
                "InternetService": np.random.choice(
                    ["DSL", "Fiber optic", "No"], size=500, p=[0.24, 0.44, 0.32]
                ),
                "PaymentMethod": np.random.choice(
                    [
                        "Electronic check",
                        "Mailed check",
                        "Bank transfer (automatic)",
                        "Credit card (automatic)",
                    ],
                    size=500,
                    p=[0.34, 0.15, 0.22, 0.29],
                ),
            }
        )
        predictions = np.random.choice([0, 1], size=500, p=[0.73, 0.27])

        # With strict threshold (0.1), should detect tenure drift
        report_strict = analyze_drift(
            reference_statistics,
            current_data,
            predictions,
            numeric_threshold=0.1,  # Strict
            categorical_threshold=0.25,
            prediction_threshold=0.1,
        )

        # With lenient threshold (0.3), might not detect
        report_lenient = analyze_drift(
            reference_statistics,
            current_data,
            predictions,
            numeric_threshold=0.3,  # Lenient
            categorical_threshold=0.25,
            prediction_threshold=0.1,
        )

        # Strict should detect more drift
        assert (
            report_strict["summary"]["n_features_drifted"]
            >= report_lenient["summary"]["n_features_drifted"]
        )


# =============================================================================
# Performance and Edge Cases
# =============================================================================


@pytest.mark.unit
class TestDriftEdgeCases:
    """Test edge cases and error handling in drift detection."""

    def test_numeric_drift_with_nan_values(self):
        """Handle NaN values in numeric data gracefully."""
        reference = {"mean": 50.0, "std": 10.0, "samples": np.random.normal(50, 10, 100)}
        # Current data with some NaN values
        current_values = np.random.normal(50, 10, 100)
        current_values[0:10] = np.nan
        current = pd.Series(current_values)

        result = detect_numeric_drift(reference, current)

        # Should handle NaN by dropping them
        assert "metrics" in result
        assert not np.isnan(result["metrics"]["current_mean"])

    def test_categorical_drift_with_nan_category(self):
        """Handle NaN as a category in categorical data."""
        reference = {"distribution": {"A": 0.5, "B": 0.5}}
        # Current data with NaN values
        current = pd.Series(["A"] * 400 + ["B"] * 400 + [np.nan] * 200)

        result = detect_categorical_drift(reference, current)

        # Should handle NaN as a category
        assert "metrics" in result
        assert "current_distribution" in result["metrics"]

    def test_zero_std_reference_data(self):
        """Handle constant-value reference data (zero std)."""
        reference = {
            "mean": 50.0,
            "std": 0.0,  # All values are 50.0
            "samples": np.array([50.0] * 100),
        }
        current = pd.Series(np.random.normal(50, 10, 100))

        result = detect_numeric_drift(reference, current)

        # Should not crash due to division by zero
        assert "metrics" in result
        # std_change calculation should handle zero denominator
        assert "std_change" in result["metrics"]

    def test_large_dataset_performance(self):
        """Ensure drift detection performs well on large datasets."""
        import time

        np.random.seed(42)
        reference = {"mean": 50.0, "std": 10.0, "samples": np.random.normal(50, 10, 10000)}
        current = pd.Series(np.random.normal(50, 10, 10000))

        start = time.time()
        result = detect_numeric_drift(reference, current)
        duration = time.time() - start

        assert result is not None
        # Should complete in under 1 second
        assert duration < 1.0, f"Drift detection took {duration:.3f}s (too slow)"
