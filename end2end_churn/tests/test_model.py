"""
Unit tests for trained model behavior.

Tests model predictions, probabilities, invariances, and robustness.

Comprehensive Testing
"""

import numpy as np
import pandas as pd
import pytest

# =============================================================================
# Model Output Shape Tests
# =============================================================================


@pytest.mark.unit
def test_model_predicts_correct_shape(trained_model, sample_features):
    """Test model predictions have correct shape."""
    predictions = trained_model.predict(sample_features)

    assert predictions.shape == (len(sample_features),)
    assert isinstance(predictions, np.ndarray)


@pytest.mark.unit
def test_model_predictions_are_binary(trained_model, sample_features):
    """Test model predictions are binary (0 or 1)."""
    predictions = trained_model.predict(sample_features)

    unique_values = set(predictions)
    assert unique_values.issubset({0, 1})


@pytest.mark.unit
def test_model_predict_proba_shape(trained_model, sample_features):
    """Test probability predictions have correct shape."""
    probabilities = trained_model.predict_proba(sample_features)

    # Should be (n_samples, 2) for binary classification
    assert probabilities.shape == (len(sample_features), 2)


# =============================================================================
# Probability Tests
# =============================================================================


@pytest.mark.unit
def test_model_probabilities_sum_to_one(trained_model, sample_features):
    """Test prediction probabilities sum to 1 for each sample."""
    probabilities = trained_model.predict_proba(sample_features)

    row_sums = probabilities.sum(axis=1)
    assert np.allclose(row_sums, 1.0)


@pytest.mark.unit
def test_model_probabilities_in_range(trained_model, sample_features):
    """Test all probabilities are between 0 and 1."""
    probabilities = trained_model.predict_proba(sample_features)

    assert (probabilities >= 0).all()
    assert (probabilities <= 1).all()


@pytest.mark.unit
def test_model_probability_consistency(trained_model, sample_features):
    """Test class probabilities are consistent with predictions."""
    predictions = trained_model.predict(sample_features)
    probabilities = trained_model.predict_proba(sample_features)

    # Predicted class should match argmax of probabilities
    predicted_classes = np.argmax(probabilities, axis=1)
    assert np.array_equal(predictions, predicted_classes)


# =============================================================================
# Robustness Tests
# =============================================================================


@pytest.mark.unit
def test_model_handles_missing_values(trained_model, sample_features):
    """Test model handles NaN values (via imputation in pipeline)."""
    features_with_nan = sample_features.copy()

    # Introduce NaN in numeric column
    features_with_nan.iloc[0, features_with_nan.columns.get_loc("tenure")] = np.nan

    # Introduce NaN in categorical column
    features_with_nan.iloc[1, features_with_nan.columns.get_loc("gender")] = np.nan

    # Should not raise error, should not produce NaN predictions
    predictions = trained_model.predict(features_with_nan)

    assert not np.isnan(predictions).any()
    assert len(predictions) == len(features_with_nan)


@pytest.mark.unit
def test_model_handles_single_sample(trained_model, sample_features):
    """Test model can predict on single sample."""
    single_sample = sample_features.iloc[[0]]

    prediction = trained_model.predict(single_sample)
    probability = trained_model.predict_proba(single_sample)

    assert prediction.shape == (1,)
    assert probability.shape == (1, 2)


@pytest.mark.unit
def test_model_handles_large_batch(trained_model, sample_features):
    """Test model handles predictions on large batch."""
    # Replicate data to create larger batch
    large_batch = pd.concat([sample_features] * 10, ignore_index=True)

    predictions = trained_model.predict(large_batch)

    assert len(predictions) == len(large_batch)


# =============================================================================
# Determinism Tests
# =============================================================================


@pytest.mark.unit
def test_model_deterministic(trained_model, sample_features):
    """Test model gives same predictions for same input."""
    pred1 = trained_model.predict(sample_features)
    pred2 = trained_model.predict(sample_features)

    assert np.array_equal(pred1, pred2)


@pytest.mark.unit
def test_model_deterministic_probabilities(trained_model, sample_features):
    """Test model gives same probabilities for same input."""
    prob1 = trained_model.predict_proba(sample_features)
    prob2 = trained_model.predict_proba(sample_features)

    assert np.allclose(prob1, prob2)


# =============================================================================
# Model Properties Tests
# =============================================================================


@pytest.mark.unit
def test_model_has_feature_importances(trained_model):
    """Test trained model has feature importances attribute."""
    # Pipeline models have feature_importances_ via the classifier
    assert hasattr(trained_model.named_steps["classifier"], "feature_importances_")

    importances = trained_model.named_steps["classifier"].feature_importances_

    # Check importances are valid
    assert len(importances) > 0
    assert (importances >= 0).all()  # Importances should be non-negative
    assert np.isclose(importances.sum(), 1.0)  # Should sum to 1 for Random Forest


@pytest.mark.unit
def test_model_has_classes_attribute(trained_model):
    """Test trained model has classes_ attribute."""
    assert hasattr(trained_model, "classes_")

    classes = trained_model.classes_

    # For binary classification, should be [0, 1]
    assert len(classes) == 2
    assert set(classes) == {0, 1}


# =============================================================================
# Edge Case Tests
# =============================================================================


@pytest.mark.unit
def test_model_handles_all_same_values(trained_model, sample_features):
    """Test model handles features with all same values."""
    # Create data where one feature has all same values
    edge_case_data = sample_features.copy()
    edge_case_data["tenure"] = 30  # All same value

    # Should not raise error
    predictions = trained_model.predict(edge_case_data)

    assert len(predictions) == len(edge_case_data)


@pytest.mark.unit
def test_model_handles_extreme_values(trained_model, sample_features):
    """Test model handles extreme feature values."""
    extreme_data = sample_features.copy()

    # Set extreme values (still within valid range)
    extreme_data.iloc[0, extreme_data.columns.get_loc("tenure")] = 72  # Max
    extreme_data.iloc[1, extreme_data.columns.get_loc("tenure")] = 0  # Min
    extreme_data.iloc[2, extreme_data.columns.get_loc("MonthlyCharges")] = 150  # Above training max

    # Should handle gracefully (StandardScaler will scale appropriately)
    predictions = trained_model.predict(extreme_data)
    probabilities = trained_model.predict_proba(extreme_data)

    assert not np.isnan(predictions).any()
    assert not np.isnan(probabilities).any()
    assert (probabilities >= 0).all() and (probabilities <= 1).all()
