"""
Unit tests for data preprocessing utilities.

Tests data loading, preprocessing, and split creation.

Comprehensive Testing
"""

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import create_three_way_split, preprocess_data

# =============================================================================
# Three-Way Split Tests
# =============================================================================


@pytest.mark.unit
def test_create_three_way_split_proportions(sample_features, sample_target):
    """Test three-way split creates correct proportions."""
    X_train, X_val, X_test, y_train, y_val, y_test = create_three_way_split(
        sample_features, sample_target, test_size=0.2, val_size=0.25, random_state=42
    )

    total_samples = len(sample_features)

    # Check sizes
    assert len(X_train) + len(X_val) + len(X_test) == total_samples
    assert len(y_train) + len(y_val) + len(y_test) == total_samples

    # Check proportions (approximately)
    # test_size=0.2 means 20% test
    # val_size=0.25 means 25% of remaining 80% = 20% of total
    # So: 60% train, 20% val, 20% test
    assert abs(len(X_test) / total_samples - 0.2) < 0.05
    assert abs(len(X_val) / total_samples - 0.2) < 0.05
    assert abs(len(X_train) / total_samples - 0.6) < 0.05


@pytest.mark.unit
def test_create_three_way_split_stratification(sample_features, sample_target):
    """Test stratification maintains class distribution."""
    X_train, X_val, X_test, y_train, y_val, y_test = create_three_way_split(
        sample_features, sample_target, test_size=0.2, val_size=0.25, random_state=42, stratify=True
    )

    # Calculate class proportions
    overall_positive_rate = sample_target.mean()
    train_positive_rate = y_train.mean()
    val_positive_rate = y_val.mean()
    test_positive_rate = y_test.mean()

    # All splits should have similar positive rates (within 10%)
    assert abs(train_positive_rate - overall_positive_rate) < 0.1
    assert abs(val_positive_rate - overall_positive_rate) < 0.1
    assert abs(test_positive_rate - overall_positive_rate) < 0.1


@pytest.mark.unit
def test_create_three_way_split_no_stratification(sample_features, sample_target):
    """Test split works without stratification."""
    X_train, X_val, X_test, y_train, y_val, y_test = create_three_way_split(
        sample_features,
        sample_target,
        test_size=0.2,
        val_size=0.25,
        random_state=42,
        stratify=False,  # No stratification
    )

    # Should still create splits
    total_samples = len(sample_features)
    assert len(X_train) + len(X_val) + len(X_test) == total_samples


@pytest.mark.unit
def test_create_three_way_split_reproducibility(sample_features, sample_target):
    """Test same random_state produces same splits."""
    split1 = create_three_way_split(sample_features, sample_target, random_state=42)
    split2 = create_three_way_split(sample_features, sample_target, random_state=42)

    X_train1, X_val1, X_test1, y_train1, y_val1, y_test1 = split1
    X_train2, X_val2, X_test2, y_train2, y_val2, y_test2 = split2

    # Indices should be identical
    assert X_train1.index.tolist() == X_train2.index.tolist()
    assert X_val1.index.tolist() == X_val2.index.tolist()
    assert X_test1.index.tolist() == X_test2.index.tolist()


@pytest.mark.unit
def test_create_three_way_split_no_data_leakage(sample_features, sample_target):
    """Test splits don't overlap (no data leakage)."""
    X_train, X_val, X_test, y_train, y_val, y_test = create_three_way_split(
        sample_features, sample_target, random_state=42
    )

    train_indices = set(X_train.index)
    val_indices = set(X_val.index)
    test_indices = set(X_test.index)

    # No overlap between sets
    assert len(train_indices & val_indices) == 0
    assert len(train_indices & test_indices) == 0
    assert len(val_indices & test_indices) == 0

    # All indices accounted for
    all_indices = train_indices | val_indices | test_indices
    assert len(all_indices) == len(sample_features)


# =============================================================================
# Preprocessing Tests
# =============================================================================


@pytest.mark.unit
def test_preprocess_data_separates_features_target(sample_data):
    """Test preprocess_data correctly separates X and y."""
    # Add customerID column as expected by preprocess_data
    test_data = sample_data.copy()
    test_data.insert(0, "customerID", [f"ID_{i:04d}" for i in range(len(test_data))])

    X, y = preprocess_data(test_data)

    # Check shapes (should be one less than input due to customerID removal)
    assert len(X) == len(test_data)
    assert len(y) == len(test_data)

    # Check 'Churn' not in features
    assert "Churn" not in X.columns
    # Check 'customerID' not in features
    assert "customerID" not in X.columns

    # Check target is binary (0/1)
    assert set(y.unique()).issubset({0, 1})


@pytest.mark.unit
def test_preprocess_data_feature_count(sample_data):
    """Test preprocess_data returns expected number of features."""
    # Add customerID column
    test_data = sample_data.copy()
    test_data.insert(0, "customerID", [f"ID_{i:04d}" for i in range(len(test_data))])

    X, y = preprocess_data(test_data)

    # Should have all columns except 'Churn' and 'customerID'
    expected_feature_count = len(test_data.columns) - 2
    assert X.shape[1] == expected_feature_count


@pytest.mark.unit
def test_preprocess_data_preserves_indices(sample_data):
    """Test preprocess_data preserves DataFrame indices."""
    # Add customerID column
    test_data = sample_data.copy()
    test_data.insert(0, "customerID", [f"ID_{i:04d}" for i in range(len(test_data))])

    X, y = preprocess_data(test_data)

    assert X.index.tolist() == test_data.index.tolist()
    assert y.index.tolist() == test_data.index.tolist()


@pytest.mark.unit
def test_preprocess_data_handles_missing_churn_column():
    """Test preprocess_data raises error if 'Churn' column missing."""
    # Create data without 'Churn' column
    data_no_churn = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})

    with pytest.raises(KeyError):
        preprocess_data(data_no_churn)


@pytest.mark.unit
def test_preprocess_data_churn_encoding():
    """Test 'Churn' column is correctly encoded to 0/1."""
    data = pd.DataFrame(
        {
            "customerID": ["ID_0001", "ID_0002", "ID_0003", "ID_0004", "ID_0005"],
            "feature1": [1, 2, 3, 4, 5],
            "TotalCharges": [100, 200, 300, 400, 500],
            "Churn": ["No", "Yes", "No", "Yes", "Yes"],
        }
    )

    X, y = preprocess_data(data)

    # Verify encoding
    assert y.tolist() == [0, 1, 0, 1, 1]


@pytest.mark.unit
def test_preprocess_data_empty_dataframe():
    """Test preprocess_data handles empty DataFrame."""
    empty_data = pd.DataFrame({"customerID": [], "feature1": [], "TotalCharges": [], "Churn": []})

    X, y = preprocess_data(empty_data)

    assert len(X) == 0
    assert len(y) == 0
