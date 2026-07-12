"""
Shared test fixtures for the test suite.

This module provides reusable fixtures for test data, configurations,
trained models, and test utilities.
"""

import shutil
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest


def convert_numpy_types(obj):
    """
    Convert numpy types to native Python types for Pydantic validation.

    Pandas DataFrames often contain numpy scalar types (np.int64, np.float64)
    which Pydantic rejects. This function recursively converts them to native
    Python types (int, float, str, bool).

    Args:
        obj: Object that may contain numpy types

    Returns:
        Object with numpy types converted to Python natives
    """
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.str_, str)):
        return str(obj)
    else:
        return obj


@pytest.fixture(scope="session")
def sample_data():
    """
    Create sample customer data for testing.

    Returns 100 synthetic customer records with realistic distributions
    matching the Telco Customer Churn dataset schema.

    TotalCharges is calculated based on tenure and MonthlyCharges to satisfy
    the validation constraint: TotalCharges <= tenure * MonthlyCharges * 1.2
    """
    np.random.seed(42)
    n_samples = 100

    # Generate tenure and monthly charges first
    tenure = np.random.randint(0, 72, n_samples)
    monthly_charges = np.random.uniform(18, 120, n_samples)

    # Calculate TotalCharges realistically (with some variation but within validator limits)
    # TotalCharges should be approximately tenure * MonthlyCharges
    # Add some noise but keep within validation constraint (<=tenure * monthly * 1.2)
    total_charges = tenure * monthly_charges * np.random.uniform(0.7, 1.1, n_samples)
    # Ensure minimum value for customers with 0 tenure
    total_charges = np.maximum(total_charges, monthly_charges * 0.5)

    data = {
        "gender": np.random.choice(["Male", "Female"], n_samples),
        "SeniorCitizen": np.random.choice([0, 1], n_samples),
        "Partner": np.random.choice(["Yes", "No"], n_samples),
        "Dependents": np.random.choice(["Yes", "No"], n_samples),
        "tenure": tenure,
        "PhoneService": np.random.choice(["Yes", "No"], n_samples),
        "MultipleLines": np.random.choice(["Yes", "No", "No phone service"], n_samples),
        "InternetService": np.random.choice(["DSL", "Fiber optic", "No"], n_samples),
        "OnlineSecurity": np.random.choice(["Yes", "No", "No internet service"], n_samples),
        "OnlineBackup": np.random.choice(["Yes", "No", "No internet service"], n_samples),
        "DeviceProtection": np.random.choice(["Yes", "No", "No internet service"], n_samples),
        "TechSupport": np.random.choice(["Yes", "No", "No internet service"], n_samples),
        "StreamingTV": np.random.choice(["Yes", "No", "No internet service"], n_samples),
        "StreamingMovies": np.random.choice(["Yes", "No", "No internet service"], n_samples),
        "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], n_samples),
        "PaperlessBilling": np.random.choice(["Yes", "No"], n_samples),
        "PaymentMethod": np.random.choice(
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
            n_samples,
        ),
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Churn": np.random.choice(["Yes", "No"], n_samples, p=[0.27, 0.73]),
    }

    return pd.DataFrame(data)


@pytest.fixture(scope="session")
def sample_features(sample_data):
    """Extract features (X) from sample data."""
    return sample_data.drop("Churn", axis=1)


@pytest.fixture(scope="session")
def sample_target(sample_data):
    """Extract target (y) from sample data."""
    return (sample_data["Churn"] == "Yes").astype(int)


@pytest.fixture
def temp_dir():
    """
    Create a temporary directory for test artifacts.

    The directory is automatically cleaned up after the test.
    """
    tmp_dir = tempfile.mkdtemp()
    yield Path(tmp_dir)
    shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.fixture
def sample_config():
    """
    Create a sample TrainingConfig for testing.

    Uses small/fast settings suitable for testing.
    """
    from src.config import DataConfig, DriftConfig, MLflowConfig, ModelConfig, TrainingConfig

    return TrainingConfig(
        data=DataConfig(
            data_path="data/test_dataset.arff",
            test_size=0.2,
            val_size=0.25,
            random_state=42,
            stratify=True,
        ),
        model=ModelConfig(
            model_type="random_forest",  # Lowercase to match validator
            n_estimators_grid=[10],  # Small for fast tests
            max_depth_grid=[5],
            min_samples_split_grid=[2],
            min_samples_leaf_grid=[1],
            cv_folds=2,  # Fast
            scoring="roc_auc",
            n_jobs=1,  # Single core for deterministic tests
        ),
        mlflow=MLflowConfig(
            tracking_uri="./test_mlruns",
            experiment_name="test_experiment",
            log_models=False,  # Skip for speed
            log_artifacts=False,
        ),
        drift=DriftConfig(
            numeric_threshold=0.2,
            categorical_threshold=0.25,
            prediction_threshold=0.1,
            min_sample_size=10,  # Lower for tests
            max_drift_batch_size=100,
        ),
    )


@pytest.fixture(scope="session")
def trained_model(sample_features, sample_target):
    """
    Train a simple model for testing.

    This fixture creates a small RandomForest model that can be used
    to test prediction, probability calculation, and model behavior.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    # Define feature types
    numeric_features = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
    categorical_features = [col for col in sample_features.columns if col not in numeric_features]

    # Numeric pipeline
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )

    # Categorical pipeline
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Full pipeline
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42, n_jobs=1),
            ),
        ]
    )

    # Train model
    model.fit(sample_features, sample_target)
    return model


@pytest.fixture
def sample_model_metadata():
    """
    Create sample model metadata for testing.

    Mimics the structure saved by train.py.
    """
    return {
        "run_id": "20250101_120000",
        "timestamp": "2025-01-01T12:00:00",
        "model_version": "1.0.0",
        "best_params": {
            "classifier__n_estimators": 100,
            "classifier__max_depth": 10,
            "classifier__min_samples_split": 2,
            "classifier__min_samples_leaf": 1,
        },
        "metrics": {
            "roc_auc": 0.85,
            "accuracy": 0.80,
            "precision": 0.70,
            "recall": 0.65,
            "f1": 0.67,
        },
        "training_time_seconds": 45.2,
        "numeric_features": ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"],
        "categorical_features": [
            "gender",
            "Partner",
            "Dependents",
            "PhoneService",
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
            "Contract",
            "PaperlessBilling",
            "PaymentMethod",
        ],
        "reference_statistics": {
            "numeric": {
                "tenure": {"mean": 32.5, "std": 24.5, "min": 0, "max": 72},
                "MonthlyCharges": {"mean": 64.76, "std": 30.09, "min": 18.25, "max": 118.75},
            },
            "categorical": {
                "Contract": {"Month-to-month": 0.55, "One year": 0.21, "Two year": 0.24}
            },
            "target": {"positive_rate": 0.265, "n_samples": 7043},
        },
    }


@pytest.fixture
def sample_reference_stats():
    """Create sample reference statistics for drift testing."""
    return {
        "numeric": {
            "tenure": {"mean": 30.0, "std": 20.0, "min": 0, "max": 72},
            "MonthlyCharges": {"mean": 65.0, "std": 30.0, "min": 18, "max": 120},
        },
        "categorical": {
            "Contract": {"distribution": {"Month-to-month": 0.5, "One year": 0.3, "Two year": 0.2}},
            "InternetService": {"distribution": {"DSL": 0.3, "Fiber optic": 0.4, "No": 0.3}},
        },
        "target": {"positive_rate": 0.27, "n_samples": 5000},
    }


@pytest.fixture
def mock_model_file(trained_model, temp_dir):
    """
    Save a trained model to a temporary file.

    Returns the path to the saved model.
    """
    model_path = temp_dir / "test_model.joblib"
    joblib.dump(trained_model, model_path)
    return model_path


@pytest.fixture(scope="session")
def numeric_features():
    """List of numeric feature names."""
    return ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]


@pytest.fixture(scope="session")
def categorical_features():
    """List of categorical feature names."""
    return [
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
    ]


@pytest.fixture
def sample_prediction_request(sample_features):
    """
    Create a sample prediction request payload for single-record API.

    Returns a single customer's features as a dict matching PredictionRequest schema.
    Converts numpy types to native Python types for Pydantic validation.
    """
    # Get first customer as dictionary
    customer_dict = sample_features.iloc[0].to_dict()

    # Convert numpy types to native Python types (required for Pydantic validation)
    return convert_numpy_types(customer_dict)


# Cleanup fixtures for Docker/persistent resources
@pytest.fixture(autouse=True, scope="session")
def cleanup_test_artifacts():
    """Cleanup test artifacts after all tests complete."""
    yield
    # Cleanup test MLflow runs
    test_mlruns = Path("test_mlruns")
    if test_mlruns.exists():
        shutil.rmtree(test_mlruns, ignore_errors=True)


# =============================================================================
# Drift Detection Fixtures
# =============================================================================


@pytest.fixture
def stable_numeric_data():
    """Generate stable numeric data for drift testing (no drift)."""
    np.random.seed(42)
    return {
        "reference": {"mean": 50.0, "std": 10.0, "samples": np.random.normal(50, 10, 1000)},
        "current": pd.Series(np.random.normal(50, 10, 1000)),
    }


@pytest.fixture
def drifted_numeric_data_mean():
    """Generate drifted numeric data (mean shift)."""
    np.random.seed(42)
    return {
        "reference": {"mean": 50.0, "std": 10.0, "samples": np.random.normal(50, 10, 1000)},
        "current": pd.Series(np.random.normal(70, 10, 1000)),  # Shifted mean
    }


@pytest.fixture
def drifted_numeric_data_std():
    """Generate drifted numeric data (std shift)."""
    np.random.seed(42)
    return {
        "reference": {"mean": 50.0, "std": 10.0, "samples": np.random.normal(50, 10, 1000)},
        "current": pd.Series(np.random.normal(50, 20, 1000)),  # Doubled std
    }


@pytest.fixture
def stable_categorical_data():
    """Generate stable categorical data for drift testing (no drift)."""
    return {
        "reference": {"distribution": {"Month-to-month": 0.55, "One year": 0.24, "Two year": 0.21}},
        "current": pd.Series(["Month-to-month"] * 550 + ["One year"] * 240 + ["Two year"] * 210),
    }


@pytest.fixture
def drifted_categorical_data():
    """Generate drifted categorical data (proportion shift)."""
    return {
        "reference": {"distribution": {"Month-to-month": 0.55, "One year": 0.24, "Two year": 0.21}},
        "current": pd.Series(["Month-to-month"] * 200 + ["One year"] * 400 + ["Two year"] * 400),
    }


@pytest.fixture
def comprehensive_reference_stats():
    """Comprehensive reference statistics for end-to-end testing."""
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
        },
        "target": {"positive_rate": 0.27, "n_samples": 5000},
    }
