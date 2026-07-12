"""
Unit tests for configuration management (Pydantic configs).

Tests configuration validation, loading, saving, and defaults.
"""

import pytest
from pydantic import ValidationError

from src.config import (
    DataConfig,
    DriftConfig,
    MLflowConfig,
    ModelConfig,
    ServiceConfig,
    TrainingConfig,
)

# =============================================================================
# DataConfig Tests
# =============================================================================


@pytest.mark.unit
def test_data_config_defaults():
    """Test DataConfig uses correct default values."""
    config = DataConfig()

    assert config.data_path == "data/dataset.arff"
    assert config.random_state == 42
    assert config.test_size == 0.2
    assert config.val_size == 0.25
    assert config.stratify is True


@pytest.mark.unit
def test_data_config_custom_values():
    """Test DataConfig accepts custom values."""
    config = DataConfig(
        data_path="custom/path.arff", random_state=123, test_size=0.3, val_size=0.2, stratify=False
    )

    assert config.data_path == "custom/path.arff"
    assert config.random_state == 123
    assert config.test_size == 0.3
    assert config.val_size == 0.2
    assert config.stratify is False


@pytest.mark.unit
def test_data_config_validates_test_size_range():
    """Test DataConfig rejects invalid test_size values."""
    # Valid values
    DataConfig(test_size=0.1)  # Should not raise
    DataConfig(test_size=0.5)  # Should not raise
    DataConfig(test_size=0.9)  # Should not raise

    # Invalid: too large
    with pytest.raises(ValidationError) as exc_info:
        DataConfig(test_size=1.5)
    assert "test_size" in str(exc_info.value).lower()

    # Invalid: negative
    with pytest.raises(ValidationError) as exc_info:
        DataConfig(test_size=-0.1)
    assert "test_size" in str(exc_info.value).lower()

    # Invalid: exactly 0
    with pytest.raises(ValidationError):
        DataConfig(test_size=0.0)

    # Invalid: exactly 1
    with pytest.raises(ValidationError):
        DataConfig(test_size=1.0)


@pytest.mark.unit
def test_data_config_validates_val_size_range():
    """Test DataConfig rejects invalid val_size values."""
    # Valid
    DataConfig(val_size=0.25)  # Should not raise

    # Invalid
    with pytest.raises(ValidationError):
        DataConfig(val_size=1.5)

    with pytest.raises(ValidationError):
        DataConfig(val_size=-0.1)


# =============================================================================
# ModelConfig Tests
# =============================================================================


@pytest.mark.unit
def test_model_config_defaults():
    """Test ModelConfig uses correct defaults."""
    config = ModelConfig()

    assert config.model_type == "random_forest"  # Lowercase to match validator
    assert config.n_estimators_grid == [100, 200]
    assert config.max_depth_grid == [10, 20, None]
    assert config.cv_folds == 5
    assert config.scoring == "roc_auc"
    assert config.n_jobs == -1


@pytest.mark.unit
def test_model_config_to_param_grid():
    """Test ModelConfig converts correctly to sklearn param_grid."""
    config = ModelConfig(
        n_estimators_grid=[100, 200, 300],
        max_depth_grid=[10, None],
        min_samples_split_grid=[2, 5],
        min_samples_leaf_grid=[1, 2, 4],
    )

    param_grid = config.to_param_grid()

    assert isinstance(param_grid, dict)
    assert param_grid["classifier__n_estimators"] == [100, 200, 300]
    assert param_grid["classifier__max_depth"] == [10, None]
    assert param_grid["classifier__min_samples_split"] == [2, 5]
    assert param_grid["classifier__min_samples_leaf"] == [1, 2, 4]


@pytest.mark.unit
def test_model_config_validates_positive_values():
    """Test ModelConfig rejects non-positive grid values."""
    # Valid
    ModelConfig(n_estimators_grid=[10, 20])  # Should not raise

    # Invalid: zero
    with pytest.raises(ValidationError):
        ModelConfig(n_estimators_grid=[0, 10])

    # Invalid: negative
    with pytest.raises(ValidationError):
        ModelConfig(n_estimators_grid=[-5, 10])


@pytest.mark.unit
def test_model_config_validates_cv_folds():
    """Test ModelConfig requires cv_folds >= 2."""
    # Valid
    ModelConfig(cv_folds=2)  # Should not raise
    ModelConfig(cv_folds=10)  # Should not raise

    # Invalid
    with pytest.raises(ValidationError):
        ModelConfig(cv_folds=1)

    with pytest.raises(ValidationError):
        ModelConfig(cv_folds=0)


@pytest.mark.unit
def test_model_config_validates_max_depth():
    """Test ModelConfig allows None or positive max_depth."""
    # Valid
    ModelConfig(max_depth_grid=[None])  # Should not raise
    ModelConfig(max_depth_grid=[10, 20])  # Should not raise
    ModelConfig(max_depth_grid=[5, None])  # Should not raise

    # Invalid: zero
    with pytest.raises(ValidationError):
        ModelConfig(max_depth_grid=[0])

    # Invalid: negative
    with pytest.raises(ValidationError):
        ModelConfig(max_depth_grid=[-5])


# =============================================================================
# ServiceConfig Tests
# =============================================================================


@pytest.mark.unit
def test_service_config_defaults():
    """Test ServiceConfig uses correct defaults."""
    config = ServiceConfig()

    assert config.host == "0.0.0.0"
    assert config.port == 8000
    assert config.log_level == "INFO"
    assert config.max_batch_size == 1000
    assert config.max_request_size_mb == 10
    assert config.service_token is None


@pytest.mark.unit
def test_service_config_validates_log_level():
    """Test ServiceConfig only accepts valid log levels."""
    # Valid log levels
    for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
        config = ServiceConfig(log_level=level)
        assert config.log_level == level

    # Case insensitive
    config = ServiceConfig(log_level="info")
    assert config.log_level == "INFO"

    # Invalid log level
    with pytest.raises(ValidationError) as exc_info:
        ServiceConfig(log_level="INVALID")
    assert "log level" in str(exc_info.value).lower()


@pytest.mark.unit
def test_service_config_validates_port_range():
    """Test ServiceConfig requires port in valid range."""
    # Valid ports
    ServiceConfig(port=80)
    ServiceConfig(port=8000)
    ServiceConfig(port=65535)

    # Invalid: too small
    with pytest.raises(ValidationError):
        ServiceConfig(port=0)

    # Invalid: too large
    with pytest.raises(ValidationError):
        ServiceConfig(port=99999)


# =============================================================================
# DriftConfig Tests
# =============================================================================


@pytest.mark.unit
def test_drift_config_defaults():
    """Test DriftConfig uses correct defaults."""
    config = DriftConfig()

    assert config.numeric_threshold == 0.2
    assert config.categorical_threshold == 0.25
    assert config.prediction_threshold == 0.1
    assert config.min_sample_size == 100
    assert config.max_drift_batch_size == 1000


@pytest.mark.unit
def test_drift_config_validates_threshold_ranges():
    """Test DriftConfig requires thresholds between 0 and 1."""
    # Valid
    DriftConfig(numeric_threshold=0.0)  # Should not raise
    DriftConfig(numeric_threshold=0.5)
    DriftConfig(numeric_threshold=1.0)

    # Invalid: negative
    with pytest.raises(ValidationError):
        DriftConfig(numeric_threshold=-0.1)

    # Invalid: too large
    with pytest.raises(ValidationError):
        DriftConfig(numeric_threshold=1.5)


# =============================================================================
# TrainingConfig Tests
# =============================================================================


@pytest.mark.unit
def test_training_config_defaults():
    """Test TrainingConfig uses nested defaults correctly."""
    config = TrainingConfig()

    # Check nested configs exist
    assert isinstance(config.data, DataConfig)
    assert isinstance(config.model, ModelConfig)
    assert isinstance(config.mlflow, MLflowConfig)
    assert isinstance(config.drift, DriftConfig)

    # Spot check some defaults
    assert config.data.test_size == 0.2
    assert config.model.cv_folds == 5
    assert config.mlflow.tracking_uri == "./mlruns"


@pytest.mark.unit
def test_training_config_save_load_yaml(temp_dir):
    """Test saving and loading TrainingConfig from YAML."""
    # Create config with custom values
    config = TrainingConfig(
        data=DataConfig(test_size=0.3, val_size=0.2), model=ModelConfig(cv_folds=3, n_jobs=2)
    )

    # Save to YAML
    config_path = temp_dir / "test_config.yaml"
    config.save(str(config_path))

    assert config_path.exists()

    # Load and verify
    loaded_config = TrainingConfig.from_yaml(str(config_path))

    assert loaded_config.data.test_size == 0.3
    assert loaded_config.data.val_size == 0.2
    assert loaded_config.model.cv_folds == 3
    assert loaded_config.model.n_jobs == 2


@pytest.mark.unit
def test_training_config_save_load_json(temp_dir):
    """Test saving and loading TrainingConfig from JSON."""
    config = TrainingConfig()

    # Save to JSON
    config_path = temp_dir / "test_config.json"
    config.save(str(config_path))

    assert config_path.exists()

    # Load and verify
    loaded_config = TrainingConfig.from_json(str(config_path))

    assert loaded_config.data.test_size == config.data.test_size
    assert loaded_config.model.cv_folds == config.model.cv_folds


@pytest.mark.unit
def test_training_config_save_invalid_extension(temp_dir):
    """Test TrainingConfig.save rejects invalid file extensions."""
    config = TrainingConfig()

    invalid_path = temp_dir / "test_config.txt"

    with pytest.raises(ValueError) as exc_info:
        config.save(str(invalid_path))

    assert "extension" in str(exc_info.value).lower()


@pytest.mark.unit
def test_training_config_to_dict():
    """Test TrainingConfig converts to dictionary correctly."""
    config = TrainingConfig()

    config_dict = config.to_dict()

    assert isinstance(config_dict, dict)
    assert "data" in config_dict
    assert "model" in config_dict
    assert "mlflow" in config_dict
    assert "drift" in config_dict

    # Check nested structure
    assert isinstance(config_dict["data"], dict)
    assert "test_size" in config_dict["data"]


# =============================================================================
# MLflowConfig Tests
# =============================================================================


@pytest.mark.unit
def test_mlflow_config_defaults():
    """Test MLflowConfig uses correct defaults."""
    config = MLflowConfig()

    assert config.tracking_uri == "./mlruns"
    assert config.experiment_name == "churn_prediction"
    assert config.log_models is True
    assert config.log_artifacts is True


@pytest.mark.unit
def test_mlflow_config_custom_values():
    """Test MLflowConfig accepts custom values."""
    config = MLflowConfig(
        tracking_uri="http://mlflow-server:5000",
        experiment_name="custom_experiment",
        log_models=False,
        log_artifacts=False,
    )

    assert config.tracking_uri == "http://mlflow-server:5000"
    assert config.experiment_name == "custom_experiment"
    assert config.log_models is False
    assert config.log_artifacts is False
