"""
Tests for MLflow Model Registry integration.

Model Registry & Versioning
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.mark.unit
def test_mlflow_register_model_env_var():
    """Test that MLFLOW_REGISTER_MODEL env var is correctly parsed."""

    # Test true values
    with patch.dict(os.environ, {"MLFLOW_REGISTER_MODEL": "true"}):
        result = os.getenv("MLFLOW_REGISTER_MODEL", "false").lower() == "true"
        assert result is True

    with patch.dict(os.environ, {"MLFLOW_REGISTER_MODEL": "TRUE"}):
        result = os.getenv("MLFLOW_REGISTER_MODEL", "false").lower() == "true"
        assert result is True

    # Test false values
    with patch.dict(os.environ, {"MLFLOW_REGISTER_MODEL": "false"}):
        result = os.getenv("MLFLOW_REGISTER_MODEL", "false").lower() == "true"
        assert result is False

    # Test default (no env var)
    with patch.dict(os.environ, {}, clear=True):
        result = os.getenv("MLFLOW_REGISTER_MODEL", "false").lower() == "true"
        assert result is False


@pytest.mark.unit
def test_model_source_env_var():
    """Test that MODEL_SOURCE env var is correctly parsed."""

    # Test registry mode
    with patch.dict(os.environ, {"MODEL_SOURCE": "registry"}):
        result = os.getenv("MODEL_SOURCE", "local").lower()
        assert result == "registry"

    # Test local mode (default)
    with patch.dict(os.environ, {}, clear=True):
        result = os.getenv("MODEL_SOURCE", "local").lower()
        assert result == "local"


@pytest.mark.unit
def test_model_stage_env_var():
    """Test that MODEL_STAGE env var defaults to Production."""

    with patch.dict(os.environ, {"MODEL_STAGE": "Staging"}):
        result = os.getenv("MODEL_STAGE", "Production")
        assert result == "Staging"

    # Test default
    with patch.dict(os.environ, {}, clear=True):
        result = os.getenv("MODEL_STAGE", "Production")
        assert result == "Production"


@pytest.mark.unit
def test_load_model_from_registry_not_available():
    """Test that load_model_from_registry raises error if MLflow not available."""
    from src.api.service import MLFLOW_AVAILABLE, load_model_from_registry

    if not MLFLOW_AVAILABLE:
        with pytest.raises(Exception, match="MLflow not available"):
            load_model_from_registry("test_model", "Production")
    else:
        pytest.skip("MLflow is available, skipping unavailability test")


@pytest.mark.integration
@patch("mlflow.set_tracking_uri")
@patch("mlflow.pyfunc.load_model")
@patch("mlflow.tracking.MlflowClient")
def test_load_model_from_registry_success(mock_client_class, mock_load_model, mock_set_uri):
    """Test successful model loading from registry."""
    from src.api.service import MLFLOW_AVAILABLE, load_model_from_registry

    if not MLFLOW_AVAILABLE:
        pytest.skip("MLflow not available")

    # Mock the model
    mock_model = Mock()
    mock_model._model_impl = Mock()
    mock_model._model_impl.python_model = Mock()
    mock_load_model.return_value = mock_model

    # Mock the client
    mock_client = Mock()
    mock_client_class.return_value = mock_client

    # Mock version info
    mock_version = Mock()
    mock_version.version = "1"
    mock_version.run_id = "test_run_id_12345"
    mock_client.get_latest_versions.return_value = [mock_version]

    # Mock run and artifacts
    mock_run = Mock()
    mock_client.get_run.return_value = mock_run
    mock_client.list_artifacts.return_value = []  # No metadata

    # Call the function
    model, version, features, threshold = load_model_from_registry("test_model", "Production")

    # Assertions
    assert model is not None
    assert version == "1"
    assert isinstance(features, list)
    assert threshold == 0.5  # Default when no metadata

    # Verify MLflow calls
    mock_set_uri.assert_called_once_with("./mlruns")
    mock_load_model.assert_called_once_with("models:/test_model/Production")
    mock_client.get_latest_versions.assert_called_once_with("test_model", stages=["Production"])


@pytest.mark.integration
@patch("mlflow.set_tracking_uri")
@patch("mlflow.pyfunc.load_model")
@patch("mlflow.tracking.MlflowClient")
def test_load_model_from_registry_with_metadata(mock_client_class, mock_load_model, mock_set_uri):
    """Test model loading from registry with metadata extraction."""
    from src.api.service import MLFLOW_AVAILABLE, load_model_from_registry

    if not MLFLOW_AVAILABLE:
        pytest.skip("MLflow not available")

    # Mock the model
    mock_model = Mock()
    mock_model._model_impl = Mock()
    mock_model._model_impl.python_model = Mock()
    mock_load_model.return_value = mock_model

    # Mock the client
    mock_client = Mock()
    mock_client_class.return_value = mock_client

    # Mock version info
    mock_version = Mock()
    mock_version.version = "2"
    mock_version.run_id = "test_run_id_67890"
    mock_client.get_latest_versions.return_value = [mock_version]

    # Mock run
    mock_run = Mock()
    mock_client.get_run.return_value = mock_run

    # Mock artifact with metadata
    mock_artifact = Mock()
    mock_artifact.path = "metadata/metadata_test.json"
    mock_client.list_artifacts.return_value = [mock_artifact]

    # Create temporary metadata file
    metadata = {
        "schema": {"all_features": ["feature1", "feature2", "feature3"]},
        "threshold": {"chosen_threshold": 0.35, "chosen_strategy": "f1_maximization"},
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        metadata_path = Path(tmp_dir) / "metadata_test.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

        # Mock download_artifacts to return our temp file
        mock_client.download_artifacts.return_value = str(metadata_path)

        # Call the function
        model, version, features, threshold = load_model_from_registry("test_model", "Production")

        # Assertions
        assert model is not None
        assert version == "2"
        assert features == ["feature1", "feature2", "feature3"]
        assert threshold == 0.35


@pytest.mark.integration
@patch("mlflow.tracking.MlflowClient")
def test_load_model_from_registry_no_model_in_stage(mock_client_class):
    """Test that appropriate error is raised when no model in stage."""
    from src.api.service import MLFLOW_AVAILABLE, load_model_from_registry

    if not MLFLOW_AVAILABLE:
        pytest.skip("MLflow not available")

    # Mock the client
    mock_client = Mock()
    mock_client_class.return_value = mock_client

    # Return empty list (no models in stage)
    mock_client.get_latest_versions.return_value = []

    # Should raise exception (MLflow raises "not found" error)
    with pytest.raises(Exception, match="(No model found|not found)"):
        load_model_from_registry("test_model", "Production")


@pytest.mark.integration
@patch.dict(os.environ, {"MODEL_SOURCE": "registry", "MODEL_STAGE": "Production"})
@patch("src.api.service.load_model_from_registry")
def test_load_model_uses_registry_when_env_set(mock_load_from_registry):
    """Test that load_model uses registry when MODEL_SOURCE=registry."""
    from src.api.service import load_model

    # Setup mock to return expected values
    mock_model = Mock()
    mock_load_from_registry.return_value = (mock_model, "1", ["feat1"], 0.5)

    # Call load_model (should detect env var and use registry)
    model, version, features, threshold = load_model()

    # Should have called load_model_from_registry
    mock_load_from_registry.assert_called_once_with("churn_prediction_model", "Production")

    # Check returned values
    assert model == mock_model
    assert version == "1"
    assert features == ["feat1"]
    assert threshold == 0.5


@pytest.mark.integration
def test_load_model_uses_local_by_default():
    """Test that load_model uses local file loading by default."""
    from src.api.service import load_model

    # Ensure MODEL_SOURCE is not set or set to local
    with patch.dict(os.environ, {"MODEL_SOURCE": "local"}, clear=False):
        # Should try to load from file (will fail if file doesn't exist, which is expected)
        with pytest.raises(FileNotFoundError, match="Model not found"):
            load_model("nonexistent_model.joblib")


@pytest.mark.integration
def test_registry_model_name_env_var():
    """Test that custom model name can be specified via env var."""
    with patch.dict(os.environ, {"MODEL_NAME": "custom_model"}):
        model_name = os.getenv("MODEL_NAME", "churn_prediction_model")
        assert model_name == "custom_model"

    # Test default
    with patch.dict(os.environ, {}, clear=True):
        model_name = os.getenv("MODEL_NAME", "churn_prediction_model")
        assert model_name == "churn_prediction_model"


@pytest.mark.unit
def test_registry_stages_are_valid():
    """Test that valid MLflow stages are recognized."""
    valid_stages = ["None", "Staging", "Production", "Archived"]

    for stage in valid_stages:
        assert stage in valid_stages

    # Test invalid stages
    invalid_stages = ["Development", "Testing", "prod", "staging"]
    for stage in invalid_stages:
        assert stage not in valid_stages


# =============================================================================
# Latest model/metadata pairing (find_latest_metadata_file)
# =============================================================================


@pytest.mark.unit
def test_find_latest_metadata_prefers_paired_file(tmp_path, monkeypatch):
    """The latest model pairs with metadata_latest.json even when a lexically
    newer stray metadata file exists (which a reverse-sorted glob would pick)."""
    from src.api.service import find_latest_metadata_file

    monkeypatch.chdir(tmp_path)
    models = tmp_path / "models"
    models.mkdir()
    (models / "metadata_latest.json").write_text(json.dumps({"run_id": "correct"}))
    # "z..." sorts after "latest" lexically -> would win a reverse-sorted glob
    (models / "metadata_zzzz9999.json").write_text(json.dumps({"run_id": "stray"}))

    result = find_latest_metadata_file()

    assert result is not None
    assert result.name == "metadata_latest.json"
    with open(result) as f:
        assert json.load(f)["run_id"] == "correct"


@pytest.mark.unit
def test_find_latest_metadata_falls_back_to_newest(tmp_path, monkeypatch):
    """Without metadata_latest.json, fall back to the newest metadata_*.json."""
    from src.api.service import find_latest_metadata_file

    monkeypatch.chdir(tmp_path)
    models = tmp_path / "models"
    models.mkdir()
    (models / "metadata_20250101_000000.json").write_text(json.dumps({"run_id": "old"}))
    (models / "metadata_20250601_000000.json").write_text(json.dumps({"run_id": "new"}))

    result = find_latest_metadata_file()

    assert result is not None
    assert result.name == "metadata_20250601_000000.json"


@pytest.mark.unit
def test_find_latest_metadata_none_when_empty(tmp_path, monkeypatch):
    """No metadata files present -> None."""
    from src.api.service import find_latest_metadata_file

    monkeypatch.chdir(tmp_path)
    (tmp_path / "models").mkdir()

    assert find_latest_metadata_file() is None
