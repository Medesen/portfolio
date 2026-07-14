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
@patch("mlflow.sklearn.load_model")
@patch("mlflow.tracking.MlflowClient")
def test_load_model_from_registry_success(mock_client_class, mock_load_model, mock_set_uri):
    """Test successful model loading from registry via the sklearn flavor."""
    from src.api.service import MLFLOW_AVAILABLE, load_model_from_registry

    if not MLFLOW_AVAILABLE:
        pytest.skip("MLflow not available")

    # The sklearn flavor returns the actual pipeline (predict_proba intact) —
    # NOT a pyfunc wrapper whose only interface is predict().
    mock_pipeline = Mock(spec=["predict", "predict_proba"])
    mock_load_model.return_value = mock_pipeline

    # Mock the client
    mock_client = Mock()
    mock_client_class.return_value = mock_client

    # Two Production versions plus one in another stage: the loader must
    # filter by stage and pick the numerically newest version.
    v1 = Mock(version="1", run_id="run_v1", current_stage="Production")
    v3 = Mock(version="3", run_id="test_run_id_12345", current_stage="Production")
    v4 = Mock(version="4", run_id="run_v4", current_stage="Staging")
    mock_client.search_model_versions.return_value = [v1, v3, v4]

    # Mock run and artifacts
    mock_run = Mock()
    mock_client.get_run.return_value = mock_run
    mock_client.list_artifacts.return_value = []  # No metadata

    # Call the function (with the tracking-URI env var unset -> ./mlruns default)
    with patch.dict(os.environ):
        os.environ.pop("MLFLOW_TRACKING_URI", None)
        model, version, features, threshold, metadata = load_model_from_registry(
            "test_model", "Production"
        )

    # The loader must hand back the sklearn pipeline itself
    assert model is mock_pipeline
    assert version == "3"
    assert isinstance(features, list)
    assert threshold == 0.5  # Default when no metadata
    assert metadata is None  # No metadata artifacts in the run

    # Verify MLflow calls: env-fallback URI, stage filtering, version-pinned URI
    mock_set_uri.assert_called_once_with("./mlruns")
    mock_load_model.assert_called_once_with("models:/test_model/3")
    mock_client.search_model_versions.assert_called_once_with("name='test_model'")


@pytest.mark.integration
@patch("mlflow.set_tracking_uri")
@patch("mlflow.sklearn.load_model")
@patch("mlflow.tracking.MlflowClient")
def test_load_model_from_registry_honors_tracking_uri_env(
    mock_client_class, mock_load_model, mock_set_uri
):
    """MLFLOW_TRACKING_URI must override the ./mlruns default."""
    from src.api.service import MLFLOW_AVAILABLE, load_model_from_registry

    if not MLFLOW_AVAILABLE:
        pytest.skip("MLflow not available")

    mock_load_model.return_value = Mock(spec=["predict", "predict_proba"])
    mock_client = Mock()
    mock_client_class.return_value = mock_client
    mock_client.search_model_versions.return_value = [
        Mock(version="1", run_id="run_v1", current_stage="Production")
    ]
    mock_client.list_artifacts.return_value = []

    with patch.dict(os.environ, {"MLFLOW_TRACKING_URI": "http://mlflow.internal:5000"}):
        load_model_from_registry("test_model", "Production")

    mock_set_uri.assert_called_once_with("http://mlflow.internal:5000")


@pytest.mark.integration
@patch("mlflow.set_tracking_uri")
@patch("mlflow.sklearn.load_model")
@patch("mlflow.tracking.MlflowClient")
def test_load_model_from_registry_with_metadata(mock_client_class, mock_load_model, mock_set_uri):
    """Test model loading from registry with metadata extraction."""
    from src.api.service import MLFLOW_AVAILABLE, load_model_from_registry

    if not MLFLOW_AVAILABLE:
        pytest.skip("MLflow not available")

    # The sklearn flavor returns the pipeline directly
    mock_pipeline = Mock(spec=["predict", "predict_proba"])
    mock_load_model.return_value = mock_pipeline

    # Mock the client
    mock_client = Mock()
    mock_client_class.return_value = mock_client

    # Mock version info
    mock_version = Mock(version="2", run_id="test_run_id_67890", current_stage="Production")
    mock_client.search_model_versions.return_value = [mock_version]

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
        model, version, features, threshold, loaded_metadata = load_model_from_registry(
            "test_model", "Production"
        )

        # Assertions
        assert model is not None
        assert version == "2"
        assert features == ["feature1", "feature2", "feature3"]
        assert threshold == 0.35
        assert loaded_metadata == metadata  # full metadata dict returned with the model


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

    # Versions exist, but none in the requested stage
    mock_client.search_model_versions.return_value = [
        Mock(version="1", run_id="run_v1", current_stage="Staging")
    ]

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
    mock_load_from_registry.return_value = (mock_model, "1", ["feat1"], 0.5, None)

    # Call load_model (should detect env var and use registry)
    model, version, features, threshold, metadata = load_model()

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


@pytest.mark.integration
def test_cached_metadata_stays_paired_with_cached_model(tmp_path, monkeypatch):
    """The cache must keep serving the metadata loaded WITH the model, even after
    a retrain overwrites the metadata files on disk.

    Regression test: the drift endpoint previously re-read the latest metadata
    from disk per request, so after a retrain (which does not swap the cached
    model until restart) it could score the old cached model against the new
    model's baseline and threshold.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline as SkPipeline
    from sklearn.preprocessing import StandardScaler

    from src.api.service import get_metadata_from_cache, get_model_from_cache
    from src.utils.io import save_model

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MODEL_SOURCE", raising=False)
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    # Real model + checksum sidecar so load_model's fail-closed check passes
    pipeline = SkPipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(n_estimators=2, random_state=42)),
        ]
    )
    model_path = models_dir / "churn_model_latest.joblib"
    save_model(pipeline, str(model_path))
    monkeypatch.setenv("MODEL_PATH", str(model_path))

    original = {"run_id": "original", "reference_statistics": {"marker": "original"}}
    (models_dir / "metadata_latest.json").write_text(json.dumps(original))

    cache = {"model": None, "version": None, "features": None, "threshold": 0.5, "metadata": None}
    get_model_from_cache(cache)
    assert cache["version"] == "original"

    # Simulate a retrain overwriting the on-disk metadata while the old model
    # keeps serving from the cache
    retrained = {"run_id": "retrained", "reference_statistics": {"marker": "retrained"}}
    (models_dir / "metadata_latest.json").write_text(json.dumps(retrained))

    metadata = get_metadata_from_cache(cache)
    assert metadata is not None
    assert metadata["run_id"] == "original"  # NOT the retrained model's metadata
    assert metadata["reference_statistics"]["marker"] == "original"
