"""
End-to-end tests for complete workflows.

Tests training scripts, config loading, and artifact generation.

Comprehensive Testing
"""

import json
import os
import subprocess
from pathlib import Path

import pytest

# =============================================================================
# Training Script Tests
# =============================================================================


@pytest.mark.e2e
@pytest.mark.slow
def test_train_script_completes_with_quick_config():
    """Test training script runs successfully with quick config."""
    result = subprocess.run(
        ["python", "train.py", "--config", "config/train_config_quick.yaml"],
        capture_output=True,
        text=True,
        timeout=300,  # 5 minutes timeout
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )

    # Check exit code
    assert result.returncode == 0, f"Training failed with: {result.stderr}"

    # Check output contains success messages
    assert "TRAINING COMPLETE" in result.stdout
    assert "Grid search complete" in result.stdout


@pytest.mark.e2e
@pytest.mark.slow
def test_train_script_creates_model_artifacts():
    """Test training script creates all expected artifacts."""
    # Run training
    result = subprocess.run(
        ["python", "train.py", "--config", "config/train_config_quick.yaml"],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )

    assert result.returncode == 0

    # Check artifacts exist
    project_root = Path(__file__).parent.parent

    # Model should exist
    latest_model = project_root / "models" / "churn_model_latest.joblib"
    assert latest_model.exists(), "Latest model not created"

    # At least one versioned model should exist
    models_dir = project_root / "models"
    versioned_models = list(models_dir.glob("churn_model_*.joblib"))
    assert len(versioned_models) > 0, "No versioned models created"


@pytest.mark.e2e
def test_train_script_creates_metadata():
    """Test training script creates metadata file."""
    # Run quick training
    result = subprocess.run(
        ["python", "train.py", "--config", "config/train_config_quick.yaml"],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )

    assert result.returncode == 0

    # Check metadata exists
    project_root = Path(__file__).parent.parent
    metadata_files = list((project_root / "models").glob("metadata_*.json"))

    assert len(metadata_files) > 0, "No metadata files created"

    # Check metadata structure
    latest_metadata = sorted(metadata_files, key=lambda p: p.stat().st_mtime)[-1]
    with open(latest_metadata, "r") as f:
        metadata = json.load(f)

    # Check required fields (match actual metadata structure)
    assert "run_id" in metadata
    assert "timestamp" in metadata
    assert "hyperparameters" in metadata  # Changed from 'best_params'
    assert "validation_metrics" in metadata  # Changed from 'metrics'
    assert "reference_statistics" in metadata


@pytest.mark.e2e
def test_train_script_creates_diagnostics():
    """Test training script creates diagnostic visualizations."""
    result = subprocess.run(
        ["python", "train.py", "--config", "config/train_config_quick.yaml"],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )

    assert result.returncode == 0

    # Check diagnostic files exist
    project_root = Path(__file__).parent.parent
    diagnostics_dir = project_root / "diagnostics"

    # Should have confusion matrix, ROC curve, PR curve, feature importance plots
    png_files = list(diagnostics_dir.glob("*.png"))
    assert len(png_files) >= 4, f"Expected at least 4 diagnostic plots, found {len(png_files)}"


# =============================================================================
# Configuration Tests
# =============================================================================


@pytest.mark.e2e
def test_config_saved_is_loadable():
    """Test saved config can be reloaded and used."""
    # Run training (creates config file)
    result = subprocess.run(
        ["python", "train.py", "--config", "config/train_config_quick.yaml"],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )

    assert result.returncode == 0

    # Find saved config
    project_root = Path(__file__).parent.parent
    config_files = list((project_root / "configs").glob("run_config_*.yaml"))

    assert len(config_files) > 0, "No config files saved"

    # Try to load the config
    from src.config import TrainingConfig

    latest_config_file = sorted(config_files, key=lambda p: p.stat().st_mtime)[-1]
    config = TrainingConfig.from_yaml(str(latest_config_file))

    # Should load without error
    assert config is not None
    assert config.model.cv_folds > 0


@pytest.mark.e2e
def test_train_with_default_config():
    """Test training works with default config (no --config arg)."""
    # subprocess.run(timeout=...) raises TimeoutExpired on timeout (it never
    # returns a -9/124 code). A timeout is a real failure — a hung or severely
    # regressed training run must turn CI red, not be skipped over. Environments
    # that cannot afford this test should deselect it (-m "not e2e") explicitly.
    try:
        result = subprocess.run(
            ["python", "train.py"],
            capture_output=True,
            text=True,
            timeout=600,  # Longer timeout for default config (more hyperparameters)
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
    except subprocess.TimeoutExpired:
        pytest.fail(
            "Default-config training exceeded the 600s timeout — the training "
            "workflow hung or regressed severely (it normally finishes well within "
            "the limit)."
        )

    # Should complete successfully
    assert result.returncode == 0, f"Training failed unexpectedly: {result.stderr[:500]}"


# =============================================================================
# MLflow Integration Tests
# =============================================================================


@pytest.mark.e2e
def test_train_script_logs_to_mlflow():
    """Test training script creates MLflow run."""
    # Clear test mlruns if it exists
    project_root = Path(__file__).parent.parent
    test_mlruns = project_root / "mlruns"

    # Run training
    result = subprocess.run(
        ["python", "train.py", "--config", "config/train_config_quick.yaml"],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(project_root),
    )

    assert result.returncode == 0

    # Check MLflow directory exists
    assert test_mlruns.exists(), "MLflow tracking directory not created"

    # Check experiment directory exists
    experiment_dirs = [d for d in test_mlruns.iterdir() if d.is_dir() and d.name.isdigit()]
    assert len(experiment_dirs) > 0, "No MLflow experiments created"


# =============================================================================
# Error Handling Tests
# =============================================================================


@pytest.mark.e2e
def test_train_script_handles_invalid_config():
    """Test training script fails gracefully with invalid config."""
    # Create invalid config file
    project_root = Path(__file__).parent.parent
    invalid_config = project_root / "config" / "invalid_test.yaml"

    try:
        with open(invalid_config, "w") as f:
            f.write("data:\n  test_size: 1.5\n")  # Invalid (> 1)

        result = subprocess.run(
            ["python", "train.py", "--config", str(invalid_config)],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(project_root),
        )

        # Should fail with non-zero exit code
        assert result.returncode != 0

        # Should have validation error in stderr
        assert "ValidationError" in result.stderr or "validation" in result.stderr.lower()

    finally:
        # Clean up invalid config
        if invalid_config.exists():
            invalid_config.unlink()


@pytest.mark.e2e
def test_train_script_handles_missing_data():
    """Test training script fails gracefully when data file missing."""
    # Create config with non-existent data path
    project_root = Path(__file__).parent.parent
    bad_config = project_root / "config" / "bad_data_test.yaml"

    try:
        with open(bad_config, "w") as f:
            f.write("""
data:
  data_path: "data/nonexistent_file.arff"
model:
  n_estimators_grid: [10]
  max_depth_grid: [5]
  cv_folds: 2
""")

        result = subprocess.run(
            ["python", "train.py", "--config", str(bad_config)],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(project_root),
        )

        # Should fail with non-zero exit code
        assert result.returncode != 0

        # Should mention file not found
        error_output = result.stderr + result.stdout
        assert "FileNotFoundError" in error_output or "not found" in error_output.lower()

    finally:
        # Clean up test config
        if bad_config.exists():
            bad_config.unlink()
