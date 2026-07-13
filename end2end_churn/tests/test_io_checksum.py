"""
Tests for I/O utilities and model checksum validation.

Tests model saving/loading with integrity verification.

Model Checksum Validation Tests
"""

import json
import os
import tempfile
from pathlib import Path

import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils.io import compute_file_checksum, save_metadata, save_model, verify_model_checksum


@pytest.mark.unit
class TestChecksumComputation:
    """Test checksum computation functionality."""

    def test_compute_checksum_sha256(self):
        """Test SHA256 checksum computation."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("test content")
            temp_path = f.name

        try:
            checksum = compute_file_checksum(temp_path, algorithm="sha256")

            # Should return 64-character hex string (SHA256 = 256 bits = 32 bytes = 64 hex chars)
            assert isinstance(checksum, str)
            assert len(checksum) == 64
            assert all(c in "0123456789abcdef" for c in checksum)
        finally:
            os.unlink(temp_path)

    def test_compute_checksum_deterministic(self):
        """Checksum should be deterministic (same content = same checksum)."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("deterministic test")
            temp_path = f.name

        try:
            checksum1 = compute_file_checksum(temp_path)
            checksum2 = compute_file_checksum(temp_path)

            assert checksum1 == checksum2
        finally:
            os.unlink(temp_path)

    def test_compute_checksum_different_content(self):
        """Different content should produce different checksums."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f1:
            f1.write("content 1")
            path1 = f1.name

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f2:
            f2.write("content 2")
            path2 = f2.name

        try:
            checksum1 = compute_file_checksum(path1)
            checksum2 = compute_file_checksum(path2)

            assert checksum1 != checksum2
        finally:
            os.unlink(path1)
            os.unlink(path2)

    def test_compute_checksum_file_not_found(self):
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            compute_file_checksum("/nonexistent/file.txt")


@pytest.mark.unit
class TestModelSaveWithChecksum:
    """Test model saving with checksum generation."""

    def test_save_model_creates_checksum_file(self):
        """Saving model should create .sha256 checksum file."""
        # Create a simple pipeline
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", RandomForestClassifier(n_estimators=10, random_state=42)),
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "test_model.joblib")
            checksum_path = f"{model_path}.sha256"

            # Save model
            checksum = save_model(pipeline, model_path)

            # Verify files created
            assert os.path.exists(model_path), "Model file should exist"
            assert os.path.exists(checksum_path), "Checksum file should exist"

            # Verify checksum returned
            assert isinstance(checksum, str)
            assert len(checksum) == 64  # SHA256 hex length

    def test_save_model_checksum_file_content(self):
        """Checksum file should contain valid JSON with required fields."""
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", RandomForestClassifier(n_estimators=10, random_state=42)),
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "test_model.joblib")
            checksum_path = f"{model_path}.sha256"

            save_model(pipeline, model_path)

            # Load checksum file
            with open(checksum_path, "r") as f:
                checksum_data = json.load(f)

            # Verify structure
            assert "checksum" in checksum_data
            assert "algorithm" in checksum_data
            assert "file" in checksum_data
            assert "size_bytes" in checksum_data
            assert "created_at" in checksum_data

            assert checksum_data["algorithm"] == "sha256"
            assert checksum_data["file"] == "test_model.joblib"
            assert checksum_data["size_bytes"] > 0

    def test_save_model_returns_correct_checksum(self):
        """Returned checksum should match file content."""
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", RandomForestClassifier(n_estimators=10, random_state=42)),
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "test_model.joblib")

            # Save model and get checksum
            returned_checksum = save_model(pipeline, model_path)

            # Compute checksum manually
            manual_checksum = compute_file_checksum(model_path)

            assert returned_checksum == manual_checksum


@pytest.mark.unit
class TestModelChecksumVerification:
    """Test model checksum verification."""

    def test_verify_checksum_valid_model(self):
        """Verification should pass for valid model with matching checksum."""
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", RandomForestClassifier(n_estimators=10, random_state=42)),
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "test_model.joblib")

            # Save model (creates checksum)
            save_model(pipeline, model_path)

            # Verify checksum
            result = verify_model_checksum(model_path)

            assert result is True

    def test_verify_checksum_missing_checksum_file(self):
        """Verification should return False (with warning) if checksum file missing."""
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", RandomForestClassifier(n_estimators=10, random_state=42)),
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "test_model.joblib")

            # Save model manually (without checksum)
            import joblib

            joblib.dump(pipeline, model_path)

            # Verify should return False (checksum file doesn't exist)
            result = verify_model_checksum(model_path)

            assert result is False

    def test_verify_checksum_corrupted_model(self):
        """Verification should raise ValueError if model file is corrupted."""
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", RandomForestClassifier(n_estimators=10, random_state=42)),
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "test_model.joblib")

            # Save model (creates checksum)
            save_model(pipeline, model_path)

            # Corrupt the model file
            with open(model_path, "ab") as f:
                f.write(b"corrupted data")

            # Verification should raise ValueError
            with pytest.raises(ValueError, match="checksum mismatch"):
                verify_model_checksum(model_path)

    def test_verify_checksum_tampered_checksum_file(self):
        """Verification should raise ValueError if checksum file is tampered."""
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", RandomForestClassifier(n_estimators=10, random_state=42)),
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "test_model.joblib")
            checksum_path = f"{model_path}.sha256"

            # Save model (creates checksum)
            save_model(pipeline, model_path)

            # Tamper with checksum file
            with open(checksum_path, "r") as f:
                checksum_data = json.load(f)

            checksum_data["checksum"] = "invalid_checksum_" + "0" * 48

            with open(checksum_path, "w") as f:
                json.dump(checksum_data, f)

            # Verification should raise ValueError
            with pytest.raises(ValueError, match="checksum mismatch"):
                verify_model_checksum(model_path)


@pytest.mark.integration
class TestChecksumIntegration:
    """Test checksum integration with model save/load workflow."""

    def test_save_and_load_with_checksum(self):
        """Full workflow: save model, verify checksum, load model."""
        import joblib

        from src.utils.io import verify_model_checksum

        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", RandomForestClassifier(n_estimators=10, random_state=42)),
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "test_model.joblib")

            # Step 1: Save model with checksum
            checksum = save_model(pipeline, model_path)
            assert checksum is not None

            # Step 2: Verify checksum
            verified = verify_model_checksum(model_path)
            assert verified is True

            # Step 3: Load model (would fail if checksum mismatch)
            loaded_pipeline = joblib.load(model_path)
            assert loaded_pipeline is not None

    def test_metadata_includes_checksum(self):
        """Metadata should include model checksum."""
        from src.config import TrainingConfig

        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", RandomForestClassifier(n_estimators=10, random_state=42)),
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            metadata_path = os.path.join(tmpdir, "metadata.json")

            # Save metadata with checksum
            save_metadata(
                run_id="test_run",
                best_model=pipeline,
                best_params={"param": "value"},
                metrics={
                    "accuracy": 0.8,
                    "precision": 0.75,
                    "recall": 0.7,
                    "f1": 0.72,
                    "roc_auc": 0.85,
                    "avg_precision": 0.8,
                    "confusion_matrix": {"tn": 100, "fp": 20, "fn": 15, "tp": 65},
                },
                test_metrics={
                    "accuracy": 0.79,
                    "precision": 0.74,
                    "recall": 0.69,
                    "f1": 0.71,
                    "roc_auc": 0.84,
                    "avg_precision": 0.79,
                },
                search_time=10.5,
                output_path=metadata_path,
                config=TrainingConfig(),
                model_checksum="abc123def456",  # Test checksum
            )

            # Load and verify metadata
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            assert "model_checksum" in metadata
            assert metadata["model_checksum"]["sha256"] == "abc123def456"
            assert metadata["model_checksum"]["algorithm"] == "sha256"


@pytest.mark.unit
class TestChecksumEdgeCases:
    """Test edge cases and error handling."""

    def test_checksum_empty_file(self):
        """Checksum should work for empty file."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name  # Empty file

        try:
            checksum = compute_file_checksum(temp_path)

            # SHA256 of empty file is a known constant
            expected = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
            assert checksum == expected
        finally:
            os.unlink(temp_path)

    def test_checksum_large_file(self):
        """Checksum should handle large files efficiently (chunked reading)."""
        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
            # Write 10MB of data
            f.write(b"x" * (10 * 1024 * 1024))
            temp_path = f.name

        try:
            import time

            start = time.time()
            checksum = compute_file_checksum(temp_path)
            duration = time.time() - start

            assert checksum is not None
            assert len(checksum) == 64
            # Should complete in reasonable time (<2s for 10MB)
            assert duration < 2.0, f"Checksum took {duration:.2f}s (too slow)"
        finally:
            os.unlink(temp_path)

    def test_verify_checksum_invalid_json_in_checksum_file(self):
        """Handle gracefully if checksum file contains invalid JSON."""
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", RandomForestClassifier(n_estimators=10, random_state=42)),
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "test_model.joblib")
            checksum_path = f"{model_path}.sha256"

            # Save model
            save_model(pipeline, model_path)

            # Corrupt checksum file with invalid JSON
            with open(checksum_path, "w") as f:
                f.write("invalid json {{{")

            # Should return False (not raise)
            result = verify_model_checksum(model_path)
            assert result is False


@pytest.mark.integration
class TestLoadModelFailClosed:
    """load_model must refuse unverified models unless explicitly overridden."""

    @staticmethod
    def _save_unverified_model(tmpdir):
        """Save a model then delete its checksum sidecar; return the model path."""
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", RandomForestClassifier(n_estimators=2, random_state=42)),
            ]
        )
        model_path = os.path.join(tmpdir, "churn_model_test.joblib")
        save_model(pipeline, model_path)
        os.remove(f"{model_path}.sha256")  # simulate missing sidecar
        return model_path

    def test_refuses_model_without_checksum(self, monkeypatch):
        """Missing checksum sidecar -> refuse to load (fail closed)."""
        from src.api.service import load_model

        monkeypatch.setenv("MODEL_SOURCE", "local")
        monkeypatch.delenv("ALLOW_UNVERIFIED_MODELS", raising=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = self._save_unverified_model(tmpdir)
            with pytest.raises(RuntimeError, match="checksum"):
                load_model(model_path)

    def test_allows_unverified_model_with_override(self, monkeypatch):
        """ALLOW_UNVERIFIED_MODELS=true is an explicit escape hatch."""
        from src.api.service import load_model

        monkeypatch.setenv("MODEL_SOURCE", "local")
        monkeypatch.setenv("ALLOW_UNVERIFIED_MODELS", "true")

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = self._save_unverified_model(tmpdir)
            model, version, features, threshold = load_model(model_path)
            assert model is not None
