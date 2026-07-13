"""Core service logic for churn prediction API."""

import json
import os
import time
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from ..utils.logger import get_logger
from ..utils.prometheus_metrics import preprocessing_time  # Track preprocessing latency separately
from ..utils.prometheus_metrics import (
    data_quality_issue_count,
    model_prediction_time,
    prediction_count,
    prediction_error_count,
    prediction_probability_histogram,
    schema_mismatch_count,
)
from .schemas import PredictionRequest, PredictionResponse
from .types import ModelCache
from .validation import align_schema, generate_alignment_warnings, validate_data

# Initialize logger
logger = get_logger("churn_api")

# Import MLflow for registry support
try:
    import mlflow
    import mlflow.pyfunc

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not available - registry loading disabled")


# Model cache removed - now managed via FastAPI app.state for multi-worker safety
# This prevents issues when running with multiple uvicorn workers
# Model state is now stored in app.state and injected via dependencies


def load_model_from_registry(
    model_name: str = "churn_prediction_model", stage: str = "Production"
) -> tuple[Pipeline, str, list, float, Optional[dict]]:
    """
    Load model from MLflow Registry by stage.

    Args:
        model_name: Name of registered model
        stage: Model stage (Production, Staging, etc.)

    Returns:
        Tuple of (model_pipeline, model_version, expected_features, threshold,
        metadata) — metadata is the run's metadata dict, or None if unavailable

    Raises:
        Exception: If model loading from registry fails
    """
    if not MLFLOW_AVAILABLE:
        raise Exception("MLflow not available - cannot load from registry")

    logger.info(f"Loading model from MLflow Registry: {model_name} ({stage} stage)")

    try:
        # Set tracking URI
        mlflow.set_tracking_uri("./mlruns")

        # Load model from registry
        model_uri = f"models:/{model_name}/{stage}"
        model = mlflow.pyfunc.load_model(model_uri)

        # Get the underlying sklearn model
        sk_model = model._model_impl.python_model if hasattr(model, "_model_impl") else model

        # Get model version info
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        versions = client.get_latest_versions(model_name, stages=[stage])

        if not versions:
            raise Exception(f"No model found in {stage} stage")

        version = versions[0].version
        run_id = versions[0].run_id

        logger.info(f"Loaded model from registry: version {version}, run {run_id[:8]}")

        # Try to load metadata from the run
        expected_features = []
        threshold = 0.5
        metadata = None

        try:
            # Get run artifacts
            run = client.get_run(run_id)
            artifacts = client.list_artifacts(run_id, "metadata")

            # Look for metadata file
            metadata_files = [a for a in artifacts if a.path.startswith("metadata/metadata_")]
            if metadata_files:
                # Download and parse metadata
                import tempfile

                with tempfile.TemporaryDirectory() as tmp_dir:
                    metadata_path = client.download_artifacts(
                        run_id, metadata_files[0].path, tmp_dir
                    )
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)

                        # Extract schema
                        if "schema" in metadata and "all_features" in metadata["schema"]:
                            expected_features = metadata["schema"]["all_features"]
                            logger.debug(
                                f"Loaded schema with {len(expected_features)} features from registry"
                            )

                        # Extract threshold
                        if "threshold" in metadata and "chosen_threshold" in metadata["threshold"]:
                            threshold = metadata["threshold"]["chosen_threshold"]
                            strategy = metadata["threshold"].get("chosen_strategy", "unknown")
                            logger.info(
                                f"Loaded tuned threshold from registry: {threshold:.4f} (strategy: {strategy})"
                            )
        except Exception as e:
            logger.warning(f"Could not load metadata from registry run: {e}")
            logger.info("Using defaults: no schema, threshold=0.5")

        logger.info(f"Model loaded successfully from registry: {model_name} v{version} ({stage})")
        return sk_model, f"{version}", expected_features, threshold, metadata

    except Exception as e:
        logger.error(f"Failed to load model from registry: {e}", exc_info=True)
        raise


def find_latest_metadata_file() -> Optional[Path]:
    """
    Return the metadata file paired with the latest model.

    Prefers ``models/metadata_latest.json`` (written next to
    ``churn_model_latest.joblib`` at save time), so the metadata normally
    matches the loaded latest model. Falls back to the newest ``metadata_*.json``
    with a warning for models trained before paired-latest metadata existed —
    that fallback relies on lexical ordering and can mis-pair a stray metadata
    file, so retraining to generate ``metadata_latest.json`` is preferred.

    Known limitation (accepted): training publishes the model and metadata
    files sequentially, and load_model() reads them sequentially, so a service
    starting exactly inside a publication window could pair a new model with
    old metadata for that process's lifetime. Once loaded, the pair is cached
    together and stays coherent. Hardening this would mean immutable versioned
    artifacts plus an atomically replaced manifest pointer — more machinery
    than this single-writer setup warrants.

    Returns:
        Path to the metadata file, or None if no metadata is present.
    """
    models_dir = Path("models")
    latest = models_dir / "metadata_latest.json"
    if latest.exists():
        return latest

    candidates = sorted(models_dir.glob("metadata_*.json"), reverse=True)
    if candidates:
        logger.warning(
            "metadata_latest.json not found; falling back to newest metadata file (%s). "
            "Retrain to generate paired latest metadata.",
            candidates[0].name,
        )
        return candidates[0]
    return None


def load_model(model_path: str = None) -> tuple[Pipeline, str, list, float, Optional[dict]]:
    """
    Load the trained model pipeline together with its metadata.

    Supports loading from:
    1. Local file path (default or from MODEL_PATH environment variable)
    2. MLflow Registry (if MODEL_SOURCE=registry)

    The metadata dict is returned alongside the model so callers can cache the
    two as one coherent unit — consumers must never pair the cached model with
    metadata re-read from disk later, which can belong to a newer retrained model.

    Args:
        model_path: Path to the model file (optional, defaults to MODEL_PATH env var
                    or "models/churn_model_latest.joblib")

    Returns:
        Tuple of (model_pipeline, model_version, expected_features, threshold,
        metadata) — metadata is the parsed dict, or None if no metadata was found

    Raises:
        FileNotFoundError: If model file doesn't exist
        Exception: If model loading fails
    """
    # Check if we should load from registry
    model_source = os.getenv("MODEL_SOURCE", "local").lower()

    if model_source == "registry":
        model_stage = os.getenv("MODEL_STAGE", "Production")
        model_name = os.getenv("MODEL_NAME", "churn_prediction_model")
        logger.info(f"Loading model from registry: {model_name} ({model_stage} stage)")
        return load_model_from_registry(model_name, model_stage)

    # Default: Load from local file
    # Check environment variable first, then use default
    if model_path is None:
        model_path = os.getenv("MODEL_PATH", "models/churn_model_latest.joblib")

    model_path_obj = Path(model_path)
    logger.info(f"Loading model from local file: {model_path}")

    if not model_path_obj.exists():
        logger.error(f"Model not found at {model_path}")
        raise FileNotFoundError(f"Model not found at {model_path}")

    # Verify model checksum before loading. A checksum MISMATCH always fails hard
    # (possible tampering/corruption). A missing or unreadable checksum sidecar
    # now also fails closed — refusing to deserialize an unverified joblib file —
    # unless ALLOW_UNVERIFIED_MODELS=true is set as an explicit, logged dev override.
    from ..utils.io import verify_model_checksum

    allow_unverified = os.getenv("ALLOW_UNVERIFIED_MODELS", "false").lower() == "true"
    try:
        verified = verify_model_checksum(model_path)
    except ValueError as e:
        # Checksum mismatch - critical security issue
        logger.critical(f"Model integrity check failed: {e}")
        raise
    except Exception as e:
        # Verification could not be completed (e.g., unreadable sidecar)
        verified = False
        logger.error(f"Checksum verification could not be completed: {e}")

    if not verified:
        if allow_unverified:
            logger.warning(
                "Loading model WITHOUT checksum verification because "
                "ALLOW_UNVERIFIED_MODELS=true. Do not use this in production."
            )
        else:
            raise RuntimeError(
                f"Refusing to load model without a valid checksum: {model_path}. "
                "The .sha256 sidecar is missing or unreadable. Re-save/retrain the "
                "model to regenerate it, or set ALLOW_UNVERIFIED_MODELS=true to "
                "override (insecure)."
            )

    # Load model
    try:
        model = joblib.load(model_path)
        logger.debug(f"Model file loaded successfully: {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model file: {e}", exc_info=True)
        raise

    # Try to load metadata to get version, schema, and threshold
    version = "unknown"
    expected_features = []
    threshold = 0.5  # Default threshold
    metadata = None

    # If loading latest model, use the metadata paired with the latest model
    if "latest" in str(model_path):
        metadata_path = find_latest_metadata_file()
    else:
        # Extract version from model filename
        model_name = model_path_obj.stem  # e.g., "churn_model_20251024_183147"
        version_str = model_name.replace("churn_model_", "")
        metadata_path = Path(f"models/metadata_{version_str}.json")

    # Load version, schema, and threshold from metadata if available
    if metadata_path and metadata_path.exists():
        logger.debug(f"Loading metadata from {metadata_path}")
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                version = metadata.get("run_id", "unknown")
                # Extract expected features from schema
                if "schema" in metadata and "all_features" in metadata["schema"]:
                    expected_features = metadata["schema"]["all_features"]
                    logger.debug(f"Loaded schema with {len(expected_features)} features")
                # Extract tuned threshold
                if "threshold" in metadata and "chosen_threshold" in metadata["threshold"]:
                    threshold = metadata["threshold"]["chosen_threshold"]
                    strategy = metadata["threshold"].get("chosen_strategy", "unknown")
                    logger.info(f"Loaded tuned threshold: {threshold:.4f} (strategy: {strategy})")
                else:
                    logger.debug("No tuned threshold in metadata, using default 0.5")
        except Exception as e:
            metadata = None
            logger.warning(f"Failed to load metadata from {metadata_path}: {e}")
    else:
        logger.warning(f"Metadata file not found: {metadata_path}")

    logger.info(
        f"Model loaded successfully: version={version}, features={len(expected_features)}, threshold={threshold:.4f}"
    )
    return model, version, expected_features, threshold, metadata


def get_model_from_cache(model_cache: ModelCache) -> tuple[Pipeline, str, list, float]:
    """
    Get cached model or load it if not yet loaded.

    Uses model_cache dict (from app.state) instead of global variables.
    This is safe for multi-worker deployments.

    Args:
        model_cache: Dictionary with keys: model, version, features, threshold

    Returns:
        Tuple of (model_pipeline, model_version, expected_features, threshold)
    """
    if model_cache.get("model") is None:
        logger.debug("Model not cached, loading from disk")
        model, version, features, threshold, metadata = load_model()
        model_cache["model"] = model
        model_cache["version"] = version
        model_cache["features"] = features
        model_cache["threshold"] = threshold
        # Cache the metadata together with the model: after a retrain overwrites
        # the files on disk, the old model keeps serving until restart, and its
        # drift baseline/threshold must keep coming from ITS metadata — not from
        # a fresh disk read that would belong to the new model.
        model_cache["metadata"] = metadata
    else:
        logger.debug(
            f"Using cached model: version {model_cache['version']}, threshold {model_cache['threshold']:.4f}"
        )

    return (
        model_cache["model"],
        model_cache["version"],
        model_cache["features"],
        model_cache["threshold"],
    )


def get_metadata_from_cache(model_cache: ModelCache) -> Optional[dict]:
    """
    Get the metadata dict belonging to the cached (serving) model.

    Ensures the model is loaded first, so the returned metadata always pairs
    with the model that get_model_from_cache serves. Returns None when the
    model was loaded without a metadata file.

    Args:
        model_cache: Model cache dictionary from app.state

    Returns:
        The cached metadata dict, or None if unavailable
    """
    get_model_from_cache(model_cache)  # ensure model + metadata are loaded together
    return model_cache.get("metadata")


def predict(
    request: PredictionRequest,
    model_cache: ModelCache,
    request_id: str = "no-request-id",
    enable_validation: bool = True,
) -> PredictionResponse:
    """
    Make a churn prediction for a single customer.

    Args:
        request: Customer features
        model_cache: Model cache dictionary from app.state
        request_id: Unique request ID for distributed tracing
        enable_validation: Whether to perform Pandera validation (can disable for performance)

    Returns:
        PredictionResponse with probability, risk level, request_id, and any warnings
    """
    # Get model, expected features, and tuned threshold
    model, version, expected_features, threshold = get_model_from_cache(model_cache)

    # Convert request to DataFrame (model expects DataFrame input)
    features = {
        "gender": request.gender,
        "SeniorCitizen": request.SeniorCitizen,
        "Partner": request.Partner,
        "Dependents": request.Dependents,
        "tenure": request.tenure,
        "PhoneService": request.PhoneService,
        "MultipleLines": request.MultipleLines,
        "InternetService": request.InternetService,
        "OnlineSecurity": request.OnlineSecurity,
        "OnlineBackup": request.OnlineBackup,
        "DeviceProtection": request.DeviceProtection,
        "TechSupport": request.TechSupport,
        "StreamingTV": request.StreamingTV,
        "StreamingMovies": request.StreamingMovies,
        "Contract": request.Contract,
        "PaperlessBilling": request.PaperlessBilling,
        "PaymentMethod": request.PaymentMethod,
        "MonthlyCharges": request.MonthlyCharges,
        "TotalCharges": request.TotalCharges,
    }

    # Create DataFrame with single row
    df = pd.DataFrame([features])

    # Collect warnings
    warnings_list = []

    # Track preprocessing time for latency monitoring
    preproc_start_time = time.time()

    # Optional: Validate data quality with Pandera
    if enable_validation:
        logger.debug("Performing data quality validation with Pandera")
        is_valid, validation_errors = validate_data(df, enable_validation=True)
        if not is_valid:
            logger.warning(f"Data quality issues detected: {validation_errors}")
            warnings_list.extend([f"Data quality: {err}" for err in validation_errors])
            # Track data quality issues in metrics
            for error in validation_errors:
                if "range" in error.lower():
                    data_quality_issue_count.labels(issue_type="invalid_range").inc()
                elif "category" in error.lower():
                    data_quality_issue_count.labels(issue_type="invalid_category").inc()
                else:
                    data_quality_issue_count.labels(issue_type="other").inc()
        else:
            logger.debug("Data quality validation passed")

    # Schema alignment - handles missing/extra/reordered columns gracefully
    if expected_features:
        logger.debug("Performing schema alignment")
        df_aligned, alignment_info = align_schema(df, expected_features)
        alignment_warnings = generate_alignment_warnings(alignment_info)
        if alignment_warnings:
            logger.debug(f"Schema alignment: {alignment_info}")

            # Track schema mismatches in metrics
            if alignment_info["missing_columns"]:
                schema_mismatch_count.labels(mismatch_type="missing_columns").inc()
            if alignment_info["extra_columns"]:
                schema_mismatch_count.labels(mismatch_type="extra_columns").inc()
            if alignment_info["reordered"]:
                schema_mismatch_count.labels(mismatch_type="reordered_columns").inc()

        warnings_list.extend(alignment_warnings)
    else:
        # Fallback if no schema information available
        logger.warning("Schema alignment skipped (no expected features available)")
        df_aligned = df

    # Record preprocessing duration
    preproc_duration = time.time() - preproc_start_time
    preprocessing_time.observe(preproc_duration)

    # Make prediction with timing for performance monitoring
    logger.debug("Generating prediction")

    # Time the model prediction (for performance monitoring)
    pred_start_time = time.time()
    try:
        churn_probability = float(
            model.predict_proba(df_aligned)[0, 1]
        )  # Probability of class 1 (churn)
        # Use tuned threshold from metadata
        churn_prediction = "Yes" if churn_probability >= threshold else "No"
        logger.debug(f"Applied threshold {threshold:.4f} for prediction")
    finally:
        pred_duration = time.time() - pred_start_time
        model_prediction_time.observe(pred_duration)

    # Determine risk level
    if churn_probability >= 0.7:
        risk_level = "High"
    elif churn_probability >= 0.4:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    logger.debug(
        f"Prediction generated: probability={churn_probability:.4f}, "
        f"prediction={churn_prediction}, risk={risk_level}"
    )

    # Track prediction metrics
    prediction_count.labels(predicted_class=churn_prediction).inc()
    prediction_probability_histogram.observe(churn_probability)

    return PredictionResponse(
        churn_probability=churn_probability,
        churn_prediction=churn_prediction,
        risk_level=risk_level,
        model_version=version,
        warnings=warnings_list if warnings_list else None,
        request_id=request_id,  # Include request ID for distributed tracing
    )


def reset_model_cache(model_cache: ModelCache) -> None:
    """
    Reset the model cache (useful for testing or reloading).

    Args:
        model_cache: Model cache dictionary from app.state
    """
    logger.debug("Resetting model cache")
    model_cache.clear()
    model_cache.update(
        {"model": None, "version": None, "features": None, "threshold": 0.5, "metadata": None}
    )
