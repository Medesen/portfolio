"""
Drift endpoints: batch drift analysis (/drift) and baseline inspection (/drift/info).

Both endpoints work strictly against the metadata cached WITH the serving
model — never a fresh disk read, which could pair the still-serving model
with a newer retrained model's baseline. Like-for-like drift requires the
model, threshold, and reference statistics to come from the same load event.
"""

from datetime import datetime
from typing import Any, Optional

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

from ...utils.drift import analyze_drift
from ...utils.logger import get_logger
from ...utils.prometheus_metrics import (
    drift_detected_count,
    features_drifted_gauge,
    prediction_error_count,
)
from .. import retraining
from ..auth import verify_token
from ..dependencies import get_model_cache
from ..schemas import DriftAnalysisRequest, DriftAnalysisResponse
from ..service import get_model_from_cache
from ..settings import RATE_LIMIT_DRIFT, RATE_LIMIT_ENABLED, drift_config, limiter
from ..types import ModelCache
from ..validation import align_schema

logger = get_logger("churn_api")

router = APIRouter()


@router.post("/drift", response_model=DriftAnalysisResponse, tags=["Drift Detection"])
@limiter.limit(RATE_LIMIT_DRIFT if RATE_LIMIT_ENABLED else "999999/minute")
async def analyze_drift_endpoint(
    drift_request: DriftAnalysisRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    model_cache: ModelCache = Depends(get_model_cache),
    _token: Optional[str] = Depends(verify_token),
) -> DriftAnalysisResponse:
    """
    Analyze drift in input data and predictions.

    CPU-bound drift analysis and model predictions are offloaded
    to a thread pool to prevent blocking the async event loop.

    Compares current data to training baseline to detect distribution shifts.
    Monitors:
    - Numeric feature drift (relative change in mean/std)
    - Categorical feature drift (Population Stability Index)
    - Prediction drift (change in positive prediction rate)

    Args:
        drift_request: Batch of customer data to analyze
        request: FastAPI Request object (required by slowapi)
        background_tasks: FastAPI background tasks for optional retraining

    Returns:
        Drift analysis report with detailed metrics

    Raises:
        400: Batch size exceeds limit or empty batch
        503: Model or reference statistics not available
        500: Drift analysis failed
    """
    logger.info(f"Drift analysis requested for {len(drift_request.customers)} customers")

    # Validate batch size
    if len(drift_request.customers) == 0:
        raise HTTPException(
            status_code=400, detail="Drift analysis requires at least one customer record"
        )

    # Enforce the configured minimum sample size: PSI/KS statistics on a handful
    # of records are noise, not drift evidence.
    if len(drift_request.customers) < drift_config.min_sample_size:
        raise HTTPException(
            status_code=400,
            detail=f"Drift analysis requires at least {drift_config.min_sample_size} records "
            f"(got {len(drift_request.customers)}). Distribution statistics on smaller "
            "batches are unreliable. Adjust DRIFT_MIN_SAMPLE_SIZE if needed.",
        )

    if len(drift_request.customers) > drift_config.max_drift_batch_size:
        raise HTTPException(
            status_code=400,
            detail=f"Drift analysis batch size ({len(drift_request.customers)}) exceeds maximum "
            f"({drift_config.max_drift_batch_size}). Please split into smaller batches or increase "
            f"MAX_DRIFT_BATCH_SIZE.",
        )

    # Get model, its recorded schema, and its deployed decision threshold from the cache
    model, version, expected_features, threshold = get_model_from_cache(model_cache)

    if model is None:
        raise HTTPException(status_code=503, detail="Drift analysis unavailable: model not loaded")

    metadata = model_cache.get("metadata")
    if metadata is None:
        raise HTTPException(
            status_code=503,
            detail="Drift analysis unavailable: no metadata loaded with the model",
        )

    if "reference_statistics" not in metadata:
        logger.error("Reference statistics not found in metadata - model needs retraining")
        raise HTTPException(
            status_code=503,
            detail="Drift analysis unavailable: reference statistics not found. "
            "Please retrain the model to generate baseline statistics.",
        )

    try:
        # Convert to DataFrame
        df = pd.DataFrame(drift_request.customers)

        # Align to the model's recorded schema — the same preprocessing step
        # /predict applies — so missing/extra/reordered columns behave
        # identically in both paths instead of failing only here.
        if expected_features:
            df, alignment_info = align_schema(df, expected_features)
            if any(alignment_info.values()):
                logger.warning(f"Drift batch schema alignment applied: {alignment_info}")

        # Score the batch once at the DEPLOYED decision threshold (the same one
        # used by /predict), not the default 0.5 cut. Prediction drift compares
        # this predicted-positive rate against the training baseline stored at
        # that threshold. Offload CPU-bound scoring to the thread pool.
        probabilities = await run_in_threadpool(lambda: model.predict_proba(df)[:, 1])
        predictions = (probabilities >= threshold).astype(int)

        # Analyze drift
        # Offload CPU-bound drift analysis to thread pool
        drift_report = await run_in_threadpool(
            analyze_drift,
            metadata["reference_statistics"],
            df,
            predictions,
            drift_config.numeric_threshold,
            drift_config.categorical_threshold,
            drift_config.prediction_threshold,
            probabilities,
        )

        # Update Prometheus metrics
        if drift_report["overall_drift_detected"]:
            # Count drift by type
            for feature, result in drift_report["numeric_features"].items():
                if result["drift_detected"]:
                    drift_detected_count.labels(drift_type="numeric").inc()

            for feature, result in drift_report["categorical_features"].items():
                if result["drift_detected"]:
                    drift_detected_count.labels(drift_type="categorical").inc()

            if drift_report["prediction_drift"]["drift_detected"]:
                drift_detected_count.labels(drift_type="prediction").inc()

            features_drifted_gauge.set(drift_report["summary"]["n_features_drifted"])

            # Log warning
            logger.warning(
                f"DRIFT DETECTED! {drift_report['summary']['n_features_drifted']} "
                f"features drifted: {drift_report['summary']['drifted_features']}"
            )

            # Optionally trigger automatic retraining (atomically reserve first)
            if retraining.AUTO_RETRAIN_ON_DRIFT and retraining.check_and_reserve_retraining():
                logger.info("AUTO_RETRAIN_ON_DRIFT enabled - triggering retraining")
                background_tasks.add_task(retraining.retrain_model_task)
        else:
            logger.info("No significant drift detected")

        # Return response
        return DriftAnalysisResponse(
            overall_drift_detected=drift_report["overall_drift_detected"],
            n_features_drifted=drift_report["summary"]["n_features_drifted"],
            drifted_features=drift_report["summary"]["drifted_features"],
            prediction_drift_detected=drift_report["prediction_drift"]["drift_detected"],
            detailed_report=drift_report,
        )

    except Exception as e:
        # Log full details; return a generic message (never leak internals — same
        # policy as the global exception handler)
        logger.error(f"Drift analysis failed: {e}", exc_info=True)
        prediction_error_count.labels(error_type="drift_analysis_error").inc()
        raise HTTPException(
            status_code=500,
            detail="Drift analysis failed due to an internal error. Please try again later.",
        )


@router.get("/drift/info", tags=["Drift Detection"])
async def drift_info(
    model_cache: ModelCache = Depends(get_model_cache),
    _token: Optional[str] = Depends(verify_token),
) -> dict[str, Any]:
    """
    Get information about drift detection configuration and baseline.

    Returns reference statistics and thresholds used for drift detection.
    Useful for understanding the baseline and debugging drift alerts.

    Returns:
        Drift configuration and baseline statistics

    Raises:
        503: Reference statistics not available
    """
    logger.info("=== DRIFT INFO ENDPOINT CALLED ===")

    # Get model version - Use model_cache from dependency
    _, version, _, _ = get_model_from_cache(model_cache)
    logger.info(f"Got model version: {version}")

    # Use the metadata cached with the serving model (never a fresh disk read,
    # which could belong to a newer retrained model)
    metadata = model_cache.get("metadata")
    if metadata is None:
        raise HTTPException(
            status_code=503, detail="Drift info unavailable: no metadata loaded with the model"
        )
    logger.info(f"Drift info: Metadata keys: {list(metadata.keys())}")

    if "reference_statistics" not in metadata:
        logger.error(
            f"reference_statistics not in metadata. Available keys: {list(metadata.keys())}"
        )
        raise HTTPException(
            status_code=503,
            detail="Drift info unavailable: reference statistics not found. "
            "Please retrain the model to generate baseline statistics.",
        )

    ref_stats = metadata["reference_statistics"]

    # Calculate model age
    model_timestamp = datetime.fromisoformat(metadata["timestamp"])
    age_days = (datetime.now() - model_timestamp).days

    prediction_baseline = ref_stats.get("prediction")

    return {
        "baseline": {
            "n_samples": ref_stats["target"]["n_samples"],
            "positive_rate": ref_stats["target"]["positive_rate"],
            "n_numeric_features": len(ref_stats["numeric"]),
            "n_categorical_features": len(ref_stats["categorical"]),
            "numeric_features": list(ref_stats["numeric"].keys()),
            "categorical_features": list(ref_stats["categorical"].keys()),
            # Prediction baseline (present for models trained with the drift fix):
            # predicted-positive rate at the tuned decision threshold.
            "prediction": (
                {
                    "threshold": prediction_baseline["threshold"],
                    "positive_rate": prediction_baseline["positive_rate"],
                    "proba_mean": prediction_baseline.get("proba_mean"),
                }
                if prediction_baseline is not None
                else None
            ),
        },
        "thresholds": {
            "numeric": drift_config.numeric_threshold,
            "categorical": drift_config.categorical_threshold,
            "prediction": drift_config.prediction_threshold,
        },
        "model_info": {
            "run_id": metadata.get("run_id", "unknown"),
            "trained_at": metadata["timestamp"],
            "age_days": age_days,
        },
        "configuration": {
            "max_drift_batch_size": drift_config.max_drift_batch_size,
            "auto_retrain_on_drift": retraining.AUTO_RETRAIN_ON_DRIFT,
            "min_retrain_interval_hours": retraining.MIN_RETRAIN_INTERVAL_HOURS,
        },
    }
