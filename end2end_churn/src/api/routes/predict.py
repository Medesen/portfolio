"""Prediction endpoint: scores a single customer at the deployed threshold."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

from ...utils.logger import get_logger
from ...utils.prometheus_metrics import prediction_error_count
from ..auth import verify_token
from ..dependencies import get_model_cache
from ..schemas import PredictionRequest, PredictionResponse
from ..service import get_model_from_cache, predict
from ..settings import RATE_LIMIT_ENABLED, RATE_LIMIT_PREDICT, limiter, service_config
from ..types import ModelCache

logger = get_logger("churn_api")

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
@limiter.limit(RATE_LIMIT_PREDICT if RATE_LIMIT_ENABLED else "999999/minute")
async def predict_churn(
    prediction_request: PredictionRequest,
    request: Request,
    model_cache: ModelCache = Depends(get_model_cache),
    token: Optional[str] = Depends(verify_token),
) -> PredictionResponse:
    """
    Predict churn probability for a customer.

    CPU-bound model inference is offloaded to a thread pool
    to prevent blocking the async event loop and maintain server responsiveness.

    Request ID is included in response for tracing logs.

    Args:
        prediction_request: Customer features
        request: FastAPI Request object (for accessing request_id, required by slowapi)
        model_cache: Model cache from app.state (injected by dependency)
        token: Optional bearer token (required if service_token is configured)

    Returns:
        Prediction with probability, binary classification, risk level, and request_id

    Raises:
        HTTPException with appropriate status codes:
            - 401: Unauthorized (invalid/missing token when auth enabled)
            - 422: Unprocessable Entity (data validation failed)
            - 500: Internal Server Error (unexpected error)
            - 503: Service Unavailable (model not loaded)

    Security:
        If SERVICE_TOKEN environment variable is set, requires Authorization header:
        Authorization: Bearer <token>
    """
    # Get request ID from middleware
    request_id = request.state.request_id

    logger.info("Prediction request received")
    logger.debug(f"Authentication: {'enabled' if service_config.service_token else 'disabled'}")

    # Check 1: Model loaded? (503 Service Unavailable)
    try:
        model, version, _, _ = get_model_from_cache(model_cache)
        if model is None:
            logger.error("Prediction attempted but model not loaded")
            prediction_error_count.labels(error_type="model_not_loaded").inc()
            raise HTTPException(
                status_code=503,
                detail="Service unavailable: Model not loaded yet. Try again in a moment.",
            )
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Error checking model: {e}", exc_info=True)
        prediction_error_count.labels(error_type="model_check_failed").inc()
        raise HTTPException(
            status_code=503, detail="Service unavailable: Unable to verify model status"
        )

    # Predict - Run in thread pool to avoid blocking event loop
    # Pass request_id to prediction function for response
    try:
        logger.debug("Processing prediction")
        # Offload CPU-bound prediction to thread pool
        response = await run_in_threadpool(predict, prediction_request, model_cache, request_id)

        # Log success
        logger.info(
            f"Prediction successful: churn={response.churn_prediction}, "
            f"probability={response.churn_probability:.4f}, "
            f"risk={response.risk_level}"
        )

        # Log warnings if any (schema mismatches, data quality issues)
        if response.warnings:
            logger.warning(f"Prediction warnings: {response.warnings}")

        return response

    except ValueError as e:
        # Data validation error (422 Unprocessable Entity)
        logger.error(f"Data validation error: {e}", exc_info=True)
        prediction_error_count.labels(error_type="validation_error").inc()
        raise HTTPException(status_code=422, detail=f"Data validation failed: {str(e)}")

    except Exception as e:
        # Unexpected error (500 Internal Server Error)
        logger.error(f"Unexpected error during prediction: {type(e).__name__}: {e}", exc_info=True)
        prediction_error_count.labels(error_type="prediction_failed").inc()
        raise HTTPException(
            status_code=500,
            detail="Prediction failed due to an internal error. Please try again later.",
        )
