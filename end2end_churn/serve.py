"""
FastAPI application for serving churn predictions.

This provides a REST API with the following endpoints:
- GET  /         - Root endpoint with API info
- GET  /health   - Detailed health check endpoint
- GET  /healthz  - Liveness probe (Kubernetes)
- GET  /readyz   - Readiness probe (Kubernetes)
- POST /predict  - Make churn prediction
- GET  /metrics  - Prometheus metrics

Simple FastAPI Service:
- Request/response validation with Pydantic
- Model loading with caching
- Health checks
- Auto-generated API documentation (Swagger UI)

Structured Logging:
- Replaced print statements with structured logging
- Console and file handlers with rotation
- Different log levels for development and production

Prometheus Metrics:
- Request counters, latency histograms, error metrics
- Prediction distribution tracking
- Schema mismatch monitoring
- /metrics endpoint for Prometheus scraping

Health Checks & Error Handling:
- Enhanced /health with detailed status (503 when unhealthy)
- Separate liveness (/healthz) and readiness (/readyz) probes
- Request validation middleware (size limits)
- Optional bearer token authentication
- Comprehensive error handling (never leak stack traces)
- Proper HTTP status codes (400, 401, 422, 500, 503)

Drift Detection:
- POST /drift - Analyze drift in input data and predictions
- GET /drift/info - Get drift configuration and baseline statistics
- POST /retrain - Trigger automatic model retraining
- Statistical drift detection (PSI for categorical, relative change for numeric)
- Prediction drift monitoring
- Automatic retraining with safeguards

Usage:
    # Development (with auto-reload)
    uvicorn serve:app --reload --host 0.0.0.0 --port 8000

    # Production
    uvicorn serve:app --host 0.0.0.0 --port 8000 --workers 4

    # With authentication
    export SERVICE_TOKEN="your-secret-token"
    uvicorn serve:app --host 0.0.0.0 --port 8000
"""

import asyncio
import fcntl
import os
import secrets
import subprocess
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from prometheus_client import Counter, make_asgi_app
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.api.schemas import PredictionRequest, PredictionResponse
from src.api.service import (
    get_metadata_from_cache,
    get_model_from_cache,
    predict,
    reset_model_cache,
)
from src.api.types import ModelCache
from src.api.validation import align_schema
from src.config import DriftConfig, ServiceConfig
from src.utils.drift import analyze_drift
from src.utils.logger import request_id_context, setup_logger
from src.utils.prometheus_metrics import (
    drift_detected_count,
    features_drifted_gauge,
    init_service_metrics,
    model_age_days,
    prediction_error_count,
    request_count,
    request_duration,
    service_start_time,
    set_model_metrics,
)

# Load configuration from environment variables
# Now supports secure secret loading via SERVICE_TOKEN_FILE
service_config = ServiceConfig.from_env()

# Rate limiting configuration
RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
RATE_LIMIT_PREDICT = os.getenv("RATE_LIMIT_PREDICT", "100/minute")
RATE_LIMIT_DRIFT = os.getenv("RATE_LIMIT_DRIFT", "20/minute")

# Request timeout configuration (Step 10)
REQUEST_TIMEOUT_SECONDS = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "30"))
REQUEST_TIMEOUT_ENABLED = os.getenv("REQUEST_TIMEOUT_ENABLED", "true").lower() == "true"

# Prometheus counter for timeout events
request_timeout_count = Counter(
    "churn_api_request_timeouts_total",
    "Total number of requests that exceeded timeout",
    ["endpoint"],
)

drift_config = DriftConfig(
    numeric_threshold=float(os.getenv("DRIFT_THRESHOLD_NUMERIC", "0.2")),
    categorical_threshold=float(os.getenv("DRIFT_THRESHOLD_CATEGORICAL", "0.25")),
    prediction_threshold=float(os.getenv("DRIFT_THRESHOLD_PREDICTION", "0.1")),
    min_sample_size=int(os.getenv("DRIFT_MIN_SAMPLE_SIZE", "100")),
    max_drift_batch_size=int(os.getenv("MAX_DRIFT_BATCH_SIZE", "1000")),
)

# Initialize logger with config
logger = setup_logger(
    name="churn_api", log_level=service_config.log_level, log_file="logs/churn_service.log"
)

# Automatic retraining configuration
AUTO_RETRAIN_ON_DRIFT = os.getenv("AUTO_RETRAIN_ON_DRIFT", "false").lower() == "true"
MIN_RETRAIN_INTERVAL_HOURS = int(os.getenv("MIN_RETRAIN_INTERVAL_HOURS", "24"))

# File path for storing last retrain timestamp (multi-worker safe)
LAST_RETRAIN_FILE = Path("models/.last_retrain")

# Store service start timestamp for uptime calculation (avoids Prometheus private API access)
SERVICE_START_TIMESTAMP = time.time()

# Security: Bearer token authentication (optional)
security = HTTPBearer(auto_error=False)  # Don't auto-fail if missing

logger.info("Configuration loaded:")
logger.info(f"  • Service config:")
logger.info(f"    - MAX_BATCH_SIZE: {service_config.max_batch_size}")
logger.info(f"    - MAX_REQUEST_SIZE_MB: {service_config.max_request_size_mb}")
logger.info(f"    - Authentication: {'enabled' if service_config.service_token else 'disabled'}")
logger.info(f"  • Drift config:")
logger.info(f"    - MAX_DRIFT_BATCH_SIZE: {drift_config.max_drift_batch_size}")
logger.info(
    f"    - Thresholds: numeric={drift_config.numeric_threshold}, "
    f"categorical={drift_config.categorical_threshold}, "
    f"prediction={drift_config.prediction_threshold}"
)
logger.info(f"  • Auto-retrain on drift: {'enabled' if AUTO_RETRAIN_ON_DRIFT else 'disabled'}")


# ==============================================================================
# DRIFT DETECTION SCHEMAS
# ==============================================================================


class DriftAnalysisRequest(BaseModel):
    """Request body for drift analysis."""

    customers: list[dict[str, Any]]

    model_config = {
        "json_schema_extra": {
            "example": {
                "customers": [
                    {
                        "gender": "Female",
                        "SeniorCitizen": 0,
                        "Partner": "Yes",
                        "Dependents": "No",
                        "tenure": 12,
                        "Contract": "Month-to-month",
                        "PaperlessBilling": "Yes",
                        "PaymentMethod": "Electronic check",
                        "MonthlyCharges": 70.35,
                        "TotalCharges": 844.20,
                        "PhoneService": "Yes",
                        "MultipleLines": "No",
                        "InternetService": "Fiber optic",
                        "OnlineSecurity": "No",
                        "OnlineBackup": "Yes",
                        "DeviceProtection": "No",
                        "TechSupport": "No",
                        "StreamingTV": "Yes",
                        "StreamingMovies": "No",
                    }
                ]
            }
        }
    }


class DriftAnalysisResponse(BaseModel):
    """Response body for drift analysis."""

    overall_drift_detected: bool
    n_features_drifted: int
    drifted_features: list[str]
    prediction_drift_detected: bool
    detailed_report: Optional[dict[str, Any]] = None


# ==============================================================================
# RETRAINING HELPERS
# ==============================================================================


def get_last_retrain_time() -> Optional[datetime]:
    """
    Read last retrain timestamp from file (multi-worker safe).

    Returns:
        datetime of last retrain, or None if never retrained
    """
    if not LAST_RETRAIN_FILE.exists():
        return None

    try:
        with open(LAST_RETRAIN_FILE, "r") as f:
            # Acquire shared lock for reading
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            try:
                timestamp_str = f.read().strip()
                if timestamp_str:
                    return datetime.fromisoformat(timestamp_str)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except Exception as e:
        logger.warning(f"Could not read last retrain time: {e}")

    return None


def check_and_reserve_retraining() -> bool:
    """
    Atomically check the retrain rate limit and reserve the retraining slot.

    Under a single exclusive lock on the timestamp file, read the last-retrain
    time and, if the minimum interval has elapsed (or no prior run exists),
    write the current time as the reservation BEFORE returning True. This closes
    the check-then-act race in the previous design (separate should_allow check
    and post-training timestamp write), where two concurrent triggers could both
    pass the check and launch duplicate 10-minute trainings.

    The reservation is written up front (not after training completes) and is
    intentionally NOT rolled back on failure: a failed run still consumes the
    interval, which prevents a crash-looping trainer from retraining
    continuously. Operators can force an earlier retry by deleting
    ``models/.last_retrain``.

    Returns:
        True if this caller reserved the slot and should proceed to retrain;
        False if rate-limited by an existing reservation (or on I/O error, in
        which case we fail closed and do not retrain).
    """
    try:
        LAST_RETRAIN_FILE.parent.mkdir(parents=True, exist_ok=True)
        # Open read+write without truncating (create if missing) so the read and
        # the reservation write happen under the same lock.
        with open(LAST_RETRAIN_FILE, "a+") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.seek(0)
                timestamp_str = f.read().strip()
                last_retrain = None
                if timestamp_str:
                    try:
                        last_retrain = datetime.fromisoformat(timestamp_str)
                    except ValueError:
                        logger.warning("Invalid last-retrain timestamp; treating as unset")

                now = datetime.now()
                if last_retrain is not None and (now - last_retrain) <= timedelta(
                    hours=MIN_RETRAIN_INTERVAL_HOURS
                ):
                    return False  # still within the interval -> rate limited

                # Reserve the slot: overwrite the file with the current time.
                f.seek(0)
                f.truncate()
                f.write(now.isoformat())
                f.flush()
                return True
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except Exception as e:
        logger.error(f"Could not check/reserve retraining: {e}")
        return False


async def retrain_model_task():
    """
    Background task to retrain the model.

    The rate-limit slot must already be reserved via check_and_reserve_retraining()
    before this task is scheduled. Training runs in a worker thread via
    asyncio.to_thread so the event loop stays responsive during the (up to
    10-minute) run. The reservation is intentionally not rolled back on failure,
    so a failed run still consumes the interval (crash-loop protection).
    """
    try:
        logger.info("=" * 60)
        logger.info("MODEL RETRAINING STARTED")
        logger.info("=" * 60)

        # Run training script in a worker thread so the blocking subprocess does
        # not stall the async event loop (MLflow tracking always enabled).
        result = await asyncio.to_thread(
            subprocess.run,
            ["python", "train.py"],
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
            cwd=os.getcwd(),
        )

        if result.returncode == 0:
            logger.info("Model retraining completed successfully")
            logger.info("New model saved but NOT deployed")
            logger.info("→ Review metrics in diagnostics/ directory")
            logger.info("→ Run 'make restart' to deploy new model")
        else:
            logger.error(f"Model retraining failed (exit {result.returncode}): {result.stderr}")

    except subprocess.TimeoutExpired:
        logger.error("Model retraining timed out after 10 minutes")
    except Exception as e:
        logger.error(f"Model retraining error: {e}", exc_info=True)


def verify_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Optional[str]:
    """
    Verify bearer token for authentication.

    If service_token is not set, authentication is disabled (returns None).
    If service_token is set, requires valid token or raises 401.

    Args:
        credentials: HTTP bearer token from Authorization header

    Returns:
        Token string if authenticated, None if auth disabled

    Raises:
        HTTPException: 401 if auth required but missing/invalid

    Security Notes:
        - Tokens are case-sensitive
        - Use environment variables or secrets manager for SERVICE_TOKEN
        - Never log the actual token value
        - In production, use rotating tokens and HTTPS
    """
    # If no token configured, allow all requests (auth disabled)
    # Check for both None and empty string (empty string means auth disabled)
    if not service_config.service_token:
        logger.debug("Authentication bypassed (service_token not set)")
        return None

    # If token configured, require it
    if credentials is None:
        logger.warning("Authentication required but no token provided")
        prediction_error_count.labels(error_type="unauthorized").inc()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Provide a valid Bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Verify token matches using constant-time comparison (prevents timing attacks)
    if not secrets.compare_digest(credentials.credentials, service_config.service_token):
        logger.warning(
            "Invalid authentication token provided " "(token value not logged for security)"
        )
        prediction_error_count.labels(error_type="unauthorized").inc()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    logger.debug("Authentication successful")
    return credentials.credentials


# Dependency injection for model cache (multi-worker safety)
def get_model_cache(request: Request) -> ModelCache:
    """
    FastAPI dependency that provides the model cache from app.state.

    This enables safe access to the cached model in multi-worker deployments.
    Each endpoint that needs the model should use this dependency.

    Returns type-safe ModelCache instead of raw dict.

    Args:
        request: FastAPI request object (provides access to app.state)

    Returns:
        Type-safe ModelCache with keys: model, version, features, threshold

    Usage:
        @app.post("/predict")
        async def predict_endpoint(
            request: PredictionRequest,
            model_cache: ModelCache = Depends(get_model_cache)
        ):
            prediction = predict(request, model_cache)
            ...
    """
    return request.app.state.model_cache


# Lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifespan events.

    Loads model on startup, initializes metrics, and cleans up on shutdown.
    Model now stored in app.state for multi-worker safety.
    """
    # Startup: Load model and initialize metrics
    logger.info("=" * 60)
    logger.info("CHURN PREDICTION SERVICE STARTING UP")
    logger.info("=" * 60)

    # Initialize model cache in app.state (multi-worker safety)
    # Type-safe cache structure using ModelCache TypedDict
    app.state.model_cache: ModelCache = {
        "model": None,
        "version": None,
        "features": None,
        "threshold": 0.5,
        "metadata": None,
    }

    # Initialize service metrics
    init_service_metrics(version="1.0.0")
    logger.info("Prometheus metrics initialized")

    try:
        # Load model into app.state cache
        model, version, expected_features, threshold = get_model_from_cache(app.state.model_cache)
        logger.info(f"Model loaded: version {version}")
        logger.info(f"Expected features: {len(expected_features)} features")
        logger.info(f"Tuned threshold: {threshold:.4f}")
        logger.debug(f"Feature list: {expected_features}")

        # Set model metrics (extract ROC AUC and model age from the metadata
        # cached with the model, so metrics describe the model actually serving)
        try:
            metadata = get_metadata_from_cache(app.state.model_cache)
            if metadata:
                roc_auc = metadata.get("validation_metrics", {}).get("roc_auc", 0.0)
                set_model_metrics(version, metadata.get("timestamp", ""), roc_auc)
                logger.debug(f"Model metrics set: ROC AUC={roc_auc:.4f}")

                # Set model age metric
                if "timestamp" in metadata:
                    model_timestamp = datetime.fromisoformat(metadata["timestamp"])
                    age_days = (datetime.now() - model_timestamp).days
                    model_age_days.set(age_days)
                    logger.debug(f"Model age: {age_days} days")
        except Exception as e:
            logger.warning(f"Could not set model metrics: {e}")

        logger.info("Model loaded successfully")
    except Exception as e:
        logger.critical(f"Failed to load model: {e}", exc_info=True)
        # Don't fail startup - health check will report unhealthy

    logger.info("=" * 60)
    logger.info("SERVICE READY TO ACCEPT REQUESTS")
    logger.info("Metrics available at /metrics")
    logger.info("=" * 60)

    yield

    # Shutdown: Cleanup
    logger.info("Shutting down service...")
    reset_model_cache(app.state.model_cache)
    logger.info("Service stopped")


# Create FastAPI app
app = FastAPI(
    title="Churn Prediction API",
    description="REST API for predicting customer churn using a trained Random Forest model",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Initialize rate limiter
# Uses IP address as the key for rate limiting (per-client)
limiter = Limiter(key_func=get_remote_address, default_limits=["200/minute"])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

if RATE_LIMIT_ENABLED:
    logger.info("Rate limiting enabled")
    logger.info(f"  /predict: {RATE_LIMIT_PREDICT}")
    logger.info(f"  /drift: {RATE_LIMIT_DRIFT}")
else:
    logger.info("Rate limiting disabled")

if REQUEST_TIMEOUT_ENABLED:
    logger.info(f"Request timeout protection enabled: {REQUEST_TIMEOUT_SECONDS}s")
else:
    logger.info("Request timeout protection disabled")

# Mount Prometheus metrics endpoint
# This exposes metrics at /metrics for Prometheus to scrape
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


# ==============================================================================
# MIDDLEWARE (Request tracing)
# ==============================================================================


# Middleware 1: Request ID generation and tracing
@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """
    Generate unique request ID for tracing.

    This middleware:
    1. Generates a UUID for each request (or uses X-Request-ID header if provided)
    2. Stores it in request.state for access by endpoints
    3. Sets it in logging context for automatic log tagging
    4. Adds X-Request-ID header to response

    Benefits:
    - Easy to grep logs for a specific request
    - Can trace a request through distributed systems
    - Helps debug production issues

    Example log output:
        2025-10-29 08:00:15 - INFO - [abc-123-def-456] - Prediction request received
        2025-10-29 08:00:15 - INFO - [abc-123-def-456] - Model inference complete
    """
    # Check if client provided request ID, otherwise generate one
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

    # Store in request state for access by endpoints
    request.state.request_id = request_id

    # Set in logging context (thread-safe for async via ContextVar)
    token = request_id_context.set(request_id)

    try:
        # Process the request
        response = await call_next(request)

        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id

        return response
    finally:
        # Reset context after request completes
        request_id_context.reset(token)


# Middleware 2: Request validation (size limits)
class LimitRequestSizeMiddleware:
    """
    Pure-ASGI middleware enforcing the maximum request body size.

    The previous BaseHTTPMiddleware only inspected the Content-Length header, so
    a chunked request (no Content-Length) or a lying header bypassed the limit
    entirely. This middleware enforces the real size in two layers:

    1. If Content-Length is present and already over the limit, reject with 413
       immediately (cheap, no body read).
    2. Otherwise count actual body bytes as they stream in and reject once the
       limit is exceeded, buffering at most ``max_bytes`` — so the documented
       "prevents memory exhaustion" guarantee holds even without Content-Length.

    Registered outermost so it bounds memory before any body-buffering middleware.
    """

    def __init__(self, app, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Layer 1: cheap Content-Length rejection (when the header is honest)
        headers = dict(scope.get("headers") or [])
        content_length = headers.get(b"content-length")
        if content_length is not None:
            try:
                if int(content_length) > self.max_bytes:
                    await self._reject(scope, receive, send)
                    return
            except ValueError:
                pass  # malformed header -> fall through to byte counting

        # Layer 2: bound and count actual streamed body bytes
        total = 0
        buffered: list[dict] = []
        more_body = True
        while more_body:
            message = await receive()
            if message["type"] != "http.request":
                buffered.append(message)  # e.g. http.disconnect
                break
            total += len(message.get("body", b""))
            if total > self.max_bytes:
                await self._reject(scope, receive, send)
                return
            buffered.append(message)
            more_body = message.get("more_body", False)

        async def replay() -> dict:
            if buffered:
                return buffered.pop(0)
            return await receive()

        await self.app(scope, replay, send)

    async def _reject(self, scope, receive, send) -> None:
        prediction_error_count.labels(error_type="payload_too_large").inc()
        max_mb = self.max_bytes / (1024 * 1024)
        logger.warning(f"Request rejected: body exceeds maximum {max_mb:.0f}MB")
        response = JSONResponse(
            status_code=413,
            content={
                "error": "Payload Too Large",
                "detail": f"Request body exceeds maximum {max_mb:.0f}MB",
                "max_size_mb": max_mb,
            },
        )
        await response(scope, receive, send)


# Middleware 3: Request timeout protection (Step 10)
@app.middleware("http")
async def timeout_middleware(request: Request, call_next):
    """
    Protect against long-running requests that could block workers.

    This middleware:
    1. Sets a timeout for request processing
    2. Cancels requests that exceed the timeout
    3. Logs timeout events for monitoring
    4. Tracks timeouts in Prometheus metrics

    Benefits:
    - Prevents worker starvation from slow requests
    - Protects against DoS via expensive operations
    - Provides visibility into timeout patterns

    Configuration:
    - REQUEST_TIMEOUT_ENABLED: Enable/disable timeout protection (default: true)
    - REQUEST_TIMEOUT_SECONDS: Timeout duration in seconds (default: 30)

    Note: Health check endpoints (/health, /healthz, /readyz) are exempt from timeout.
    """
    # Skip timeout for health checks and metrics (they should always be fast)
    exempt_paths = {"/health", "/healthz", "/readyz", "/metrics", "/"}
    if request.url.path in exempt_paths or not REQUEST_TIMEOUT_ENABLED:
        return await call_next(request)

    # Apply timeout
    try:
        # Use asyncio.wait_for to enforce timeout
        response = await asyncio.wait_for(call_next(request), timeout=REQUEST_TIMEOUT_SECONDS)
        return response

    except asyncio.TimeoutError:
        # Request exceeded timeout
        endpoint = request.url.path
        logger.error(
            f"Request timeout after {REQUEST_TIMEOUT_SECONDS}s: " f"{request.method} {endpoint}"
        )

        # Track timeout in metrics
        request_timeout_count.labels(endpoint=endpoint).inc()

        # Return 504 Gateway Timeout
        return JSONResponse(
            status_code=504,
            content={
                "error": "Gateway Timeout",
                "detail": f"Request processing exceeded {REQUEST_TIMEOUT_SECONDS} seconds timeout",
                "timeout_seconds": REQUEST_TIMEOUT_SECONDS,
                "suggestion": "Try reducing request size or complexity",
            },
        )


# Middleware 4: Metrics tracking (request duration)
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """
    Automatically track request duration for all endpoints.

    This middleware:
    1. Records start time
    2. Processes the request
    3. Calculates duration
    4. Updates Prometheus histogram
    5. Logs slow requests (>1s)

    Note: Skips /metrics endpoint to avoid recursion.
    """
    # Skip /metrics endpoint to avoid infinite recursion
    if request.url.path == "/metrics":
        return await call_next(request)

    # Start timer
    start_time = time.time()
    endpoint = request.url.path

    # Process request
    status_code = 500  # Default to error if something goes wrong
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    except Exception as e:
        logger.error(f"Unhandled error in request: {e}", exc_info=True)
        raise
    finally:
        # Record duration
        duration = time.time() - start_time
        request_duration.labels(endpoint=endpoint).observe(duration)

        # Count request
        status = "success" if 200 <= status_code < 400 else "error"
        request_count.labels(endpoint=endpoint, status=status).inc()

        # Log slow requests
        if duration > 1.0:
            logger.warning(f"Slow request: {endpoint} took {duration:.2f}s (status={status_code})")


# Registered last so it is the OUTERMOST middleware: it bounds request body size
# before any body-buffering middleware runs (enforces the size limit for chunked
# bodies with no Content-Length, which the header check alone cannot).
app.add_middleware(
    LimitRequestSizeMiddleware,
    max_bytes=service_config.max_request_size_mb * 1024 * 1024,
)


@app.get("/", tags=["Info"])
async def root():
    """
    Root endpoint with API information.

    Returns:
        API metadata and available endpoints
    """
    logger.debug("Root endpoint accessed")
    return {
        "name": "Churn Prediction API",
        "version": "1.0.0",
        "description": "Predict customer churn probability",
        "endpoints": {
            "health": "GET /health - Health check",
            "predict": "POST /predict - Make prediction",
            "docs": "GET /docs - Swagger UI documentation",
            "redoc": "GET /redoc - ReDoc documentation",
        },
    }


@app.get("/health", tags=["Health"])
async def health(model_cache: ModelCache = Depends(get_model_cache)):
    """
    Comprehensive health check endpoint for orchestrators.

    Returns detailed status including:
    - Overall health status (healthy/degraded/unhealthy)
    - Readiness for serving traffic
    - Uptime
    - Model information (version, training metrics)
    - Individual component checks

    HTTP Status Codes:
        200 OK: Service healthy and ready
        503 Service Unavailable: Model not loaded, cannot serve traffic

    This endpoint is suitable for:
    - Docker/Kubernetes health checks
    - Load balancer health probes
    - Monitoring systems
    - Manual health verification
    """
    try:
        # Get model and metadata from cache
        model, version, expected_features, threshold = get_model_from_cache(model_cache)
        model_loaded = model is not None

        # Calculate uptime using public timestamp (not Prometheus private API)
        current_time = time.time()
        uptime_seconds = current_time - SERVICE_START_TIMESTAMP

        # Metadata cached with the model — describes the model actually serving
        metadata = model_cache.get("metadata")

        # Determine overall status
        if not model_loaded:
            health_status = "unhealthy"
            ready = False
        elif metadata is None:
            health_status = "degraded"  # Model loaded but no metadata
            ready = True
        else:
            health_status = "healthy"
            ready = True

        # Build response
        health_info = {
            "status": health_status,
            "ready": ready,
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": int(uptime_seconds),
            "checks": {
                "model_loaded": model_loaded,
                "metadata_available": metadata is not None,
                "expected_features_available": expected_features is not None,
            },
        }

        # Add model info if available
        if model_loaded and version:
            health_info["model"] = {"version": version}
            if metadata:
                health_info["model"].update(
                    {
                        "run_id": metadata.get("run_id", "unknown"),
                        "trained_at": metadata.get("timestamp", "unknown"),
                        "validation_roc_auc": metadata.get("validation_metrics", {}).get(
                            "roc_auc", 0.0
                        ),
                        "n_features": len(expected_features) if expected_features else 0,
                    }
                )

        # Return appropriate HTTP status code
        if health_status == "unhealthy":
            logger.warning("Health check: service unhealthy (model not loaded)")
            return JSONResponse(content=health_info, status_code=503)
        elif health_status == "degraded":
            logger.info("Health check: service degraded but operational")
            return JSONResponse(content=health_info, status_code=200)
        else:
            logger.debug("Health check: service healthy")
            return JSONResponse(content=health_info, status_code=200)

    except Exception as e:
        logger.error(f"Health check failed with exception: {e}", exc_info=True)
        return JSONResponse(
            content={
                "status": "unhealthy",
                "ready": False,
                "timestamp": datetime.now().isoformat(),
                "checks": {
                    "model_loaded": False,
                    "metadata_available": False,
                    "expected_features_available": False,
                },
                "error": "Health check failed",
            },
            status_code=503,
        )


@app.get("/healthz", tags=["Health"])
async def liveness():
    """
    Liveness probe: Is the process alive?

    Returns 200 if the service process is running, regardless of model state.
    This endpoint should NEVER fail unless the process is dead/frozen.

    Usage:
        Kubernetes liveness probe - if this fails, restart the container

    Example:
        livenessProbe:
          httpGet:
            path: /healthz
            port: 8000
          periodSeconds: 10
          failureThreshold: 3
    """
    logger.debug("Liveness check: process alive")
    return {"status": "alive", "timestamp": datetime.now().isoformat()}


@app.get("/readyz", tags=["Health"])
async def readiness(model_cache: ModelCache = Depends(get_model_cache)):
    """
    Readiness probe: Is the service ready to serve traffic?

    Returns:
        200 if model is loaded and service can handle requests
        503 if model is not loaded (don't send traffic yet)

    Usage:
        Kubernetes readiness probe - if this fails, remove from load balancer
        but don't restart

    Example:
        readinessProbe:
          httpGet:
            path: /readyz
            port: 8000
          periodSeconds: 5
          failureThreshold: 2
    """
    try:
        model, version, _, _ = get_model_from_cache(model_cache)
        if model is None:
            logger.warning("Readiness check failed: model not loaded")
            return JSONResponse(
                content={
                    "status": "not_ready",
                    "reason": "model_not_loaded",
                    "timestamp": datetime.now().isoformat(),
                },
                status_code=503,
            )
        logger.debug(f"Readiness check passed: model {version} loaded")
        return {
            "status": "ready",
            "model_version": version,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Readiness check failed: {e}", exc_info=True)
        return JSONResponse(
            content={
                "status": "not_ready",
                "reason": "check_failed",
                "timestamp": datetime.now().isoformat(),
            },
            status_code=503,
        )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
@limiter.limit(RATE_LIMIT_PREDICT if RATE_LIMIT_ENABLED else "999999/minute")
async def predict_churn(
    prediction_request: PredictionRequest,
    request: Request,
    model_cache: ModelCache = Depends(get_model_cache),
    token: Optional[str] = Depends(verify_token),
):
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


# ==============================================================================
# DRIFT DETECTION ENDPOINTS
# ==============================================================================


@app.post("/drift", response_model=DriftAnalysisResponse, tags=["Drift Detection"])
@limiter.limit(RATE_LIMIT_DRIFT if RATE_LIMIT_ENABLED else "999999/minute")
async def analyze_drift_endpoint(
    drift_request: DriftAnalysisRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    model_cache: ModelCache = Depends(get_model_cache),
    _token: Optional[str] = Depends(verify_token),
):
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

    # Use the metadata cached WITH the model. Re-reading the latest metadata from
    # disk here could pair the cached (still-serving) model with a newer retrained
    # model's baseline and threshold — like-for-like drift requires the model,
    # threshold, and reference statistics to come from the same load event.
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
            if AUTO_RETRAIN_ON_DRIFT and check_and_reserve_retraining():
                logger.info("AUTO_RETRAIN_ON_DRIFT enabled - triggering retraining")
                background_tasks.add_task(retrain_model_task)
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
        # policy as the global exception handler below)
        logger.error(f"Drift analysis failed: {e}", exc_info=True)
        prediction_error_count.labels(error_type="drift_analysis_error").inc()
        raise HTTPException(
            status_code=500,
            detail="Drift analysis failed due to an internal error. Please try again later.",
        )


@app.get("/drift/info", tags=["Drift Detection"])
async def drift_info(
    model_cache: ModelCache = Depends(get_model_cache),
    _token: Optional[str] = Depends(verify_token),
):
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
            "auto_retrain_on_drift": AUTO_RETRAIN_ON_DRIFT,
            "min_retrain_interval_hours": MIN_RETRAIN_INTERVAL_HOURS,
        },
    }


@app.post("/retrain", tags=["Model Management"])
async def trigger_retraining(
    background_tasks: BackgroundTasks, _token: Optional[str] = Depends(verify_token)
):
    """
    Trigger model retraining in the background.

    Safeguards:
    - Rate limited to once per MIN_RETRAIN_INTERVAL_HOURS (default 24h)
    - Runs in background (doesn't block API)
    - New model saved but NOT auto-deployed
    - Requires manual review and deployment

    Returns:
        Retraining status and next steps

    Raises:
        429: Retraining triggered too recently (rate limit)
    """
    # Atomically check the rate limit and reserve the slot before scheduling, so
    # concurrent /retrain calls cannot both pass the check and launch duplicate runs.
    if not check_and_reserve_retraining():
        last_retrain = get_last_retrain_time()
        time_since_last = datetime.now() - last_retrain if last_retrain else timedelta(0)
        hours_remaining = MIN_RETRAIN_INTERVAL_HOURS - (time_since_last.total_seconds() / 3600)
        raise HTTPException(
            status_code=429,
            detail=f"Retraining triggered too recently. Minimum {MIN_RETRAIN_INTERVAL_HOURS}h "
            f"between retrains. Try again in {hours_remaining:.1f} hours.",
        )

    # Trigger background retraining (slot already reserved above)
    background_tasks.add_task(retrain_model_task)

    logger.warning("Model retraining triggered via API")

    return {
        "status": "retraining_triggered",
        "message": "Model retraining started in background. Check logs for progress.",
        "note": "New model will be saved but NOT auto-deployed. Review metrics before deploying.",
        "next_steps": [
            "1. Monitor logs: docker compose logs -f api",
            "2. Review metrics: cat diagnostics/evaluation_report_*.txt",
            "3. Deploy if satisfied: make restart",
        ],
        "timestamp": datetime.now().isoformat(),
    }


# ==============================================================================
# Error Handlers
# ==============================================================================
# These handlers ensure we NEVER leak stack traces or internal details to users
# All errors are logged with full details, but users get sanitized messages


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Handle Pydantic validation errors with user-friendly messages.

    Includes request_id in error response for tracing.

    Triggered when:
    - Missing required fields
    - Wrong data types (string instead of int)
    - Invalid values (negative numbers where positive expected)
    - Extra fields not in schema (if configured)

    Returns 422 Unprocessable Entity with helpful error details.
    """
    # Get request ID from middleware
    request_id = getattr(request.state, "request_id", "no-request-id")

    logger.warning(f"Request validation error on {request.url.path}")
    logger.debug(f"Validation errors: {exc.errors()}")

    # Extract user-friendly error messages
    errors = []
    for error in exc.errors():
        # Build field path (e.g., "customers -> 0 -> tenure")
        field_path = " -> ".join(str(x) for x in error["loc"])
        message = error["msg"]
        error_type = error["type"]
        errors.append({"field": field_path, "message": message, "type": error_type})

    prediction_error_count.labels(error_type="validation_error").inc()

    response = JSONResponse(
        status_code=422,  # Unprocessable Entity
        content={
            "error": "Validation Error",
            "detail": "Request does not match expected schema",
            "errors": errors,
            "hint": "Check API documentation at /docs for correct schema",
            "request_id": request_id,  # Include for tracing
        },
    )
    # Add X-Request-ID header
    response.headers["X-Request-ID"] = request_id
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Catch-all for unexpected exceptions.

    Uses request_id from middleware for tracing.

    Security: Logs full details but returns generic message to user.
    NEVER leak stack traces, internal paths, or sensitive data.

    This handler should only trigger for truly unexpected errors.
    Most errors should be caught by specific handlers or try/except blocks.
    """
    # Get request ID from middleware
    request_id = getattr(request.state, "request_id", "no-request-id")

    # Log full details (including stack trace)
    logger.critical(
        f"UNHANDLED EXCEPTION on {request.url.path}: {type(exc).__name__}: {exc}", exc_info=True
    )

    prediction_error_count.labels(error_type="unhandled_exception").inc()

    # Return generic error (don't leak internal details)
    response = JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": "An unexpected error occurred. Please try again later.",
            "request_id": request_id,
            "hint": "If this persists, contact support with the request_id",
        },
    )
    # Add X-Request-ID header
    response.headers["X-Request-ID"] = request_id
    return response


if __name__ == "__main__":
    import uvicorn

    logger.info("=" * 70)
    logger.info("CHURN PREDICTION API")
    logger.info("Drift Detection & Automatic Retraining")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Starting server...")
    logger.info("  • Swagger UI:  http://localhost:8000/docs")
    logger.info("  • ReDoc:       http://localhost:8000/redoc")
    logger.info("  • Health:      http://localhost:8000/health")
    logger.info("  • Liveness:    http://localhost:8000/healthz")
    logger.info("  • Readiness:   http://localhost:8000/readyz")
    logger.info("  • Metrics:     http://localhost:8000/metrics")
    logger.info("  • Drift:       http://localhost:8000/drift")
    logger.info("  • Drift Info:  http://localhost:8000/drift/info")
    logger.info("  • Retrain:     http://localhost:8000/retrain")
    logger.info("")

    uvicorn.run("serve:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
