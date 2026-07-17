"""
Application factory and lifespan for the churn prediction service.

`create_app()` assembles the FastAPI application from the package's parts:
lifespan (model loading into app.state), rate limiting, the Prometheus
/metrics mount, the four middleware layers, the route modules, and the
sanitizing error handlers. The module-level `app` is what uvicorn serves
(via the `serve.py` shim as `serve:app`).
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI
from prometheus_client import make_asgi_app
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from ..utils.logger import get_logger
from ..utils.prometheus_metrics import init_service_metrics, model_age_days, set_model_metrics
from . import retraining
from .errors import register_error_handlers
from .middleware import (
    LimitRequestSizeMiddleware,
    metrics_middleware,
    request_id_middleware,
    timeout_middleware,
)
from .routes import drift, health, info, predict, retrain
from .service import get_metadata_from_cache, get_model_from_cache, reset_model_cache
from .settings import (
    RATE_LIMIT_DRIFT,
    RATE_LIMIT_ENABLED,
    RATE_LIMIT_PREDICT,
    REQUEST_TIMEOUT_ENABLED,
    REQUEST_TIMEOUT_SECONDS,
    drift_config,
    limiter,
    service_config,
)
from .types import ModelCache

logger = get_logger("churn_api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Manage application lifespan events.

    Loads model on startup, initializes metrics, and cleans up on shutdown.
    The model lives in app.state so every worker process gets its own copy.
    """
    # Startup: Load model and initialize metrics
    logger.info("=" * 60)
    logger.info("CHURN PREDICTION SERVICE STARTING UP")
    logger.info("=" * 60)

    # Initialize model cache in app.state (multi-worker safety)
    model_cache: ModelCache = {
        "model": None,
        "version": None,
        "features": None,
        "threshold": 0.5,
        "metadata": None,
    }
    app.state.model_cache = model_cache

    # Initialize service metrics
    init_service_metrics(version="1.0.0")
    logger.info("Prometheus metrics initialized")

    try:
        # Load model into app.state cache
        model, version, expected_features, threshold = get_model_from_cache(app.state.model_cache)
        if model is None or version is None:
            raise RuntimeError("Model cache empty after load")
        logger.info(f"Model loaded: version {version}")
        n_features = len(expected_features) if expected_features is not None else 0
        logger.info(f"Expected features: {n_features} features")
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


def create_app() -> FastAPI:
    """Build the FastAPI application with all middleware, routes, and handlers."""
    app = FastAPI(
        title="Churn Prediction API",
        description="REST API for predicting customer churn using a trained Random Forest model",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Rate limiting (per-client IP)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]

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

    # Mount Prometheus metrics endpoint (scraped at /metrics)
    app.mount("/metrics", make_asgi_app())

    # Middleware. Registration order matters: each add wraps the previous, so
    # the LAST one added is OUTERMOST. The size limit is registered last so it
    # bounds request body size before any body-buffering middleware runs
    # (enforcing the limit for chunked bodies with no Content-Length, which the
    # header check alone cannot).
    app.middleware("http")(request_id_middleware)
    app.middleware("http")(timeout_middleware)
    app.middleware("http")(metrics_middleware)
    app.add_middleware(
        LimitRequestSizeMiddleware,
        max_bytes=service_config.max_request_size_mb * 1024 * 1024,
    )

    # Routes
    app.include_router(info.router)
    app.include_router(health.router)
    app.include_router(predict.router)
    app.include_router(drift.router)
    app.include_router(retrain.router)

    # Sanitizing error handlers (log everything, leak nothing)
    register_error_handlers(app)

    return app


logger.info("Configuration loaded:")
logger.info("  • Service config:")
logger.info(f"    - MAX_BATCH_SIZE: {service_config.max_batch_size}")
logger.info(f"    - MAX_REQUEST_SIZE_MB: {service_config.max_request_size_mb}")
logger.info(f"    - Authentication: {'enabled' if service_config.service_token else 'disabled'}")
logger.info("  • Drift config:")
logger.info(f"    - MAX_DRIFT_BATCH_SIZE: {drift_config.max_drift_batch_size}")
logger.info(
    f"    - Thresholds: numeric={drift_config.numeric_threshold}, "
    f"categorical={drift_config.categorical_threshold}, "
    f"prediction={drift_config.prediction_threshold}"
)
logger.info(
    f"  • Auto-retrain on drift: {'enabled' if retraining.AUTO_RETRAIN_ON_DRIFT else 'disabled'}"
)

app = create_app()
