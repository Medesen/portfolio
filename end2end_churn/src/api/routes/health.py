"""
Health endpoints: /health (detailed), /healthz (liveness), /readyz (readiness).

The three probes serve different orchestration needs: /healthz answers "is the
process alive" and never depends on model state; /readyz answers "can this
instance serve traffic" (503 until the model is loaded); /health reports the
full component picture for monitoring and humans.
"""

import time
from datetime import datetime
from typing import Any, Union

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from ...utils.logger import get_logger
from ..dependencies import get_model_cache
from ..service import get_model_from_cache
from ..settings import SERVICE_START_TIMESTAMP
from ..types import ModelCache

logger = get_logger("churn_api")

router = APIRouter()


@router.get("/health", tags=["Health"])
async def health(model_cache: ModelCache = Depends(get_model_cache)) -> JSONResponse:
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
        health_info: dict[str, Any] = {
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


@router.get("/healthz", tags=["Health"])
async def liveness() -> dict[str, str]:
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


@router.get("/readyz", tags=["Health"], response_model=None)
async def readiness(
    model_cache: ModelCache = Depends(get_model_cache),
) -> Union[JSONResponse, dict[str, Any]]:
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
