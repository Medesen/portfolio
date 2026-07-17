"""Root endpoint: API name, version, and endpoint directory."""

from typing import Any

from fastapi import APIRouter

from ...utils.logger import get_logger

logger = get_logger("churn_api")

router = APIRouter()


@router.get("/", tags=["Info"])
async def root() -> dict[str, Any]:
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
