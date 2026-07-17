"""
Uvicorn entry point for the churn prediction service.

The application itself lives in ``src/api``: `create_app()` in
``src/api/app.py`` assembles routes, middleware, auth, drift detection, and
retraining coordination. This shim exists so deployment surfaces keep a
stable target — ``uvicorn serve:app`` works unchanged in Docker, Kubernetes,
and the Makefile.

Usage:
    # Development (with auto-reload)
    uvicorn serve:app --reload --host 0.0.0.0 --port 8000

    # Production
    uvicorn serve:app --host 0.0.0.0 --port 8000 --workers 4

    # With authentication
    export SERVICE_TOKEN="your-secret-token"
    uvicorn serve:app --host 0.0.0.0 --port 8000
"""

from src.api.app import app
from src.utils.logger import get_logger

__all__ = ["app"]

if __name__ == "__main__":
    import uvicorn

    logger = get_logger("churn_api")
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
