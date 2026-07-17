"""
Tests for application startup/shutdown (lifespan).

The lifespan contract: startup loads the model into app.state and never
raises — a missing model leaves the service alive (liveness passes) but not
ready (readiness 503), so orchestrators keep the pod and hold traffic.
"""

import os
import sys

import pytest
from fastapi.testclient import TestClient

os.environ.pop("SERVICE_TOKEN_FILE", None)
os.environ.pop("SERVICE_TOKEN", None)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from serve import app  # noqa: E402


@pytest.mark.integration
def test_lifespan_loads_model_and_reports_ready():
    """With a trained model on disk, startup loads it and /readyz goes green."""
    with TestClient(app) as client:
        response = client.get("/readyz")
        assert response.status_code == 200
        assert response.json()["status"] == "ready"

        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["checks"]["model_loaded"] is True


@pytest.mark.unit
def test_lifespan_survives_missing_model(monkeypatch):
    """A missing model must not crash startup: alive but not ready."""
    monkeypatch.setenv("MODEL_SOURCE", "local")
    monkeypatch.setenv("MODEL_PATH", "models/does_not_exist.joblib")

    with TestClient(app) as client:
        assert client.get("/healthz").status_code == 200

        readyz = client.get("/readyz")
        assert readyz.status_code == 503
        # "check_failed": the readiness probe's own lazy load attempt raises
        # on the missing file; "model_not_loaded" would mean a quiet cache miss
        assert readyz.json()["reason"] in ("model_not_loaded", "check_failed")
