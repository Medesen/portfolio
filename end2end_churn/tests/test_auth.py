"""
Unit tests for bearer-token authentication.

The auth dependency has three behaviors worth pinning down: requests are
rejected with 401 when a token is configured but missing or wrong, and a
correct token passes through to the endpoint (surfacing as the endpoint's
own response, not an auth error). Auth-disabled behavior is exercised by
the rest of the API test suite, which runs without a configured token.
"""

import os
import sys

import pytest
from fastapi.testclient import TestClient

os.environ.pop("SERVICE_TOKEN_FILE", None)
os.environ.pop("SERVICE_TOKEN", None)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from serve import app  # noqa: E402
from src.api.settings import service_config  # noqa: E402


@pytest.fixture
def auth_client(monkeypatch):
    """Client against an app with a configured service token.

    The cache holds a sentinel (non-None) model so endpoints never lazy-load
    from disk: with metadata absent, /drift/info answers a deterministic 503
    whether or not a trained model exists in the environment (CI's unit job
    has none; a developer machine usually does).
    """
    monkeypatch.setattr(service_config, "service_token", "correct-token")
    app.state.model_cache = {
        "model": object(),
        "version": "test-sentinel",
        "features": [],
        "threshold": 0.5,
        "metadata": None,
    }
    return TestClient(app)


@pytest.mark.unit
def test_missing_token_rejected(auth_client):
    response = auth_client.get("/drift/info")
    assert response.status_code == 401
    assert response.json()["detail"] == "Authentication required. Provide a valid Bearer token."
    assert response.headers["WWW-Authenticate"] == "Bearer"


@pytest.mark.unit
def test_wrong_token_rejected(auth_client):
    response = auth_client.get("/drift/info", headers={"Authorization": "Bearer wrong-token"})
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid authentication token"


@pytest.mark.unit
def test_correct_token_reaches_endpoint(auth_client):
    """A valid token must clear auth: the 503 is the endpoint's own
    no-metadata response, proving the request got past verify_token."""
    response = auth_client.get("/drift/info", headers={"Authorization": "Bearer correct-token"})
    assert response.status_code == 503
    assert "metadata" in response.json()["detail"]


@pytest.mark.unit
def test_health_endpoints_do_not_require_auth(auth_client):
    """Liveness must stay probe-safe even with auth enabled."""
    assert auth_client.get("/healthz").status_code == 200
