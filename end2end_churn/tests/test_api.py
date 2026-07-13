"""
Integration tests for FastAPI endpoints.

Tests API contracts, error handling, and endpoint behavior.

Comprehensive Testing
"""

import os
import sys

import pytest
from fastapi.testclient import TestClient

# Disable authentication for tests
# Unset secret env vars BEFORE importing serve module
os.environ.pop("SERVICE_TOKEN_FILE", None)
os.environ.pop("SERVICE_TOKEN", None)

# Add parent directory to path to import serve module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from serve import app


@pytest.fixture
def client():
    """
    Create FastAPI test client.

    Initialize app.state.model_cache for multi-worker safety pattern.
    The TestClient doesn't automatically trigger lifespan events in test mode,
    so we manually initialize the model cache here.

    Uses ModelCache type for type safety.

    Note: Authentication is disabled for tests by unsetting env vars at module import time.
    """
    # Import ModelCache type for type annotation
    from src.api.types import ModelCache

    # Initialize model cache in app.state (same as lifespan does)
    app.state.model_cache: ModelCache = {
        "model": None,
        "version": None,
        "features": None,
        "threshold": 0.5,
        "metadata": None,
    }

    return TestClient(app)


# =============================================================================
# Root Endpoint Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.api
def test_root_endpoint(client):
    """Test root endpoint returns API information."""
    response = client.get("/")

    assert response.status_code == 200
    data = response.json()

    # Check expected fields
    assert "name" in data
    assert "version" in data
    assert "description" in data


# =============================================================================
# Health Check Endpoints
# =============================================================================


@pytest.mark.integration
@pytest.mark.api
def test_health_endpoint_structure(client):
    """Test /health endpoint returns expected structure."""
    response = client.get("/health")

    # Can be 200 (healthy) or 503 (model not loaded)
    assert response.status_code in [200, 503]

    data = response.json()

    # Check required top-level fields
    assert "status" in data
    assert "ready" in data
    assert "timestamp" in data

    # Check nested checks object
    assert "checks" in data
    assert "model_loaded" in data["checks"]


@pytest.mark.integration
@pytest.mark.api
def test_healthz_liveness_probe(client):
    """Test /healthz liveness probe always returns 200."""
    response = client.get("/healthz")

    # Liveness should ALWAYS return 200 if process is alive
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "alive"


@pytest.mark.integration
@pytest.mark.api
def test_readyz_readiness_probe(client):
    """Test /readyz readiness probe checks if service is ready."""
    response = client.get("/readyz")

    # Can be 200 (ready) or 503 (not ready)
    assert response.status_code in [200, 503]

    data = response.json()
    # Response has 'status' field (not 'ready')
    assert "status" in data
    assert data["status"] in ["ready", "not_ready"]


# =============================================================================
# Prediction Endpoint Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.api
def test_predict_endpoint_valid_request(client, sample_prediction_request):
    """Test /predict endpoint with valid single-record request."""
    response = client.post("/predict", json=sample_prediction_request)

    # If model not loaded, skip test
    if response.status_code == 503:
        pytest.skip("Model not loaded in test environment")

    assert response.status_code == 200
    data = response.json()

    # Check response structure (single prediction, not batch)
    assert "churn_prediction" in data
    assert "churn_probability" in data
    assert "risk_level" in data
    assert "model_version" in data
    assert "request_id" in data

    # Check prediction values
    assert data["churn_prediction"] in ["Yes", "No"]
    assert 0 <= data["churn_probability"] <= 1
    assert data["risk_level"] in ["Low", "Medium", "High"]


@pytest.mark.integration
@pytest.mark.api
def test_predict_endpoint_single_customer(client, sample_features):
    """Test /predict endpoint with single customer (same as valid request test)."""
    from tests.conftest import convert_numpy_types

    # Convert pandas row to dict with native Python types
    customer_dict = sample_features.iloc[0].to_dict()
    single_customer = convert_numpy_types(customer_dict)

    response = client.post("/predict", json=single_customer)

    if response.status_code == 503:
        pytest.skip("Model not loaded")

    assert response.status_code == 200
    data = response.json()

    # Verify single response structure
    assert "churn_prediction" in data
    assert "churn_probability" in data
    assert "request_id" in data


@pytest.mark.integration
@pytest.mark.api
def test_predict_endpoint_missing_required_fields(client):
    """Test /predict endpoint rejects request with missing required fields."""
    # Empty dict missing all required fields
    request_data = {}

    response = client.post("/predict", json=request_data)

    # Should return validation error (422)
    assert response.status_code == 422


@pytest.mark.integration
@pytest.mark.api
def test_predict_endpoint_invalid_field_values(client):
    """Test /predict endpoint rejects invalid field values."""
    # Invalid gender value (should be 'Male' or 'Female')
    request_data = {
        "gender": "Invalid",
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

    response = client.post("/predict", json=request_data)

    # Should return validation error
    assert response.status_code == 422


@pytest.mark.integration
@pytest.mark.api
def test_predict_endpoint_invalid_json(client):
    """Test /predict endpoint handles invalid JSON."""
    response = client.post(
        "/predict", data="invalid json{{{", headers={"Content-Type": "application/json"}
    )

    # Should return 422 (Unprocessable Entity)
    assert response.status_code == 422


@pytest.mark.integration
@pytest.mark.api
def test_predict_endpoint_with_all_features(client, sample_features):
    """Test /predict endpoint with all customer features present."""
    from tests.conftest import convert_numpy_types

    # Use a complete customer record with all features
    # Convert pandas row to dict with native Python types
    customer_dict = sample_features.iloc[10].to_dict()
    customer = convert_numpy_types(customer_dict)

    response = client.post("/predict", json=customer)

    if response.status_code == 503:
        pytest.skip("Model not loaded")

    assert response.status_code == 200
    data = response.json()

    # Verify complete response structure
    assert "churn_prediction" in data
    assert "churn_probability" in data
    assert "risk_level" in data
    assert "model_version" in data


# =============================================================================
# Drift Detection Endpoints
# =============================================================================


@pytest.mark.integration
@pytest.mark.api
def test_drift_endpoint_valid_request(client, sample_features):
    """Test /drift endpoint with valid data."""
    customers = sample_features.head(50).to_dict("records")
    request_data = {"customers": customers}

    response = client.post("/drift", json=request_data)

    # Can be 503 if model not loaded or drift detection not available
    if response.status_code == 503:
        pytest.skip("Drift detection not available")

    assert response.status_code == 200
    data = response.json()

    # Check response structure matches DriftAnalysisResponse schema
    assert "overall_drift_detected" in data
    assert isinstance(data["overall_drift_detected"], bool)
    assert "n_features_drifted" in data
    assert isinstance(data["n_features_drifted"], int)
    assert "drifted_features" in data
    assert isinstance(data["drifted_features"], list)
    assert "prediction_drift_detected" in data
    assert isinstance(data["prediction_drift_detected"], bool)
    assert "detailed_report" in data


@pytest.mark.integration
@pytest.mark.api
def test_drift_endpoint_empty_batch(client):
    """Test /drift endpoint rejects empty batch."""
    request_data = {"customers": []}

    response = client.post("/drift", json=request_data)

    # Should return 400 (Bad Request) - business logic error
    assert response.status_code in [400, 422]


@pytest.mark.integration
@pytest.mark.api
def test_drift_info_endpoint(client):
    """Test /drift/info endpoint returns configuration."""
    response = client.get("/drift/info")

    # Can be 503 if model not loaded
    if response.status_code == 503:
        pytest.skip("Drift info not available")

    assert response.status_code == 200
    data = response.json()

    # Check expected fields
    assert "baseline" in data
    assert "thresholds" in data
    assert "model_info" in data


# =============================================================================
# Retrain Endpoint Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.api
@pytest.mark.slow
def test_retrain_endpoint_exists(client):
    """Test /retrain endpoint exists and accepts POST."""
    response = client.post("/retrain")

    # Should return 200 (success) or 429 (rate limited) or 503 (not available)
    # Not 404 (endpoint should exist)
    assert response.status_code != 404


# =============================================================================
# Metrics Endpoint Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.api
def test_metrics_endpoint(client):
    """Test /metrics endpoint returns Prometheus format."""
    response = client.get("/metrics")

    assert response.status_code == 200

    # Check for Prometheus format (plain text with metrics)
    assert response.headers["content-type"] == "text/plain; version=0.0.4; charset=utf-8"

    # Check for expected metric names
    metrics_text = response.text
    assert (
        "churn_prediction_requests_total" in metrics_text or "http_requests_total" in metrics_text
    )


# =============================================================================
# Error Handling Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.api
def test_404_not_found(client):
    """Test non-existent endpoint returns 404."""
    response = client.get("/nonexistent-endpoint")

    assert response.status_code == 404


@pytest.mark.integration
@pytest.mark.api
def test_method_not_allowed(client):
    """Test wrong HTTP method returns 405."""
    # /predict only accepts POST
    response = client.get("/predict")

    assert response.status_code == 405


# =============================================================================
# Documentation Endpoints
# =============================================================================


@pytest.mark.integration
@pytest.mark.api
def test_docs_endpoint_exists(client):
    """Test Swagger UI documentation is available."""
    response = client.get("/docs")

    assert response.status_code == 200


@pytest.mark.integration
@pytest.mark.api
def test_openapi_json_endpoint(client):
    """Test OpenAPI schema is available."""
    response = client.get("/openapi.json")

    assert response.status_code == 200

    # Should return valid JSON
    data = response.json()
    assert "openapi" in data
    assert "info" in data
    assert "paths" in data
