"""
Prometheus metrics for the churn prediction service.

This module defines all metrics exposed at the /metrics endpoint:
- Request counters (total, by status)
- Prediction counters (by predicted class)
- Latency histograms (request duration)
- Error counters (by error type)
- Schema mismatch counters
- Model metadata (version, timestamp)
- Service metadata (start time, uptime)
- Drift detection metrics

Metrics follow Prometheus naming conventions:
- Counter names end with _total
- Use underscores, not camelCase
- Units in the metric name (e.g., _seconds, _bytes)

For more on Prometheus best practices:
https://prometheus.io/docs/practices/naming/
"""

import time

from prometheus_client import Counter, Gauge, Histogram, Info

# =============================================================================
# Service Information
# =============================================================================

service_info = Info("churn_service", "Churn prediction service metadata")

service_start_time = Gauge(
    "churn_service_start_time_seconds", "Unix timestamp when the service started"
)

# =============================================================================
# Request Metrics (Four Golden Signals: Traffic)
# =============================================================================

request_count = Counter(
    "churn_prediction_requests_total",
    "Total number of prediction requests",
    ["endpoint", "status"],  # Labels: endpoint=/predict, status=success/error
)

# =============================================================================
# Prediction Metrics (ML-specific)
# =============================================================================

prediction_count = Counter(
    "churn_predictions_total",
    "Total number of individual predictions made",
    ["predicted_class"],  # Labels: 0 (no churn), 1 (churn)
)

prediction_probability_histogram = Histogram(
    "churn_prediction_probability",
    "Distribution of predicted churn probabilities",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

# =============================================================================
# Latency Metrics (Four Golden Signals: Latency)
# =============================================================================
# Histogram buckets optimized for ML inference latency percentiles (p50, p95, p99)
# Buckets are carefully chosen to provide accurate percentile calculations:
# - Fine granularity in 0-100ms range (most predictions)
# - Coarser granularity for slower requests
# - Covers range from 5ms to 10s

request_duration = Histogram(
    "churn_prediction_request_duration_seconds",
    "End-to-end request processing time including all middleware",
    ["endpoint"],  # Label: endpoint=/predict, /health, /drift, etc.
    # Buckets: 5ms, 10ms, 25ms, 50ms, 75ms, 100ms, 250ms, 500ms, 1s, 2.5s, 5s, 10s
    buckets=[0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

# =============================================================================
# Error Metrics (Four Golden Signals: Errors)
# =============================================================================

prediction_error_count = Counter(
    "churn_prediction_errors_total",
    "Total number of prediction errors",
    ["error_type"],  # Labels: model_not_loaded, validation_error, prediction_error
)

schema_mismatch_count = Counter(
    "churn_schema_mismatches_total",
    "Total number of schema mismatches detected",
    ["mismatch_type"],  # Labels: missing_columns, extra_columns, reordered_columns
)

data_quality_issue_count = Counter(
    "churn_data_quality_issues_total",
    "Total number of data quality issues detected by Pandera",
    ["issue_type"],  # Labels: invalid_range, invalid_category, missing_value
)

# =============================================================================
# Model Metrics
# =============================================================================

model_info = Gauge(
    "churn_model_info",
    "Information about the loaded model (value=1 when model loaded)",
    ["run_id", "timestamp", "roc_auc"],
)

model_prediction_time = Histogram(
    "churn_model_prediction_duration_seconds",
    "Time spent in model inference (predict_proba call)",
    # Optimized for single-record inference latency
    # Buckets: 1ms, 5ms, 10ms, 25ms, 50ms, 100ms, 250ms, 500ms, 1s
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

# Separate metric for preprocessing time
preprocessing_time = Histogram(
    "churn_preprocessing_duration_seconds",
    "Time spent in data preprocessing (alignment, validation)",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
)

# =============================================================================
# Drift Detection Metrics
# =============================================================================

drift_detected_count = Counter(
    "churn_drift_detected_total",
    "Total number of drift detections",
    ["drift_type"],  # Labels: numeric, categorical, prediction
)

features_drifted_gauge = Gauge(
    "churn_features_drifted", "Number of features currently showing drift"
)

model_age_days = Gauge("churn_model_age_days", "Days since model was trained")

# =============================================================================
# Helper Functions
# =============================================================================


def init_service_metrics(version: str = "1.0.0"):
    """
    Initialize service-level metrics on startup.

    Args:
        version: Service version string
    """
    import sys

    service_start_time.set(time.time())
    service_info.info({"version": version, "python_version": sys.version.split()[0]})


def set_model_metrics(run_id: str, timestamp: str, roc_auc: float):
    """
    Set model metadata metrics.

    Args:
        run_id: Model training run ID
        timestamp: Model training timestamp
        roc_auc: Model validation ROC AUC score
    """
    model_info.labels(run_id=run_id, timestamp=timestamp, roc_auc=f"{roc_auc:.4f}").set(1)
