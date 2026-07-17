"""
Runtime configuration for the churn prediction service.

Everything env-derived is read once here, at import time: the service and
drift configs, rate-limit and request-timeout knobs, and the shared slowapi
rate limiter. Importing this module also bootstraps the "churn_api" logger,
so every other module in the package can rely on `get_logger("churn_api")`
returning a configured logger.
"""

import os
import time

from slowapi import Limiter
from slowapi.util import get_remote_address

from ..config import DriftConfig, ServiceConfig
from ..utils.logger import setup_logger

# Service config supports secure secret loading via SERVICE_TOKEN_FILE
service_config = ServiceConfig.from_env()

# Rate limiting (per-client, keyed by IP address)
RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
RATE_LIMIT_PREDICT = os.getenv("RATE_LIMIT_PREDICT", "100/minute")
RATE_LIMIT_DRIFT = os.getenv("RATE_LIMIT_DRIFT", "20/minute")

# Request timeout protection
REQUEST_TIMEOUT_SECONDS = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "30"))
REQUEST_TIMEOUT_ENABLED = os.getenv("REQUEST_TIMEOUT_ENABLED", "true").lower() == "true"

drift_config = DriftConfig(
    numeric_threshold=float(os.getenv("DRIFT_THRESHOLD_NUMERIC", "0.2")),
    categorical_threshold=float(os.getenv("DRIFT_THRESHOLD_CATEGORICAL", "0.25")),
    prediction_threshold=float(os.getenv("DRIFT_THRESHOLD_PREDICTION", "0.1")),
    min_sample_size=int(os.getenv("DRIFT_MIN_SAMPLE_SIZE", "100")),
    max_drift_batch_size=int(os.getenv("MAX_DRIFT_BATCH_SIZE", "1000")),
)

# Service start timestamp for uptime calculation (avoids Prometheus private API)
SERVICE_START_TIMESTAMP = time.time()

# Shared rate limiter. Created here (not in the app factory) because route
# modules need it at import time for their @limiter.limit decorators.
limiter = Limiter(key_func=get_remote_address, default_limits=["200/minute"])

logger = setup_logger(
    name="churn_api", log_level=service_config.log_level, log_file="logs/churn_service.log"
)
