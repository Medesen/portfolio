"""
Bearer-token authentication for the churn prediction service.

Authentication is optional: when no service token is configured, all requests
are allowed. When a token is set (via SERVICE_TOKEN or SERVICE_TOKEN_FILE),
requests must present it as a Bearer token and it is compared in constant
time to prevent timing attacks.
"""

import secrets
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ..utils.logger import get_logger
from ..utils.prometheus_metrics import prediction_error_count
from .settings import service_config

logger = get_logger("churn_api")

security = HTTPBearer(auto_error=False)  # Don't auto-fail if missing


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
