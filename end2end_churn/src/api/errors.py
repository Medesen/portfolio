"""
Exception handlers for the churn prediction service.

Policy: errors are logged with full details (including stack traces), but the
caller only ever receives a sanitized message plus the request ID for tracing.
Internal paths, stack traces, and data values never leave the service.
"""

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from ..utils.logger import get_logger
from ..utils.prometheus_metrics import prediction_error_count

logger = get_logger("churn_api")


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """
    Handle Pydantic validation errors with user-friendly messages.

    Includes request_id in error response for tracing.

    Triggered when:
    - Missing required fields
    - Wrong data types (string instead of int)
    - Invalid values (negative numbers where positive expected)
    - Extra fields not in schema (if configured)

    Returns 422 Unprocessable Entity with helpful error details.
    """
    # Get request ID from middleware
    request_id = getattr(request.state, "request_id", "no-request-id")

    logger.warning(f"Request validation error on {request.url.path}")
    logger.debug(f"Validation errors: {exc.errors()}")

    # Extract user-friendly error messages
    errors = []
    for error in exc.errors():
        # Build field path (e.g., "customers -> 0 -> tenure")
        field_path = " -> ".join(str(x) for x in error["loc"])
        message = error["msg"]
        error_type = error["type"]
        errors.append({"field": field_path, "message": message, "type": error_type})

    prediction_error_count.labels(error_type="validation_error").inc()

    response = JSONResponse(
        status_code=422,  # Unprocessable Entity
        content={
            "error": "Validation Error",
            "detail": "Request does not match expected schema",
            "errors": errors,
            "hint": "Check API documentation at /docs for correct schema",
            "request_id": request_id,  # Include for tracing
        },
    )
    # Add X-Request-ID header
    response.headers["X-Request-ID"] = request_id
    return response


async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Catch-all for unexpected exceptions.

    Uses request_id from middleware for tracing.

    Security: Logs full details but returns generic message to user.
    NEVER leak stack traces, internal paths, or sensitive data.

    This handler should only trigger for truly unexpected errors.
    Most errors should be caught by specific handlers or try/except blocks.
    """
    # Get request ID from middleware
    request_id = getattr(request.state, "request_id", "no-request-id")

    # Log full details (including stack trace)
    logger.critical(
        f"UNHANDLED EXCEPTION on {request.url.path}: {type(exc).__name__}: {exc}", exc_info=True
    )

    prediction_error_count.labels(error_type="unhandled_exception").inc()

    # Return generic error (don't leak internal details)
    response = JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": "An unexpected error occurred. Please try again later.",
            "request_id": request_id,
            "hint": "If this persists, contact support with the request_id",
        },
    )
    # Add X-Request-ID header
    response.headers["X-Request-ID"] = request_id
    return response


def register_error_handlers(app: FastAPI) -> None:
    """Attach the sanitizing exception handlers to the application."""
    app.add_exception_handler(RequestValidationError, validation_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, global_exception_handler)
