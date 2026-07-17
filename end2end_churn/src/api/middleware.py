"""
HTTP middleware for the churn prediction service.

Four layers, registered by the app factory in this order (so the size limit
is outermost, bounding memory before anything else buffers the body):

- request-ID generation and log-context tagging
- request timeout protection (504 on overrun; health endpoints exempt)
- Prometheus request-duration and request-count tracking
- request body-size enforcement (pure ASGI, handles chunked bodies)
"""

import asyncio
import time
import uuid
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import Counter
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from ..utils.logger import get_logger, request_id_context
from ..utils.prometheus_metrics import (
    prediction_error_count,
    request_count,
    request_duration,
)
from . import settings

logger = get_logger("churn_api")

request_timeout_count = Counter(
    "churn_api_request_timeouts_total",
    "Total number of requests that exceeded timeout",
    ["endpoint"],
)


async def request_id_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """
    Tag every request with a unique ID for tracing.

    Uses the client's X-Request-ID header if provided, otherwise generates a
    UUID. The ID is stored in request.state, set in the logging context (a
    ContextVar, so async-safe), and echoed back in the X-Request-ID response
    header — one greppable ID from request to logs to response.
    """
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id
    token = request_id_context.set(request_id)

    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        request_id_context.reset(token)


async def timeout_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """
    Cancel requests that exceed the configured processing timeout.

    Returns 504 and increments the timeout counter on overrun, which protects
    workers from starvation by slow or expensive requests. Health and metrics
    endpoints are exempt — they must always answer quickly.

    Configuration: REQUEST_TIMEOUT_ENABLED (default true) and
    REQUEST_TIMEOUT_SECONDS (default 30).
    """
    exempt_paths = {"/health", "/healthz", "/readyz", "/metrics", "/"}
    if request.url.path in exempt_paths or not settings.REQUEST_TIMEOUT_ENABLED:
        return await call_next(request)

    try:
        return await asyncio.wait_for(call_next(request), timeout=settings.REQUEST_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        endpoint = request.url.path
        logger.error(
            f"Request timeout after {settings.REQUEST_TIMEOUT_SECONDS}s: "
            f"{request.method} {endpoint}"
        )
        request_timeout_count.labels(endpoint=endpoint).inc()
        return JSONResponse(
            status_code=504,
            content={
                "error": "Gateway Timeout",
                "detail": (
                    f"Request processing exceeded {settings.REQUEST_TIMEOUT_SECONDS} "
                    "seconds timeout"
                ),
                "timeout_seconds": settings.REQUEST_TIMEOUT_SECONDS,
                "suggestion": "Try reducing request size or complexity",
            },
        )


async def metrics_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """
    Track request duration and outcome for every endpoint.

    Updates the Prometheus duration histogram and success/error counters, and
    logs requests slower than one second. Skips /metrics itself to avoid
    recursion.
    """
    if request.url.path == "/metrics":
        return await call_next(request)

    start_time = time.time()
    endpoint = request.url.path

    status_code = 500  # Default to error if something goes wrong
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    except Exception as e:
        logger.error(f"Unhandled error in request: {e}", exc_info=True)
        raise
    finally:
        duration = time.time() - start_time
        request_duration.labels(endpoint=endpoint).observe(duration)

        status = "success" if 200 <= status_code < 400 else "error"
        request_count.labels(endpoint=endpoint, status=status).inc()

        if duration > 1.0:
            logger.warning(f"Slow request: {endpoint} took {duration:.2f}s (status={status_code})")


class LimitRequestSizeMiddleware:
    """
    Pure-ASGI middleware enforcing the maximum request body size.

    The previous BaseHTTPMiddleware only inspected the Content-Length header, so
    a chunked request (no Content-Length) or a lying header bypassed the limit
    entirely. This middleware enforces the real size in two layers:

    1. If Content-Length is present and already over the limit, reject with 413
       immediately (cheap, no body read).
    2. Otherwise count actual body bytes as they stream in and reject once the
       limit is exceeded, buffering at most ``max_bytes`` — so the documented
       "prevents memory exhaustion" guarantee holds even without Content-Length.

    Registered outermost so it bounds memory before any body-buffering middleware.
    """

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Layer 1: cheap Content-Length rejection (when the header is honest)
        headers = dict(scope.get("headers") or [])
        content_length = headers.get(b"content-length")
        if content_length is not None:
            try:
                if int(content_length) > self.max_bytes:
                    await self._reject(scope, receive, send)
                    return
            except ValueError:
                pass  # malformed header -> fall through to byte counting

        # Layer 2: bound and count actual streamed body bytes
        total = 0
        buffered: list[Message] = []
        more_body = True
        while more_body:
            message = await receive()
            if message["type"] != "http.request":
                buffered.append(message)  # e.g. http.disconnect
                break
            total += len(message.get("body", b""))
            if total > self.max_bytes:
                await self._reject(scope, receive, send)
                return
            buffered.append(message)
            more_body = message.get("more_body", False)

        async def replay() -> Message:
            if buffered:
                return buffered.pop(0)
            return await receive()

        await self.app(scope, replay, send)

    async def _reject(self, scope: Scope, receive: Receive, send: Send) -> None:
        prediction_error_count.labels(error_type="payload_too_large").inc()
        max_mb = self.max_bytes / (1024 * 1024)
        logger.warning(f"Request rejected: body exceeds maximum {max_mb:.0f}MB")
        response = JSONResponse(
            status_code=413,
            content={
                "error": "Payload Too Large",
                "detail": f"Request body exceeds maximum {max_mb:.0f}MB",
                "max_size_mb": max_mb,
            },
        )
        await response(scope, receive, send)
