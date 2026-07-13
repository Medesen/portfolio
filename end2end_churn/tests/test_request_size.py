"""
Unit tests for the request body-size limit (LimitRequestSizeMiddleware).

Exercises the middleware at the ASGI level so oversized bodies can be simulated
without sending multi-megabyte payloads, including the chunked case (no
Content-Length) that the old header-only check could not enforce.
"""

import asyncio
import os
import sys

import pytest

os.environ.pop("SERVICE_TOKEN_FILE", None)
os.environ.pop("SERVICE_TOKEN", None)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from serve import LimitRequestSizeMiddleware  # noqa: E402


def _run(middleware, scope, messages):
    """Drive an ASGI middleware with a scripted receive; return sent messages."""
    queue = list(messages)

    async def receive():
        assert queue, "receive called after body exhausted"
        return queue.pop(0)

    sent = []

    async def send(message):
        sent.append(message)

    asyncio.run(middleware(scope, receive, send))
    return sent


async def _drain_app(scope, receive, send):
    """Downstream app that reads the whole body and echoes its length."""
    body = b""
    while True:
        message = await receive()
        body += message.get("body", b"")
        if not message.get("more_body", False):
            break
    _drain_app.body_len = len(body)
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b"ok"})


def _status(sent):
    starts = [m for m in sent if m["type"] == "http.response.start"]
    assert starts, "no response.start sent"
    return starts[0]["status"]


@pytest.mark.unit
def test_rejects_chunked_body_over_limit():
    """A streamed body with no Content-Length is rejected once it exceeds the limit."""
    mw = LimitRequestSizeMiddleware(_drain_app, max_bytes=1000)
    scope = {"type": "http", "method": "POST", "headers": []}  # no content-length
    messages = [
        {"type": "http.request", "body": b"x" * 800, "more_body": True},
        {"type": "http.request", "body": b"x" * 800, "more_body": False},  # 1600 > 1000
    ]

    sent = _run(mw, scope, messages)

    assert _status(sent) == 413


@pytest.mark.unit
def test_allows_body_under_limit_and_replays_intact():
    """A body under the limit passes through and is replayed to the app intact."""
    mw = LimitRequestSizeMiddleware(_drain_app, max_bytes=10000)
    scope = {"type": "http", "method": "POST", "headers": []}
    messages = [
        {"type": "http.request", "body": b"x" * 500, "more_body": True},
        {"type": "http.request", "body": b"y" * 500, "more_body": False},
    ]

    sent = _run(mw, scope, messages)

    assert _status(sent) == 200
    assert _drain_app.body_len == 1000  # full body delivered downstream


@pytest.mark.unit
def test_rejects_by_content_length_without_reading_body():
    """An honest oversized Content-Length is rejected before the body is read."""
    mw = LimitRequestSizeMiddleware(_drain_app, max_bytes=1000)
    scope = {
        "type": "http",
        "method": "POST",
        "headers": [(b"content-length", b"999999999")],
    }

    # receive() asserts if called -> proves Layer 1 short-circuited
    sent = _run(mw, scope, messages=[])

    assert _status(sent) == 413
