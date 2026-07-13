"""
Unit tests for retraining rate-limit reservation.

Covers the atomic check-and-reserve gate that replaced the previous
check-then-act design (which could let concurrent triggers launch duplicate
10-minute trainings, and never recorded failed attempts).
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta

import pytest

# Unset auth secrets before importing serve (matches test_api.py)
os.environ.pop("SERVICE_TOKEN_FILE", None)
os.environ.pop("SERVICE_TOKEN", None)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import serve  # noqa: E402


@pytest.fixture
def serve_module(tmp_path, monkeypatch):
    """serve module with an isolated reservation file and a 24h interval."""
    monkeypatch.setattr(serve, "LAST_RETRAIN_FILE", tmp_path / ".last_retrain")
    monkeypatch.setattr(serve, "MIN_RETRAIN_INTERVAL_HOURS", 24)
    return serve


class _FakeCompleted:
    def __init__(self, returncode: int, stderr: str = ""):
        self.returncode = returncode
        self.stderr = stderr


@pytest.mark.unit
def test_reserve_allows_first_blocks_second(serve_module):
    """First caller reserves the slot; a concurrent second caller is blocked."""
    assert serve_module.check_and_reserve_retraining() is True
    assert serve_module.check_and_reserve_retraining() is False


@pytest.mark.unit
def test_reservation_written_before_training(serve_module):
    """The reservation timestamp is written up front, not after training."""
    assert serve_module.check_and_reserve_retraining() is True
    assert serve_module.LAST_RETRAIN_FILE.exists()

    reserved = serve_module.get_last_retrain_time()
    assert reserved is not None
    assert (datetime.now() - reserved) < timedelta(minutes=1)


@pytest.mark.unit
def test_reserve_allowed_again_after_interval(serve_module):
    """Once the interval has elapsed, a new reservation is allowed."""
    old = datetime.now() - timedelta(hours=25)
    serve_module.LAST_RETRAIN_FILE.write_text(old.isoformat())

    assert serve_module.check_and_reserve_retraining() is True


@pytest.mark.unit
def test_failed_run_keeps_reservation(serve_module, monkeypatch):
    """A failed training run does not roll back the reservation (crash-loop guard)."""
    assert serve_module.check_and_reserve_retraining() is True
    reserved = serve_module.LAST_RETRAIN_FILE.read_text()

    def fake_run(*args, **kwargs):
        return _FakeCompleted(returncode=1, stderr="boom")

    monkeypatch.setattr(serve_module.subprocess, "run", fake_run)
    asyncio.run(serve_module.retrain_model_task())

    # Reservation is unchanged and still blocks a subsequent trigger
    assert serve_module.LAST_RETRAIN_FILE.read_text() == reserved
    assert serve_module.check_and_reserve_retraining() is False
