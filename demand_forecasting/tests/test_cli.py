"""CLI validation tests for the ``demandcast`` entry point."""

import sys

import pytest

from demandcast import main as main_mod


def _run_cli(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["demandcast", *argv])
    main_mod.main()


def test_lgbm_horizon_over_shift_is_rejected(monkeypatch):
    """--model lgbm with a horizon beyond the fixed feature shift is a CLI error
    (it would read future sales), and fails before any data is loaded."""
    monkeypatch.setattr(main_mod, "load_long", lambda: pytest.fail("data loaded before guard"))

    with pytest.raises(SystemExit) as exc:
        _run_cli(monkeypatch, ["backtest", "--model", "lgbm", "--horizon", "35"])

    assert exc.value.code == 2  # argparse parser.error exit code


def test_lgbm_horizon_at_shift_is_allowed(monkeypatch):
    """Horizon exactly at the feature shift is allowed (guard uses > not >=)."""

    def reached():
        raise RuntimeError("reached load_long")

    monkeypatch.setattr(main_mod, "load_long", reached)

    with pytest.raises(RuntimeError, match="reached load_long"):
        _run_cli(monkeypatch, ["backtest", "--model", "lgbm", "--horizon", "28"])


def test_non_lgbm_horizon_is_not_restricted(monkeypatch):
    """Baselines/SARIMAX have no fixed feature shift, so a long horizon is allowed."""

    def reached():
        raise RuntimeError("reached load_long")

    monkeypatch.setattr(main_mod, "load_long", reached)

    with pytest.raises(RuntimeError, match="reached load_long"):
        _run_cli(monkeypatch, ["backtest", "--model", "seasonal_naive", "--horizon", "35"])
