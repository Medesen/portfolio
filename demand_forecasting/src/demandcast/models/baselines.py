"""Naive baselines. Every model in this project has to beat these to matter.

Most forecasting write-ups skip the baseline and compare fancy models only
against each other; without the seasonal-naive reference there is no way to
tell whether *any* of them adds value over "sell what you sold last week".
"""

from __future__ import annotations

import pandas as pd

from demandcast.evaluation.backtest import Fold


def _grid(fold: Fold, skus: pd.Index) -> pd.DataFrame:
    """Full (test date × SKU) prediction grid for a fold."""
    return (
        pd.MultiIndex.from_product([fold.test_dates, skus], names=["date", "sku"])
        .to_frame(index=False)
    )


class Naive:
    """Forecast = last observed value, held flat across the horizon."""

    name = "naive"

    def fit_predict(self, train: pd.DataFrame, fold: Fold) -> pd.DataFrame:
        last = train.sort_values("date").groupby("sku")["qty"].last()
        out = _grid(fold, last.index)
        out["y_pred"] = out["sku"].map(last).astype(float)
        return out


class SeasonalNaive:
    """Forecast = last observed value on the same weekday.

    Weekday-aligned rather than positional (lag-7 on the trading-day grid)
    so that dropped holiday dates cannot shift the weekly cycle out of phase.
    """

    name = "seasonal_naive"

    def fit_predict(self, train: pd.DataFrame, fold: Fold) -> pd.DataFrame:
        train = train.sort_values("date")
        last_by_dow = (
            train.assign(dow=train["date"].dt.dayofweek)
            .groupby(["sku", "dow"])["qty"]
            .last()
        )
        skus = train["sku"].unique()
        out = _grid(fold, pd.Index(skus))
        keys = pd.MultiIndex.from_arrays([out["sku"], out["date"].dt.dayofweek])
        out["y_pred"] = last_by_dow.reindex(keys).to_numpy(dtype=float)
        return out
