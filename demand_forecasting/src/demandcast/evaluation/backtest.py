"""Rolling-origin backtesting on the trading-day grid.

Folds are built backwards from the last trading day: each fold holds out the
next ``horizon`` trading days after its origin and trains on everything up to
and including the origin (expanding window). A random train/test split would
leak future information into training; rolling-origin evaluation is the
time-series-correct alternative and the single most important design choice
in this project's evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd


@dataclass(frozen=True)
class Fold:
    fold: int
    train_end: pd.Timestamp        # last trading day available for training
    test_dates: pd.DatetimeIndex   # the next `horizon` trading days


class Forecaster(Protocol):
    """Anything with a name that can fit on a train slice and predict a fold."""

    name: str

    def fit_predict(self, train: pd.DataFrame, fold: Fold) -> pd.DataFrame:
        """Return a frame with columns ``date, sku, y_pred`` covering every
        (test date × SKU) combination of the fold."""
        ...


def make_folds(
    trading_days: pd.DatetimeIndex | pd.Series,
    n_folds: int = 12,
    horizon: int = 28,
    stride: int = 28,
) -> list[Fold]:
    """Build ``n_folds`` rolling-origin folds, newest data last.

    With the defaults (12 × 28-day non-overlapping folds) the evaluation
    window covers roughly the final year of data.
    """
    td = pd.DatetimeIndex(sorted(pd.unique(trading_days)))
    needed = horizon + (n_folds - 1) * stride
    if needed >= len(td) - 1:
        raise ValueError(
            f"{n_folds} folds x stride {stride} with horizon {horizon} need "
            f"{needed} trading days for testing; only {len(td)} available"
        )

    folds = []
    for k in range(n_folds):
        test_end = len(td) - k * stride           # exclusive
        test_start = test_end - horizon
        folds.append(
            Fold(
                fold=n_folds - k,
                train_end=td[test_start - 1],
                test_dates=td[test_start:test_end],
            )
        )
    return sorted(folds, key=lambda f: f.fold)


def run_backtest(
    long: pd.DataFrame,
    model: Forecaster,
    folds: list[Fold],
    train_long: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Run ``model`` over all folds; return stacked predictions with actuals.

    The model only ever sees rows with ``date <= fold.train_end`` — the test
    slice is used exclusively to attach ``y_true`` afterwards. ``train_long``
    optionally widens the training pool (e.g. the full 118-SKU panel while
    evaluating on a subset); evaluation rows and the coverage check always
    come from ``long``. A coverage check fails loudly if the model drops SKUs
    or dates instead of letting missing predictions silently vanish from the
    metrics.
    """
    source = long if train_long is None else train_long
    out = []
    for fold in folds:
        train = source[source["date"] <= fold.train_end]
        test = long[long["date"].isin(fold.test_dates)]

        pred = model.fit_predict(train, fold)

        # validate="one_to_one": both sides are unique on (date, sku), so a model
        # returning duplicate predictions raises here instead of silently
        # fanning out rows and overweighting those pairs in the metrics.
        merged = test[["date", "sku", "qty"]].merge(
            pred, on=["date", "sku"], how="left", validate="one_to_one"
        )
        if merged["y_pred"].isna().any():
            missing = merged[merged["y_pred"].isna()][["date", "sku"]].head(5)
            raise RuntimeError(
                f"{model.name}: fold {fold.fold} left (date, sku) pairs "
                f"unpredicted, e.g.\n{missing}"
            )

        merged = merged.rename(columns={"qty": "y_true"})
        merged["fold"] = fold.fold
        merged["train_end"] = fold.train_end
        # quantile columns (y_q10, ...) pass through when the model emits them
        extra = [c for c in merged.columns if c.startswith("y_q")]
        out.append(merged[["fold", "train_end", "date", "sku", "y_true", "y_pred", *extra]])

    return pd.concat(out, ignore_index=True)
