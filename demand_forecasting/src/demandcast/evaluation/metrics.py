"""Forecast accuracy metrics.

Chosen for zero-inflated, low-volume count data (see DATA_NOTES.md):

- **MASE** — headline metric. Scale-free, so it can be averaged across SKUs
  whose volumes differ by ~60×. Values < 1 mean "better than the in-sample
  positional lag-7 naive" (≈ seasonal-naive; see the ``mase`` docstring for
  how the two differ in dropped-holiday weeks).
- **WAPE** — pooled ``sum|error| / sum|actual|``. Plays the role sMAPE would,
  without sMAPE's divide-by-zero pathology on zero-sales days.
- **RMSE** — pooled, in units sold; kept for interpretability.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Seasonal period used for the MASE scaling factor (trading-day week).
SEASONAL_PERIOD = 7

_EPS = 1e-9


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    denom = np.abs(y_true).sum()
    if denom < _EPS:
        return float("nan")
    return float(np.abs(y_true - np.asarray(y_pred)).sum() / denom)


def mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    m: int = SEASONAL_PERIOD,
) -> float:
    """MASE with a seasonal (lag-``m``) in-sample naive scaling factor.

    The scale is the *standard* MASE denominator: the mean absolute positional
    lag-``m`` (weekly, m=7) difference of the in-sample series on the trading-day
    grid. This is subtly different from the weekday-aligned ``seasonal_naive``
    baseline row — the two coincide except in weeks where a dropped holiday
    shifts the positional cycle out of phase. So "MASE < 1" means "beats a
    positional lag-7 naive", which only *approximately* equals "beats the
    weekday-aligned seasonal-naive baseline" (see DATA_NOTES.md).

    Returns NaN when the training series is (seasonally) constant, rather than
    dividing by ~0 and reporting a meaningless huge number.
    """
    y_train = np.asarray(y_train, dtype=float)
    if len(y_train) <= m:
        return float("nan")
    scale = np.abs(y_train[m:] - y_train[:-m]).mean()
    if scale < _EPS:
        return float("nan")
    y_true = np.asarray(y_true, dtype=float)
    return float(np.abs(y_true - np.asarray(y_pred)).mean() / scale)


def pinball(y_true: np.ndarray, y_q: np.ndarray, alpha: float) -> float:
    y_true = np.asarray(y_true, dtype=float)
    diff = y_true - np.asarray(y_q, dtype=float)
    return float(np.mean(np.maximum(alpha * diff, (alpha - 1) * diff)))


def score_quantiles(preds: pd.DataFrame) -> pd.DataFrame:
    """Pinball loss per quantile plus empirical central-interval coverage.

    For P10/P90 the interval coverage target is 0.80; materially lower means
    the intervals are overconfident, materially higher means they are wastefully
    wide — both matter to a planner sizing safety stock.
    """
    qcols = sorted(c for c in preds.columns if c.startswith("y_q"))
    rows = [
        {
            "quantile": (a := int(c.removeprefix("y_q")) / 100),
            "pinball": pinball(preds["y_true"].to_numpy(), preds[c].to_numpy(), a),
        }
        for c in qcols
    ]
    out = pd.DataFrame(rows).set_index("quantile")
    if {"y_q10", "y_q90"} <= set(qcols):
        covered = (preds["y_true"] >= preds["y_q10"]) & (preds["y_true"] <= preds["y_q90"])
        out.attrs["coverage_p10_p90"] = float(covered.mean())
    return out


def score(preds: pd.DataFrame, long: pd.DataFrame, m: int = SEASONAL_PERIOD) -> pd.DataFrame:
    """Score backtest predictions overall and by SKU volume tercile.

    ``preds`` is the output of :func:`demandcast.evaluation.backtest.run_backtest`
    (columns ``fold, train_end, date, sku, y_true, y_pred``); ``long`` is the
    full long frame, used for MASE scaling (train slice per fold) and for the
    tercile assignment. Terciles are computed from full-sample mean quantity —
    they are a *reporting* grouping, not a model input, so this is not leakage.

    Returns one row per segment (``overall``, ``low``, ``mid``, ``high``) with
    columns ``mase`` (mean of per-SKU-per-fold MASE), ``wape``, ``rmse``
    (pooled within segment) and ``n_pairs``.
    """
    mean_qty = long.groupby("sku")["qty"].mean()
    tercile = pd.qcut(mean_qty, 3, labels=["low", "mid", "high"])

    # Per-(fold, sku) MASE: scaling factor comes from that fold's train slice.
    per_series = []
    for (fold, train_end), fold_preds in preds.groupby(["fold", "train_end"]):
        train = long[long["date"] <= train_end]
        train_by_sku = dict(tuple(train.groupby("sku")["qty"]))
        for sku, p in fold_preds.groupby("sku"):
            per_series.append(
                {
                    "fold": fold,
                    "sku": sku,
                    "mase": mase(p["y_true"].to_numpy(), p["y_pred"].to_numpy(),
                                 train_by_sku[sku].to_numpy(), m=m),
                }
            )
    per_series = pd.DataFrame(per_series)
    per_series["tercile"] = per_series["sku"].map(tercile)

    preds = preds.assign(tercile=preds["sku"].map(tercile))
    segments: list[tuple[str, pd.DataFrame, pd.DataFrame]] = [
        ("overall", preds, per_series)
    ]
    for t in ["low", "mid", "high"]:
        segments.append((t, preds[preds["tercile"] == t], per_series[per_series["tercile"] == t]))

    rows = []
    for name, p, s in segments:
        rows.append(
            {
                "segment": name,
                # NaN-skipping mean over series-folds; the two count columns
                # make the exclusion visible: a MASE is NaN when the fold's
                # train slice is too short or seasonally constant, and those
                # series-folds simply drop out of the mean.
                "mase": s["mase"].mean(),
                "wape": wape(p["y_true"].to_numpy(), p["y_pred"].to_numpy()),
                "rmse": rmse(p["y_true"].to_numpy(), p["y_pred"].to_numpy()),
                "n_pairs": len(p),
                "n_series_folds": len(s),
                "n_mase_defined": int(s["mase"].notna().sum()),
            }
        )
    return pd.DataFrame(rows).set_index("segment")
