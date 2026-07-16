"""Per-series SARIMAX on a representative subset of high-volume SKUs.

Why a subset and not all 118 series: SARIMAX assumes (log-)Gaussian
innovations. The median SKU here sells ~3 units/day and is zero on 20% of
days — fitting a Gaussian state-space model to such series and presenting the
result as a fair comparison would be methodological theatre (see
DATA_NOTES.md §2). So the classical model is fit where its assumptions
roughly hold: the highest-volume, least-intermittent SKUs. The global
LightGBM covers the full assortment; the comparison between the two happens
on this common subset.

Known-in-advance covariates: the promotion calendar is passed in at
construction and treated as known at forecast time — retail promotions are
planned weeks ahead, so this mirrors the real forecasting situation. Sales
history is only ever taken from the training slice.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

logger = logging.getLogger(__name__)

from demandcast.data.calendar import calendar_features
from demandcast.evaluation.backtest import Fold

#: (order, seasonal_order) candidates; the best by AIC on the first training
#: window is selected once per SKU, then reused across folds.
CANDIDATE_ORDERS: list[tuple[tuple[int, int, int], tuple[int, int, int, int]]] = [
    ((1, 0, 1), (1, 0, 1, 7)),
    ((1, 0, 1), (0, 1, 1, 7)),
    ((2, 0, 2), (1, 0, 1, 7)),
]


def select_skus(long: pd.DataFrame, n: int = 8, max_zero_share: float = 0.10) -> list[str]:
    """Deterministic subset rule: top-``n`` SKUs by mean volume among those
    with zero-sales share <= ``max_zero_share``."""
    g = long.groupby("sku")["qty"]
    stats = pd.DataFrame(
        {"mean_qty": g.mean(), "zero_share": g.apply(lambda s: (s == 0).mean())}
    )
    eligible = stats[stats["zero_share"] <= max_zero_share]
    return eligible.sort_values("mean_qty", ascending=False).head(n).index.tolist()


def _exog(df: pd.DataFrame) -> np.ndarray:
    """Exogenous regressors: promo flag + open-holiday + holiday-eve.

    ``is_holiday`` is not degenerate on the trading-day grid: only ~half of
    Italian public holidays close the store (those dates are absent); the
    rest are open holidays with distinct demand.
    """
    cal = calendar_features(df["date"])
    return np.column_stack(
        [
            df["promo"].to_numpy(dtype=float),
            cal["is_holiday"].to_numpy(dtype=float),
            cal["is_holiday_eve"].to_numpy(dtype=float),
        ]
    )


class Sarimax:
    name = "sarimax"

    def __init__(self, promo_schedule: pd.DataFrame):
        """``promo_schedule``: long frame with ``date, sku, promo`` covering
        the full period — the known-in-advance promotion calendar."""
        self._schedule = promo_schedule[["date", "sku", "promo"]]
        self._order_cache: dict[str, tuple] = {}

    def fit_predict(self, train: pd.DataFrame, fold: Fold) -> pd.DataFrame:
        preds = []
        for sku, ts in train.groupby("sku"):
            ts = ts.sort_values("date")
            # log1p stabilizes variance; forecasts are back-transformed with
            # expm1 (a median-type forecast under log-normality) and clipped
            # at zero.
            y = np.log1p(ts["qty"].to_numpy(dtype=float))
            X = _exog(ts)

            future = self._schedule[
                (self._schedule["sku"] == sku)
                & (self._schedule["date"].isin(fold.test_dates))
            ].sort_values("date")
            X_future = _exog(future)

            order, seasonal_order = self._select_order(sku, y, X)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = SARIMAX(
                    y,
                    exog=X,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False, maxiter=200)
                fc = res.forecast(steps=len(future), exog=X_future)

            preds.append(
                pd.DataFrame(
                    {
                        "date": future["date"].to_numpy(),
                        "sku": sku,
                        "y_pred": np.clip(np.expm1(fc), 0.0, None),
                    }
                )
            )
        return pd.concat(preds, ignore_index=True)

    def _select_order(self, sku: str, y: np.ndarray, X: np.ndarray) -> tuple:
        if sku not in self._order_cache:
            fits: list[tuple[float, bool, tuple]] = []  # (aic, converged, candidate)
            failures: list[str] = []
            for order, seasonal_order in CANDIDATE_ORDERS:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        res = SARIMAX(
                            y,
                            exog=X,
                            order=order,
                            seasonal_order=seasonal_order,
                            enforce_stationarity=False,
                            enforce_invertibility=False,
                        ).fit(disp=False, maxiter=200)
                except Exception as exc:
                    failures.append(f"{(order, seasonal_order)}: {exc}")
                    continue
                aic = float(res.aic)
                if not np.isfinite(aic):
                    failures.append(f"{(order, seasonal_order)}: non-finite AIC")
                    continue
                converged = bool(getattr(res, "mle_retvals", {}).get("converged", True))
                fits.append((aic, converged, (order, seasonal_order)))
            if failures:
                logger.warning(
                    "sku %s: %d of %d SARIMAX order candidates failed: %s",
                    sku, len(failures), len(CANDIDATE_ORDERS), "; ".join(failures),
                )
            if not fits:
                raise RuntimeError(
                    f"SARIMAX order selection failed for sku {sku!r}: "
                    f"every candidate raised or produced a non-finite AIC "
                    f"({'; '.join(failures)})"
                )
            converged_fits = [f for f in fits if f[1]]
            if not converged_fits:
                logger.warning(
                    "sku %s: no SARIMAX candidate converged; "
                    "selecting best AIC among non-converged fits",
                    sku,
                )
            self._order_cache[sku] = min(converged_fits or fits, key=lambda f: f[0])[2]
        return self._order_cache[sku]
