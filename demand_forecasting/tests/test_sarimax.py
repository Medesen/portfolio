import numpy as np
import pandas as pd

from demandcast.evaluation import make_folds, run_backtest
from demandcast.models.sarimax import Sarimax, select_skus


def _toy_long(n_days=180, skus=(("A_1", 20.0), ("A_2", 3.0))):
    rng = np.random.default_rng(0)
    dates = pd.date_range("2020-01-06", periods=n_days, freq="D")  # starts Monday
    rows = []
    for sku, level in skus:
        promo = (rng.random(n_days) < 0.2).astype(int)
        weekly = 1.0 + 0.3 * np.sin(2 * np.pi * dates.dayofweek / 7)
        qty = np.maximum(0, level * weekly * (1 + 0.8 * promo) + rng.normal(0, 1, n_days))
        for d, q, p in zip(dates, qty, promo):
            rows.append({"date": d, "sku": sku, "brand": "A", "qty": q, "promo": p})
    return pd.DataFrame(rows)


def test_select_skus_filters_and_orders():
    long = _toy_long()
    # A_2 is low-volume with zeros; make it clearly intermittent
    long.loc[(long.sku == "A_2") & (long.date.dt.dayofweek < 3), "qty"] = 0.0
    picked = select_skus(long, n=2, max_zero_share=0.10)
    assert picked == ["A_1"]  # A_2 excluded by zero-share, ordering by volume


def test_sarimax_smoke_backtest():
    """One high-volume SKU, one fold: finite, non-negative, full coverage."""
    long = _toy_long(skus=(("A_1", 20.0),))
    folds = make_folds(long["date"], n_folds=1, horizon=14, stride=14)
    preds = run_backtest(long, Sarimax(promo_schedule=long), folds)
    assert len(preds) == 14
    assert preds["y_pred"].notna().all()
    assert (preds["y_pred"] >= 0).all()
    # sanity: forecasts are in the right ballpark, not degenerate
    assert 5 < preds["y_pred"].mean() < 60
