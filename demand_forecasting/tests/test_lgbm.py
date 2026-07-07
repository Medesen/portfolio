import numpy as np
import pandas as pd
import pytest

from demandcast.evaluation import make_folds, run_backtest
from demandcast.models.lgbm import HORIZON, LgbmForecaster, build_features


@pytest.fixture(scope="module")
def toy_long():
    rng = np.random.default_rng(1)
    dates = pd.date_range("2020-01-06", periods=200, freq="D")
    rows = []
    for sku, level in [("A_1", 20.0), ("A_2", 5.0), ("B_1", 10.0)]:
        promo = (rng.random(len(dates)) < 0.25).astype(int)
        weekly = 1 + 0.3 * np.sin(2 * np.pi * dates.dayofweek / 7)
        qty = np.maximum(0, level * weekly * (1 + promo) + rng.normal(0, 1, len(dates)))
        brand = sku.split("_")[0]
        for d, q, p in zip(dates, qty, promo):
            rows.append({"date": d, "sku": sku, "brand": brand, "qty": q, "promo": p})
    return pd.DataFrame(rows)


def test_sales_features_respect_horizon_shift(toy_long):
    """Changing qty within HORIZON days before date d must not change any
    sales-history feature at d (promo/calendar features are exempt)."""
    feats_clean = build_features(toy_long)
    d = toy_long["date"].max()

    corrupted = toy_long.copy()
    recent = corrupted["date"] > d - pd.Timedelta(days=HORIZON)
    corrupted.loc[recent, "qty"] += 1_000_000
    feats_corrupted = build_features(corrupted)

    sales_cols = [c for c in feats_clean.columns if c.startswith(("lag_", "rmean_", "rstd_", "zero_share_"))]
    row_clean = feats_clean[feats_clean["date"] == d][sales_cols]
    row_corrupted = feats_corrupted[feats_corrupted["date"] == d][sales_cols]
    pd.testing.assert_frame_equal(row_clean, row_corrupted)


def test_lgbm_smoke_backtest_with_quantiles(toy_long):
    folds = make_folds(toy_long["date"], n_folds=1, horizon=14, stride=14)
    preds = run_backtest(toy_long, LgbmForecaster(full_long=toy_long), folds)
    assert len(preds) == 14 * 3
    assert preds["y_pred"].notna().all()
    assert (preds["y_pred"] >= 0).all()
    for c in ["y_q10", "y_q50", "y_q90"]:
        assert c in preds.columns and preds[c].notna().all()
    # quantiles ordered on average (individual crossings are possible)
    assert preds["y_q10"].mean() < preds["y_q90"].mean()


def test_lgbm_no_leakage_from_test_window(toy_long):
    folds = make_folds(toy_long["date"], n_folds=1, horizon=14, stride=14)
    preds_clean = run_backtest(toy_long, LgbmForecaster(full_long=toy_long), folds)

    corrupted = toy_long.copy()
    in_test = corrupted["date"].isin(folds[0].test_dates)
    corrupted.loc[in_test, "qty"] += 1_000_000
    preds_corrupted = run_backtest(corrupted, LgbmForecaster(full_long=corrupted), folds)

    assert np.allclose(preds_clean["y_pred"], preds_corrupted["y_pred"])
