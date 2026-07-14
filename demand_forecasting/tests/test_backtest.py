import numpy as np
import pandas as pd
import pytest

from demandcast.evaluation import make_folds, run_backtest
from demandcast.models import Naive, SeasonalNaive


@pytest.fixture
def toy_long():
    """Two SKUs, 120 consecutive days, deterministic weekly pattern + level."""
    dates = pd.date_range("2020-01-01", periods=120, freq="D")
    rows = []
    for sku, level in [("A_1", 10.0), ("A_2", 100.0)]:
        for d in dates:
            rows.append(
                {"date": d, "sku": sku, "brand": "A",
                 "qty": level + d.dayofweek, "promo": 0}
            )
    return pd.DataFrame(rows)


def test_make_folds_shape_and_disjointness(toy_long):
    folds = make_folds(toy_long["date"], n_folds=3, horizon=14, stride=14)
    assert [f.fold for f in folds] == [1, 2, 3]
    all_test = pd.DatetimeIndex([])
    for f in folds:
        assert len(f.test_dates) == 14
        assert f.train_end < f.test_dates.min()  # train strictly precedes test
        assert all_test.intersection(f.test_dates).empty  # no overlap
        all_test = all_test.union(f.test_dates)
    # newest fold ends on the final date
    assert folds[-1].test_dates.max() == toy_long["date"].max()


def test_make_folds_rejects_impossible_request(toy_long):
    with pytest.raises(ValueError):
        make_folds(toy_long["date"], n_folds=10, horizon=28, stride=28)


def test_make_folds_accepts_minimal_panel():
    """Exactly one training day plus the requested test days is valid."""
    dates = pd.Series(pd.date_range("2020-01-01", periods=15, freq="D"))
    folds = make_folds(dates, n_folds=1, horizon=14, stride=14)
    assert len(folds) == 1
    assert folds[0].train_end == dates.iloc[0]  # the single training day
    assert len(folds[0].test_dates) == 14

    # one day fewer (no training day at all) must still be rejected
    with pytest.raises(ValueError):
        make_folds(dates.iloc[1:], n_folds=1, horizon=14, stride=14)


def test_seasonal_naive_is_exact_on_pure_weekly_pattern(toy_long):
    folds = make_folds(toy_long["date"], n_folds=2, horizon=14, stride=14)
    preds = run_backtest(toy_long, SeasonalNaive(), folds)
    # the series is a deterministic function of weekday -> zero error
    assert np.allclose(preds["y_true"], preds["y_pred"])


def test_naive_holds_last_value_flat(toy_long):
    folds = make_folds(toy_long["date"], n_folds=1, horizon=7, stride=7)
    preds = run_backtest(toy_long, Naive(), folds)
    last_train_day = folds[0].train_end
    for sku in ["A_1", "A_2"]:
        expected = toy_long.query("sku == @sku and date == @last_train_day")["qty"].iloc[0]
        assert (preds.query("sku == @sku")["y_pred"] == expected).all()


def test_backtest_coverage_is_complete(toy_long):
    folds = make_folds(toy_long["date"], n_folds=2, horizon=14, stride=14)
    preds = run_backtest(toy_long, SeasonalNaive(), folds)
    assert len(preds) == 2 * 14 * 2  # folds x horizon x skus
    assert preds["y_pred"].notna().all()


def test_no_leakage_from_test_window(toy_long):
    """Corrupting the future must not change the predictions."""
    folds = make_folds(toy_long["date"], n_folds=1, horizon=14, stride=14)
    preds_clean = run_backtest(toy_long, SeasonalNaive(), folds)

    corrupted = toy_long.copy()
    in_test = corrupted["date"].isin(folds[0].test_dates)
    corrupted.loc[in_test, "qty"] += 1_000_000
    preds_corrupted = run_backtest(corrupted, SeasonalNaive(), folds)

    assert np.allclose(preds_clean["y_pred"], preds_corrupted["y_pred"])


def test_duplicate_predictions_raise(toy_long):
    """Duplicate (date, sku) predictions must fail the merge, not silently
    fan out rows and overweight those pairs in the metrics."""
    folds = make_folds(toy_long["date"], n_folds=1, horizon=7, stride=7)

    class DuplicatingModel:
        name = "dup"

        def fit_predict(self, train, fold):
            skus = train["sku"].unique()
            rows = [
                {"date": d, "sku": sku, "y_pred": 1.0}
                for d in fold.test_dates
                for sku in skus
            ]
            pred = pd.DataFrame(rows)
            return pd.concat([pred, pred], ignore_index=True)  # each pair twice

    with pytest.raises(pd.errors.MergeError):
        run_backtest(toy_long, DuplicatingModel(), folds)
