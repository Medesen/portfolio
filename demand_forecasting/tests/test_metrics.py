import numpy as np
import pandas as pd
import pytest

from demandcast.evaluation.metrics import mase, rmse, wape


def test_rmse_hand_computed():
    assert rmse(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 5.0])) == pytest.approx(
        np.sqrt(4 / 3)
    )


def test_wape_hand_computed():
    # sum|err| = 3, sum|actual| = 10
    assert wape(np.array([4.0, 6.0]), np.array([5.0, 4.0])) == pytest.approx(0.3)


def test_wape_zero_denominator_is_nan():
    assert np.isnan(wape(np.array([0.0, 0.0]), np.array([1.0, 1.0])))


def test_mase_hand_computed():
    # train with lag-2 abs diffs: |3-1|, |4-2|, |5-3|, |6-4| -> mean 2.0
    y_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    y_true = np.array([7.0, 8.0])
    y_pred = np.array([8.0, 6.0])  # MAE = 1.5
    assert mase(y_true, y_pred, y_train, m=2) == pytest.approx(1.5 / 2.0)


def test_mase_constant_train_is_nan():
    y_train = np.full(20, 5.0)
    assert np.isnan(mase(np.array([5.0]), np.array([5.0]), y_train, m=7))


def test_mase_short_train_is_nan():
    assert np.isnan(mase(np.array([1.0]), np.array([1.0]), np.array([1.0, 2.0]), m=7))
