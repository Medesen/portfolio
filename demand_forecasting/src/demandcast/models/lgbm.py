"""Global LightGBM across all 118 SKUs, with quantile companions.

**One model, all series.** Cross-learning is the standard way to make ML
forecasting work on low-volume series: slow movers borrow strength from the
patterns of fast movers via shared features and the SKU/brand categoricals.

**Leakage discipline.** Every feature derived from sales history is shifted
by at least ``HORIZON`` (28) trading days, so each prediction for date ``d``
uses sales information from ``d - 28`` or older — a *true* 28-day-ahead
forecast for every day in the test window, not just the first. Promo and
calendar features are exempt from the shift because they are known in
advance (the promo calendar is planned weeks ahead; see DATA_NOTES.md).

**Objective.** Tweedie (variance power 1.2) as the default for zero-inflated
counts, with Poisson and plain L2 available via ``--objective`` for the
ablation reported in DATA_NOTES.md §2. Quantile models for P10/P50/P90 share
the same features and ride along in the same backtest, giving prediction
intervals — what a supply planner actually consumes.

Hyperparameters are sensible fixed values, deliberately not tuned: with 12
folds × 4 models the honest tuning protocol (nested validation) would
dominate the project's runtime for marginal insight. Stated as future work.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb

from demandcast.data.calendar import calendar_features
from demandcast.evaluation.backtest import Fold

HORIZON = 28  # minimum trading-day shift for all sales-history features

QTY_LAGS = [28, 29, 35, 42, 56, 91, 182, 364]
ROLL_WINDOWS = [7, 28, 91]

LGB_PARAMS = {
    "objective": "tweedie",
    "tweedie_variance_power": 1.2,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.9,
    "verbosity": -1,
}
NUM_BOOST_ROUND = 1000

#: Point-forecast objectives for the DATA_NOTES §2 ablation. Tweedie is the
#: count-appropriate default; the LightGBM objective string for "l2" is
#: "regression". tweedie_variance_power only applies to tweedie.
OBJECTIVES = {"tweedie": "tweedie", "poisson": "poisson", "l2": "regression"}

CATEGORICAL = ["sku", "brand"]


def build_features(long: pd.DataFrame) -> pd.DataFrame:
    """Feature frame for the full panel (one row per date × SKU).

    Sales-history features use ``groupby(sku).shift(>= HORIZON)`` on the
    trading-day grid. LightGBM handles the NaNs at the start of each series
    natively; only rows where even ``lag_28`` is missing are unusable.
    """
    df = long.sort_values(["sku", "date"]).reset_index(drop=True)
    g = df.groupby("sku", observed=True)["qty"]

    for lag in QTY_LAGS:
        df[f"lag_{lag}"] = g.shift(lag)

    base = g.shift(HORIZON)  # windows end HORIZON days back
    grouped_base = base.groupby(df["sku"], observed=True)
    for w in ROLL_WINDOWS:
        df[f"rmean_{w}"] = grouped_base.transform(lambda s: s.rolling(w).mean())
    df["rstd_28"] = grouped_base.transform(lambda s: s.rolling(28).std())
    df["zero_share_28"] = grouped_base.transform(lambda s: (s == 0).rolling(28).mean())

    # Known-in-advance promo schedule: no shift needed, leads are legitimate.
    gp = df.groupby("sku", observed=True)["promo"]
    df["promo_lag1"] = gp.shift(1)
    df["promo_lead1"] = gp.shift(-1)
    df["promo_share_next7"] = gp.transform(
        lambda s: s.shift(-6).rolling(7).mean()
    )

    cal = calendar_features(df["date"])
    df = pd.concat([df, cal], axis=1)

    for c in CATEGORICAL:
        df[c] = df[c].astype("category")
    return df


FEATURES = (
    [f"lag_{lag}" for lag in QTY_LAGS]
    + [f"rmean_{w}" for w in ROLL_WINDOWS]
    + ["rstd_28", "zero_share_28"]
    + ["promo", "promo_lag1", "promo_lead1", "promo_share_next7"]
    + ["dayofweek", "month", "dayofmonth", "weekofyear",
       "is_weekend", "is_holiday", "is_holiday_eve"]
    + CATEGORICAL
)


class LgbmForecaster:
    name = "lgbm"

    def __init__(
        self,
        full_long: pd.DataFrame,
        quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
        objective: str = "tweedie",
    ):
        """``full_long`` supplies the promo/calendar schedule and the feature
        grid; sales targets for training only ever come from the per-fold
        train slice (and features are >= HORIZON days old by construction)."""
        if objective not in OBJECTIVES:
            raise ValueError(f"objective must be one of {sorted(OBJECTIVES)}, got {objective!r}")
        self._features = build_features(full_long)
        self._quantiles = quantiles
        self._objective = objective

    def fit_predict(self, train: pd.DataFrame, fold: Fold) -> pd.DataFrame:
        feats = self._features
        train_rows = feats[
            feats["date"].isin(train["date"].unique())
            & feats["lag_28"].notna()
        ]
        # Targets from the train argument, honoring the harness contract.
        train_rows = train_rows.drop(columns=["qty"]).merge(
            train[["date", "sku", "qty"]], on=["date", "sku"], how="inner"
        )
        for c in CATEGORICAL:  # merge degrades category dtype; pin the full
            # category set so train/predict category codes always align
            train_rows[c] = train_rows[c].astype(feats[c].dtype)

        # Early stopping on the most recent 28 trading days of train — still
        # strictly before the test window.
        val_dates = np.sort(train_rows["date"].unique())[-HORIZON:]
        is_val = train_rows["date"].isin(val_dates)
        X_tr, y_tr = train_rows.loc[~is_val, FEATURES], train_rows.loc[~is_val, "qty"]
        X_val, y_val = train_rows.loc[is_val, FEATURES], train_rows.loc[is_val, "qty"]

        test_rows = feats[feats["date"].isin(fold.test_dates)]
        X_te = test_rows[FEATURES]

        point_params = dict(LGB_PARAMS)
        if self._objective != "tweedie":
            point_params["objective"] = OBJECTIVES[self._objective]
            point_params.pop("tweedie_variance_power")

        out = test_rows[["date", "sku"]].copy()
        out["y_pred"] = self._fit_one(point_params, X_tr, y_tr, X_val, y_val, X_te)
        for q in self._quantiles:
            params = {**LGB_PARAMS, "objective": "quantile", "alpha": q}
            params.pop("tweedie_variance_power")
            out[f"y_q{int(q * 100)}"] = self._fit_one(params, X_tr, y_tr, X_val, y_val, X_te)
        return out

    @staticmethod
    def _fit_one(params, X_tr, y_tr, X_val, y_val, X_te) -> np.ndarray:
        # Native API rather than the sklearn wrapper: no scikit-learn
        # dependency, and explicit control of early stopping.
        dtrain = lgb.Dataset(X_tr, label=y_tr, categorical_feature=CATEGORICAL)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        booster = lgb.train(
            params,
            dtrain,
            num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        pred = booster.predict(X_te, num_iteration=booster.best_iteration)
        return np.clip(pred, 0.0, None)
