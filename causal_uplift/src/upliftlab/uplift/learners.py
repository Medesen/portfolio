"""Meta-learners for the conditional average treatment effect (CATE / uplift).

All three wrap the same base learner (gradient-boosted trees) and differ only in
how they turn response models into an effect estimate:

* **S-learner** — one model with treatment as a feature; uplift = f(x, 1) − f(x, 0).
  Simple, but can wash out a weak effect because the tree may ignore the treatment
  column.
* **T-learner** — separate treated and control models; uplift = m₁(x) − m₀(x).
  No shared structure, so it spends data on modelling the baseline twice.
* **X-learner** (Künzel et al. 2019) — a T-learner, then a second stage that models
  the *imputed* individual effects and blends them by the propensity. Designed to
  do well exactly here: many controls, fewer treated, real heterogeneity.

The base learner is deliberately fixed to sensible values, not tuned — matching
the sibling forecasting project's stance that un-nested tuning is either leakage
or runtime theatre for marginal gain.
"""

from __future__ import annotations

import warnings

import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor


def default_lgbm_params() -> dict:
    """Sensible, fixed, fast base-learner settings (not tuned)."""
    return dict(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=0,
        n_jobs=-1,
        verbosity=-1,
    )


def _as_array(X) -> np.ndarray:
    return X.to_numpy(dtype=float) if hasattr(X, "to_numpy") else np.asarray(X, dtype=float)


def _make(kind: str, params: dict):
    if kind == "binary":
        return LGBMClassifier(**params)
    if kind == "continuous":
        return LGBMRegressor(**params)
    raise ValueError(f"kind must be 'binary' or 'continuous', got {kind!r}")


def _predict(model, X: np.ndarray) -> np.ndarray:
    """Response on the outcome scale: P(Y=1|X) for classifiers, E[Y|X] for regressors."""
    with warnings.catch_warnings():
        # LightGBM's sklearn wrapper invents feature names ("Column_0"...) when
        # fit on a numpy array, then warns when asked to predict on a (nameless)
        # numpy array. We use numpy end-to-end on purpose, so this is spurious.
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)[:, 1]
        return model.predict(X)


class SLearner:
    name = "S-learner"

    def __init__(self, kind: str = "binary", params: dict | None = None):
        self.kind = kind
        self.params = params or default_lgbm_params()

    def fit(self, X, t, y):
        X = _as_array(X)
        t = np.asarray(t, dtype=float).reshape(-1, 1)
        self.model_ = _make(self.kind, self.params).fit(np.hstack([X, t]), np.asarray(y))
        return self

    def predict_uplift(self, X) -> np.ndarray:
        X = _as_array(X)
        ones = np.ones((X.shape[0], 1))
        p1 = _predict(self.model_, np.hstack([X, ones]))
        p0 = _predict(self.model_, np.hstack([X, 0 * ones]))
        return p1 - p0


class TLearner:
    name = "T-learner"

    def __init__(self, kind: str = "binary", params: dict | None = None):
        self.kind = kind
        self.params = params or default_lgbm_params()

    def fit(self, X, t, y):
        X, t, y = _as_array(X), np.asarray(t), np.asarray(y)
        self.m1_ = _make(self.kind, self.params).fit(X[t == 1], y[t == 1])
        self.m0_ = _make(self.kind, self.params).fit(X[t == 0], y[t == 0])
        return self

    def predict_uplift(self, X) -> np.ndarray:
        X = _as_array(X)
        return _predict(self.m1_, X) - _predict(self.m0_, X)


class XLearner:
    name = "X-learner"

    def __init__(self, kind: str = "binary", params: dict | None = None):
        self.kind = kind
        self.params = params or default_lgbm_params()

    def fit(self, X, t, y):
        X, t, y = _as_array(X), np.asarray(t), np.asarray(y, dtype=float)
        # Stage 1: T-learner response models.
        self.m1_ = _make(self.kind, self.params).fit(X[t == 1], y[t == 1])
        self.m0_ = _make(self.kind, self.params).fit(X[t == 0], y[t == 0])
        # Stage 2: impute individual effects and model them (always regression).
        d1 = y[t == 1] - _predict(self.m0_, X[t == 1])   # treated: actual - imputed control
        d0 = _predict(self.m1_, X[t == 0]) - y[t == 0]   # control: imputed treated - actual
        # Effect models are always regressors, but use the learner's configured
        # params (default_lgbm_params() by default) so a custom XLearner(params=...)
        # is honored in both stages, not just stage 1.
        self.tau1_ = LGBMRegressor(**self.params).fit(X[t == 1], d1)
        self.tau0_ = LGBMRegressor(**self.params).fit(X[t == 0], d0)
        self.propensity_ = float(t.mean())
        return self

    def predict_uplift(self, X) -> np.ndarray:
        X = _as_array(X)
        g = self.propensity_
        return g * _predict(self.tau0_, X) + (1 - g) * _predict(self.tau1_, X)


LEARNERS = {"s-learner": SLearner, "t-learner": TLearner, "x-learner": XLearner}


def response_model_scores(X_train, y_train, X_eval, kind: str = "binary") -> np.ndarray:
    """Predicted *response* (not uplift) from a model fit on the treated arm.

    This is the naive "target whoever is most likely to act" policy — the thing
    uplift modelling is meant to beat, because high responders are often people
    who would have acted anyway.
    """
    model = _make(kind, default_lgbm_params()).fit(_as_array(X_train), np.asarray(y_train))
    return _predict(model, _as_array(X_eval))
