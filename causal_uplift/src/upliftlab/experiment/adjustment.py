"""Variance reduction: CUPED and regression adjustment.

In a randomized trial, adjusting for **pre-treatment** covariates cannot change
what the treatment effect *is* (the covariates are balanced by design), but it
can estimate that effect more precisely by soaking up outcome variance the
treatment never touched. Two estimators, same idea:

* **CUPED** (Deng, Xu, Kohavi & Walker 2013) — subtract ``theta * (X - E[X])``
  from the outcome, with ``theta = Cov(Y, X) / Var(X)``. A single pre-period
  covariate. The variance of the effect estimate falls by ``rho**2``, where
  ``rho = Corr(Y, X)`` — so the technique is only ever as good as the
  correlation between the covariate and the outcome.
* **Regression adjustment** (Lin 2013) — OLS of the outcome on treatment, the
  centered covariates, *and their interactions*, with heteroskedasticity-robust
  SEs. The treatment coefficient is a consistent ATE that is never less precise
  than the unadjusted difference, asymptotically. This is CUPED generalized to
  many covariates.

The honest caveat, made concrete on this dataset in the README/DATA_NOTES: when
the available pre-period covariates barely correlate with a two-week response
outcome, the reduction is small. The method is not magic; it is a covariance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from upliftlab.data.load import design_matrix
from upliftlab.experiment.ate import diff_in_means


@dataclass(frozen=True)
class AdjustmentResult:
    """Unadjusted vs adjusted ATE and the precision gain between them."""

    outcome: str
    method: str
    covariates: str
    ate_unadj: float
    se_unadj: float
    ate_adj: float
    se_adj: float
    ci_low: float
    ci_high: float
    var_reduction: float          # 1 - (se_adj / se_unadj)**2
    eff_n_multiplier: float       # (se_unadj / se_adj)**2
    theta: float | None = None    # only for single-covariate CUPED

    def __str__(self) -> str:
        theta = "" if self.theta is None else f", theta={self.theta:.4f}"
        return (
            f"{self.method} on {self.outcome} [{self.covariates}]: "
            f"ATE {self.ate_adj:+.5f} (95% CI [{self.ci_low:+.5f}, {self.ci_high:+.5f}]); "
            f"se {self.se_unadj:.5f} -> {self.se_adj:.5f} "
            f"({self.var_reduction:+.1%} variance, "
            f"{self.eff_n_multiplier:.3f}x effective N){theta}"
        )


def _result(outcome, method, covariates, y, t, ate_adj, se_adj, theta=None, alpha=0.05):
    ate_unadj, se_unadj = diff_in_means(y, t)
    var_reduction = 1 - (se_adj / se_unadj) ** 2 if se_unadj > 0 else 0.0
    zcrit = stats.norm.ppf(1 - alpha / 2)
    return AdjustmentResult(
        outcome=outcome,
        method=method,
        covariates=covariates,
        ate_unadj=ate_unadj,
        se_unadj=se_unadj,
        ate_adj=float(ate_adj),
        se_adj=float(se_adj),
        ci_low=float(ate_adj - zcrit * se_adj),
        ci_high=float(ate_adj + zcrit * se_adj),
        var_reduction=float(var_reduction),
        eff_n_multiplier=float((se_unadj / se_adj) ** 2) if se_adj > 0 else float("nan"),
        theta=theta,
    )


def cuped(
    df: pd.DataFrame,
    outcome: str,
    pre_covariate: str,
    arm_col: str = "segment",
    control="No E-Mail",
    treatment: str = "Womens E-Mail",
    alpha: float = 0.05,
) -> AdjustmentResult:
    """Classic single-covariate CUPED using ``pre_covariate``."""
    sub = df[df[arm_col].isin([control, treatment])]
    t = (sub[arm_col] == treatment).to_numpy().astype(int)
    y = sub[outcome].to_numpy(dtype=float)
    x = sub[pre_covariate].to_numpy(dtype=float)

    if not (np.isfinite(y).all() and np.isfinite(x).all()):
        raise ValueError("cuped requires finite outcome and covariate values (no NaN/inf)")
    var_x = x.var(ddof=1) if len(x) >= 2 else float("nan")
    if not var_x > 1e-12:
        raise ValueError(
            f"pre_covariate {pre_covariate!r} has (near-)zero variance; "
            "theta = Cov(Y, X) / Var(X) is undefined for a constant covariate"
        )
    theta = np.cov(y, x, ddof=1)[0, 1] / var_x
    y_adj = y - theta * (x - x.mean())
    ate_adj, se_adj = diff_in_means(y_adj, t)
    return _result(outcome, "CUPED", pre_covariate, y, t, ate_adj, se_adj, theta=theta, alpha=alpha)


def regression_adjustment(
    df: pd.DataFrame,
    outcome: str,
    numeric: list[str],
    categorical: list[str] | None = None,
    arm_col: str = "segment",
    control="No E-Mail",
    treatment: str = "Womens E-Mail",
    alpha: float = 0.05,
) -> AdjustmentResult:
    """Lin (2013) regression adjustment: treatment interacted with centered covariates.

    Fits ``y ~ 1 + t + Xc + t:Xc`` (Xc = mean-centered covariates) with HC1
    robust standard errors; the coefficient on ``t`` is the adjusted ATE.
    """
    categorical = categorical or []
    sub = df[df[arm_col].isin([control, treatment])].reset_index(drop=True)
    t = (sub[arm_col] == treatment).to_numpy().astype(float)
    y = sub[outcome].to_numpy(dtype=float)

    X, names = design_matrix(sub, numeric=numeric, categorical=categorical)
    Xc = X.to_numpy(dtype=float)
    Xc = Xc - Xc.mean(axis=0, keepdims=True)          # center so main-t coef is the ATE
    inter = Xc * t[:, None]                            # treatment × covariate interactions
    design = np.column_stack([np.ones_like(t), t, Xc, inter])

    res = sm.OLS(y, design).fit(cov_type="HC1")
    ate_adj = res.params[1]                            # coefficient on t
    se_adj = res.bse[1]
    label = "+".join(numeric + categorical)
    return _result(outcome, "regression-adjustment", label, y, t, ate_adj, se_adj, alpha=alpha)
