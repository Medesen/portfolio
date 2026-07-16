import numpy as np
import pytest

from upliftlab.data import make_synthetic_rct
from upliftlab.experiment import cuped, regression_adjustment


def _df(rho, hetero=0.0, ate=0.3, seed=5):
    return make_synthetic_rct(
        n=20000, seed=seed, kind="continuous", ate=ate, hetero=hetero, rho_pre=rho
    ).data


def test_cuped_reduction_equals_squared_outcome_correlation():
    # CUPED's exact identity: variance reduction == Corr(Y, X)**2, where the
    # correlation is with the OUTCOME (diluted by outcome variance the covariate
    # doesn't explain) — not the raw rho_pre knob.
    d = _df(rho=0.7)
    r = cuped(d, "y", pre_covariate="pre", arm_col="t", control=0, treatment=1)
    rho = np.corrcoef(d["y"], d["pre"])[0, 1]
    assert r.var_reduction == pytest.approx(rho**2, abs=0.03)
    assert r.var_reduction > 0.2                       # materially positive here
    # and it must not move the effect estimate (covariate is pre-treatment)
    assert r.ci_low < 0.3 < r.ci_high


def test_cuped_no_reduction_when_covariate_uncorrelated():
    r = cuped(_df(rho=0.0), "y", pre_covariate="pre", arm_col="t", control=0, treatment=1)
    assert abs(r.var_reduction) < 0.05


def test_cuped_rejects_constant_covariate():
    # A zero-variance covariate makes theta = Cov(Y, X) / Var(X) undefined;
    # this must be a clear error, not a silent division by zero.
    d = _df(rho=0.5).copy()
    d["pre"] = 1.0
    with pytest.raises(ValueError, match="zero variance"):
        cuped(d, "y", pre_covariate="pre", arm_col="t", control=0, treatment=1)


def test_cuped_rejects_non_finite_values():
    d = _df(rho=0.5).copy()
    d.loc[0, "pre"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        cuped(d, "y", pre_covariate="pre", arm_col="t", control=0, treatment=1)


def test_regression_adjustment_beats_single_covariate_cuped():
    # y depends on x2, x3, and pre; adjusting for all should reduce variance
    # more than CUPED on `pre` alone.
    d = _df(rho=0.5)
    c = cuped(d, "y", pre_covariate="pre", arm_col="t", control=0, treatment=1)
    r = regression_adjustment(
        d, "y", numeric=["x1", "x2", "x3", "pre"], categorical=[],
        arm_col="t", control=0, treatment=1,
    )
    assert r.var_reduction > c.var_reduction
    assert r.ci_low < 0.3 < r.ci_high
