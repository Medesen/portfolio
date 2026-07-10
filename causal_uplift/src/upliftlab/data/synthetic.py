"""A synthetic randomized-trial generator with known ground truth.

Everything the estimators claim to recover is *known* here, which is what makes
the test-suite meaningful: a true average effect, a true per-unit treatment
effect (CATE) that varies with a covariate, a tunable pre-period covariate whose
correlation with the outcome sets the CUPED variance-reduction ceiling, and an
optional knob to break randomization so the balance check has something to catch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit

FEATURES = ["x1", "x2", "x3"]


@dataclass(frozen=True)
class SyntheticRCT:
    """A generated trial plus the ground truth used to score estimators."""

    data: pd.DataFrame          # columns: x1,x2,x3, pre, t, y, cate_true
    ate: float                  # population average treatment effect
    sate: float                 # sample ATE = mean(cate_true) over the drawn units


def make_synthetic_rct(
    n: int = 8000,
    seed: int = 0,
    kind: str = "continuous",
    p_treat: float = 0.5,
    ate: float = 1.0,
    hetero: float = 1.0,
    rho_pre: float = 0.0,
    confound: float = 0.0,
) -> SyntheticRCT:
    """Generate a synthetic trial.

    Parameters
    ----------
    kind : ``"continuous"`` (spend-like) or ``"binary"`` (visit-like).
    ate : population average treatment effect. For ``continuous`` it is the
        additive effect on the outcome; for ``binary`` it is the logit shift's
        base and the realised probability-scale ATE is returned in ``.ate``.
    hetero : strength of effect heterogeneity in ``x1``. ``0`` gives a constant
        (homogeneous) effect; larger values make *who* you treat matter more.
    rho_pre : correlation between the pre-period covariate ``pre`` and the
        outcome. Sets the CUPED variance-reduction ceiling (``~rho_pre**2``).
    confound : if > 0, treatment probability depends on ``x1`` (assignment is no
        longer randomized) — used to prove the balance check flags imbalance.
    """
    rng = np.random.default_rng(seed)
    x1, x2, x3 = (rng.standard_normal(n) for _ in range(3))
    pre = rng.standard_normal(n)

    # Assignment: randomized unless `confound` tilts it on x1.
    logit_t = np.log(p_treat / (1 - p_treat)) + confound * x1
    t = (rng.random(n) < expit(logit_t)).astype(int)

    if kind == "continuous":
        cate_true = ate + hetero * x1                      # mean = ate (E[x1]=0)
        base = 0.5 * x2 - 0.3 * x3
        noise = rho_pre * pre + np.sqrt(max(1e-12, 1 - rho_pre**2)) * rng.standard_normal(n)
        y = base + noise + cate_true * t
        pop_ate = float(ate)
    elif kind == "binary":
        base = -1.0 + 0.8 * x2 - 0.5 * x3 + 0.6 * rho_pre * pre
        shift = ate + hetero * x1                          # heterogeneous logit shift
        p0, p1 = expit(base), expit(base + shift)
        cate_true = p1 - p0                                # true per-unit uplift
        u = rng.random(n)
        y = np.where(t == 1, (u < p1).astype(int), (u < p0).astype(int)).astype(float)
        pop_ate = float(cate_true.mean())
    else:
        raise ValueError(f"kind must be 'continuous' or 'binary', got {kind!r}")

    data = pd.DataFrame(
        {"x1": x1, "x2": x2, "x3": x3, "pre": pre, "t": t, "y": y, "cate_true": cate_true}
    )
    return SyntheticRCT(data=data, ate=pop_ate, sate=float(cate_true.mean()))
