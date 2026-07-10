"""Average treatment effects with honest inference.

Because assignment is randomized, the average treatment effect is identified by a
plain difference in means — no outcome model required. What separates a careful
analysis from a naive one is everything *around* that number:

* unpooled (Welch) standard errors and normal-approximation CIs, valid for both
  the binary outcomes and the heavily zero-inflated ``spend``;
* a **minimum detectable effect** at 80% power, so a non-significant result can be
  read as "no effect this experiment could detect" rather than "no effect";
* **multiplicity control** (Holm and Benjamini-Hochberg) across the arm × outcome
  grid, because testing six effects at α = 0.05 is not the same as testing one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


@dataclass(frozen=True)
class ATEEstimate:
    """One difference-in-means effect with normal-approximation inference."""

    arm: str
    outcome: str
    n_control: int
    n_treat: int
    mean_control: float
    mean_treat: float
    effect: float
    se: float
    ci_low: float
    ci_high: float
    z: float
    p_value: float

    @property
    def rel_effect(self) -> float:
        """Effect as a fraction of the control mean (NaN if control mean is 0)."""
        return float(self.effect / self.mean_control) if self.mean_control else float("nan")

    def __str__(self) -> str:
        return (
            f"{self.arm} on {self.outcome}: {self.effect:+.5f} "
            f"(95% CI [{self.ci_low:+.5f}, {self.ci_high:+.5f}], "
            f"{self.rel_effect:+.1%} vs control, "
            f"control={self.mean_control:.5f}, se={self.se:.5f}, "
            f"z={self.z:.2f}, p={self.p_value:.2e})"
        )


def diff_in_means(y: np.ndarray, t: np.ndarray) -> tuple[float, float]:
    """Treatment-minus-control mean of ``y`` and its unpooled standard error.

    ``t`` is a 0/1 assignment vector aligned with ``y``. The SE is the Welch /
    two-sample form ``sqrt(var_t/n_t + var_c/n_c)`` with ``ddof=1``.
    """
    y = np.asarray(y, dtype=float)
    t = np.asarray(t)
    yt, yc = y[t == 1], y[t == 0]
    effect = yt.mean() - yc.mean()
    se = np.sqrt(yt.var(ddof=1) / yt.size + yc.var(ddof=1) / yc.size)
    return float(effect), float(se)


def estimate_ate(
    df: pd.DataFrame,
    outcome: str,
    arm: str,
    arm_col: str = "segment",
    control="No E-Mail",
    alpha: float = 0.05,
) -> ATEEstimate:
    """Difference-in-means effect of ``arm`` vs ``control`` on ``outcome``."""
    sub = df[df[arm_col].isin([control, arm])]
    t = (sub[arm_col] == arm).to_numpy().astype(int)
    y = sub[outcome].to_numpy(dtype=float)
    effect, se = diff_in_means(y, t)
    zcrit = stats.norm.ppf(1 - alpha / 2)
    z = effect / se if se > 0 else np.nan
    p = 2 * stats.norm.sf(abs(z))
    return ATEEstimate(
        arm=str(arm),
        outcome=outcome,
        n_control=int((t == 0).sum()),
        n_treat=int((t == 1).sum()),
        mean_control=float(y[t == 0].mean()),
        mean_treat=float(y[t == 1].mean()),
        effect=effect,
        se=se,
        ci_low=effect - zcrit * se,
        ci_high=effect + zcrit * se,
        z=float(z),
        p_value=float(p),
    )


def minimum_detectable_effect(
    n_treat: int, n_control: int, sd: float, alpha: float = 0.05, power: float = 0.8
) -> float:
    """Smallest true effect detectable at the given power (two-sided).

    Uses the control-arm standard deviation for both groups — the standard
    planning approximation. Returns an absolute effect on the outcome scale.
    """
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_power = stats.norm.ppf(power)
    se_unit = sd * np.sqrt(1 / n_treat + 1 / n_control)
    return float((z_alpha + z_power) * se_unit)


def achieved_power(effect: float, se: float, alpha: float = 0.05) -> float:
    """Post-hoc power to detect the observed effect at its estimated SE."""
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    return float(stats.norm.cdf(abs(effect) / se - z_alpha)) if se > 0 else float("nan")


def estimate_all(
    df: pd.DataFrame,
    outcomes: list[str],
    treatments: list[str],
    control="No E-Mail",
    arm_col: str = "segment",
    alpha: float = 0.05,
    power: float = 0.8,
) -> pd.DataFrame:
    """Every arm × outcome effect in one tidy table with multiplicity control.

    Adds Holm and Benjamini-Hochberg adjusted p-values across the whole grid,
    the minimum detectable effect at ``power``, and the post-hoc achieved power.
    """
    rows = []
    for outcome in outcomes:
        ctrl_y = df.loc[df[arm_col] == control, outcome].to_numpy(dtype=float)
        sd_c = ctrl_y.std(ddof=1)
        for arm in treatments:
            est = estimate_ate(df, outcome, arm, arm_col=arm_col, control=control, alpha=alpha)
            mde = minimum_detectable_effect(est.n_treat, est.n_control, sd_c, alpha, power)
            rows.append(
                {
                    "arm": est.arm,
                    "outcome": outcome,
                    "control_mean": est.mean_control,
                    "treat_mean": est.mean_treat,
                    "effect": est.effect,
                    "rel_effect": est.rel_effect,
                    "se": est.se,
                    "ci_low": est.ci_low,
                    "ci_high": est.ci_high,
                    "z": est.z,
                    "p_value": est.p_value,
                    "mde_80": mde,
                    "power": achieved_power(est.effect, est.se, alpha),
                }
            )
    table = pd.DataFrame(rows)
    table["p_holm"] = multipletests(table["p_value"], alpha=alpha, method="holm")[1]
    table["p_bh"] = multipletests(table["p_value"], alpha=alpha, method="fdr_bh")[1]
    return table
