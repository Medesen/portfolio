"""Promotion-lift estimation via fixed-effects panel regression.

**Main specification — PPML.** Poisson pseudo-maximum-likelihood regression of
quantity on the promo flag with SKU, weekday, month and year fixed effects
plus holiday indicators, standard errors clustered by SKU. PPML is the
appropriate estimator for count outcomes with zeros (no ad-hoc log transform;
zeros enter naturally) and its coefficients are exact multiplicative effects:
``exp(beta) - 1`` is the promo lift on expected units sold. Consistency
requires only the conditional-mean specification, not the Poisson variance
assumption (Gourieroux, Monfort & Trognon 1984; Santos Silva & Tenreyro 2006).

**Robustness check — log1p-OLS.** The same specification on ``log1p(qty)``.
Its coefficient is only an approximate percentage effect (exact only for
qty >> 1), which is why it is the check and not the headline.

**Identification, stated plainly.** Promotions are scheduled by the retailer,
not randomized. The fixed effects remove *stable* SKU attractiveness and
*calendar* demand patterns, so the promo coefficient is identified from
within-SKU timing variation. The remaining assumption is that promo timing is
not systematically aligned with residual demand shocks (e.g. promos triggered
by anticipated local demand spikes beyond calendar patterns). That is
plausible for supermarket promo calendars, which are planned weeks ahead, but
it is an assumption — this estimate is a well-controlled association, not a
randomized-experiment effect. Note also that the 15 perma-promo SKUs (all
brand B2, promo on >90% of days) carry almost no within-SKU variation and
therefore contribute ~nothing to identification.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from demandcast.data.calendar import calendar_features

FORMULA_PPML = (
    "qty ~ promo + C(sku) + C(dayofweek) + C(month) + C(year)"
    " + is_holiday + is_holiday_eve"
)
FORMULA_OLS = FORMULA_PPML.replace("qty ~", "log1p_qty ~")


@dataclass(frozen=True)
class LiftEstimate:
    """One promo coefficient with cluster-robust uncertainty, in both raw
    (log-point) and percentage-lift form."""

    estimator: str
    coef: float
    se: float
    ci_low: float
    ci_high: float
    n_obs: int
    n_skus: int

    @property
    def lift_pct(self) -> float:
        return float(np.expm1(self.coef) * 100)

    @property
    def lift_ci_pct(self) -> tuple[float, float]:
        return float(np.expm1(self.ci_low) * 100), float(np.expm1(self.ci_high) * 100)

    def __str__(self) -> str:
        lo, hi = self.lift_ci_pct
        return (
            f"{self.estimator}: promo lift {self.lift_pct:+.1f}% "
            f"(95% CI [{lo:+.1f}%, {hi:+.1f}%], "
            f"beta={self.coef:.4f}, cluster-robust se={self.se:.4f}, "
            f"n={self.n_obs:,}, skus={self.n_skus})"
        )


@dataclass(frozen=True)
class PromoLiftResult:
    ppml: LiftEstimate
    ols_log1p: LiftEstimate
    by_brand: pd.DataFrame = field(repr=False)  # PPML per brand


def _prepare(long: pd.DataFrame) -> pd.DataFrame:
    df = long.copy()
    cal = calendar_features(df["date"])
    df["dayofweek"] = cal["dayofweek"]
    df["month"] = cal["month"]
    df["year"] = df["date"].dt.year
    df["is_holiday"] = cal["is_holiday"]
    df["is_holiday_eve"] = cal["is_holiday_eve"]
    df["log1p_qty"] = np.log1p(df["qty"])
    return df


def _extract(res, estimator: str, df: pd.DataFrame) -> LiftEstimate:
    ci = res.conf_int().loc["promo"]
    return LiftEstimate(
        estimator=estimator,
        coef=float(res.params["promo"]),
        se=float(res.bse["promo"]),
        ci_low=float(ci[0]),
        ci_high=float(ci[1]),
        n_obs=len(df),
        n_skus=df["sku"].nunique(),
    )


def _fit_ppml(df: pd.DataFrame) -> LiftEstimate:
    res = smf.poisson(FORMULA_PPML, data=df).fit(disp=False, **_cov_kwargs(df))
    return _extract(res, "PPML", df)


def _cov_kwargs(df: pd.DataFrame) -> dict:
    """Cluster by SKU; with a single cluster (degenerate) fall back to HC1."""
    if df["sku"].nunique() >= 2:
        return {"cov_type": "cluster", "cov_kwds": {"groups": df["sku"]}}
    return {"cov_type": "HC1"}


def estimate_promo_lift(long: pd.DataFrame) -> PromoLiftResult:
    """Estimate the average promo lift, overall and per brand.

    Brand-level regressions cluster SEs by SKU with few clusters (42/45/21/10
    for B1–B4). Cluster-robust variance is asymptotic in the number of
    clusters and tends to under-cover below ~30, so the B3 and especially B4
    intervals are approximate.
    """
    df = _prepare(long)

    ppml = _fit_ppml(df)

    ols_res = smf.ols(FORMULA_OLS, data=df).fit(**_cov_kwargs(df))
    ols = _extract(ols_res, "OLS log1p", df)

    brand_rows = []
    for brand, bdf in df.groupby("brand"):
        est = _fit_ppml(bdf)
        lo, hi = est.lift_ci_pct
        brand_rows.append(
            {
                "brand": brand,
                "lift_pct": est.lift_pct,
                "ci_low_pct": lo,
                "ci_high_pct": hi,
                "n_skus": est.n_skus,
                "promo_share": float(bdf["promo"].mean()),
                # Cluster-robust variance is asymptotic in the number of
                # clusters and tends to under-cover below ~30, so intervals
                # from few-cluster brands are flagged as approximate right in
                # the emitted table, not only in prose.
                "ci_approx_few_clusters": bool(est.n_skus < 30),
            }
        )
    by_brand = pd.DataFrame(brand_rows).set_index("brand")

    return PromoLiftResult(ppml=ppml, ols_log1p=ols, by_brand=by_brand)
