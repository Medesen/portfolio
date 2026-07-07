import numpy as np
import pandas as pd
import pytest

from demandcast.analysis.promo_lift import estimate_promo_lift


@pytest.fixture(scope="module")
def synthetic_panel():
    """Poisson panel with known multiplicative promo effect of 1.5 (beta=log 1.5).

    SKU base rates differ 10x and promo probability differs by SKU, so a
    pooled naive comparison would be badly confounded — the FE estimator must
    still recover the true effect.
    """
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=400, freq="D")
    true_beta = np.log(1.5)
    rows = []
    for sku, base, p_promo in [("A_1", 2.0, 0.1), ("A_2", 20.0, 0.5), ("B_1", 8.0, 0.3)]:
        promo = (rng.random(len(dates)) < p_promo).astype(int)
        dow_effect = 1 + 0.2 * np.sin(2 * np.pi * dates.dayofweek / 7)
        lam = base * dow_effect * np.exp(true_beta * promo)
        qty = rng.poisson(lam)
        brand = sku.split("_")[0]
        for d, q, p in zip(dates, qty, promo):
            rows.append({"date": d, "sku": sku, "brand": brand, "qty": float(q), "promo": p})
    return pd.DataFrame(rows), true_beta


def test_ppml_recovers_known_lift(synthetic_panel):
    long, true_beta = synthetic_panel
    result = estimate_promo_lift(long)
    # true lift is +50%; PPML should land close with a covering CI
    assert result.ppml.coef == pytest.approx(true_beta, abs=0.05)
    lo, hi = result.ppml.lift_ci_pct
    assert lo < 50 < hi


def test_ols_robustness_same_direction(synthetic_panel):
    long, _ = synthetic_panel
    result = estimate_promo_lift(long)
    # log1p-OLS is approximate on low counts; only direction/magnitude class
    assert result.ols_log1p.coef > 0.2


def test_by_brand_table_shape(synthetic_panel):
    long, _ = synthetic_panel
    result = estimate_promo_lift(long)
    assert set(result.by_brand.index) == {"A", "B"}
    assert (result.by_brand["n_skus"] == [2, 1]).all()
