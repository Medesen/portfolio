# Data notes: EDA findings and the design decisions they forced

Findings from the initial data-quality assessment of the pasta dataset
(118 SKUs × 1,798 trading days, 2014-01-02 → 2018-12-31), and what each one
changes about the modelling and evaluation design. Kept in the repo because
several downstream choices are only defensible with this context.

## 1. The 27 missing dates are store closures, not data errors

The calendar has 27 gaps in 5 years. Every one falls on an Italian public
holiday: New Year's Day (×4), Easter Monday (moving date, ×5), Liberation Day,
May Day (×4), Republic Day, Ferragosto (×4), All Saints', Immaculate
Conception, Christmas (×5), St. Stephen's.

**Decision:** the series stays on the observed *trading-day* grid. Lags are
trading-day lags. No zero-filling of closure days (a zero would be a fake
observation of demand that was never offered). Calendar features carry
`is_holiday` / `is_holiday_eve` so models can see closures coming.

## 2. This is low-volume count data with real intermittency

Median SKU sells ~3 units/day; the median SKU has zeros on 20% of days;
10 of 118 SKUs are zero on more than half of all days (worst: 76%).
No SKU is dead, none enters late or exits early — all 118 are kept.

**Decisions:**
- **sMAPE is out, WAPE is in.** With this many zero-actuals, sMAPE is
  undefined/explosive on exactly the days that matter. Metrics: **MASE**
  (scale-free, seasonal-naive-normalized, the headline), **RMSE**, **WAPE**.
- **LightGBM uses a count-appropriate objective.** The ablation (12 rolling
  folds × 28 trading days, all 118 SKUs, point forecast only; reproduce with
  `make backtest ARGS="--model lgbm --objective poisson"` / `l2`):

  | Objective | MASE | WAPE | RMSE | MASE, low-volume tercile |
  |---|---|---|---|---|
  | **Tweedie (power 1.2)** | **0.652** | **0.644** | 5.23 | **0.64** |
  | Poisson | 0.658 | 0.646 | 5.19 | 0.65 |
  | plain L2 | 0.725 | 0.677 | **5.13** | 0.81 |

  Tweedie and Poisson are nearly tied. Plain L2 posts the best RMSE — a
  metric fast movers dominate — while degrading sharply in scale-free terms
  on the slow movers (low-tercile MASE 0.81 vs 0.64), which is exactly the
  regime the count objectives exist for. Tweedie stays the default.
- **SARIMAX is fit on high-volume, low-intermittency SKUs** where a Gaussian
  ARMA approximation is defensible, and the writeup says so explicitly —
  fitting SARIMAX to a 76%-zeros count series and calling the result a fair
  comparison would be methodological theatre.

## 3. Scale heterogeneity is large (≈60× between smallest and largest SKU)

Mean daily quantity ranges 0.39 → 22.9 across SKUs.

**Decision:** cross-SKU aggregation of errors happens only in scale-free terms
(MASE, WAPE); results are additionally reported by volume tercile so the
global model can't hide poor performance on slow movers behind good
performance on fast movers.

## 4. Promotion structure is strongly brand-patterned

37.6% of SKU-days are promo days overall, but the distribution is extreme:
median SKU 20%, while **all 15 SKUs with promo_share > 0.9 belong to brand
B2** (near-permanent promotion), and the 4 SKUs with promo_share < 5% are
also B2. A pooled naive comparison shows promo-day mean ≈ 3.5× non-promo mean
(per-SKU median ≈ 5.9×) — but that number confounds SKU mix and is exactly
what the fixed-effects regression is there to correct.

**Decisions:**
- Promo-lift estimation uses SKU fixed effects (within-SKU contrast only);
  perma-promo SKUs contribute ~no identifying variation and effectively
  self-exclude. Lift is additionally reported by brand.
- The promo flag is a *treatment the retailer chose*, not a randomized one;
  the writeup states the identification assumptions plainly.

## 5. Seasonality: weekly strong, yearly mild

Clear weekly cycle (Sat peak 5.7, Sun trough 2.7 — mean units across SKUs);
yearly pattern is mild with an August dip (Ferragosto) and December peak.

**Decision:** seasonal-naive baseline uses period m = 7 (trading-day weeks);
weekly seasonality is the pattern every model must beat, yearly effects enter
via calendar features rather than long seasonal orders.
