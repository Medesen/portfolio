# Data notes: EDA findings and the design decisions they forced

Findings from the initial assessment of the Hillstrom e-mail experiment (64,000
customers, three randomized arms), and what each one changes about the analysis.
Kept in the repo because several downstream choices are only defensible with this
context.

## 1. Randomization holds — cleanly

Arm sizes are ~equal (Mens 21,307 / Womens 21,387 / Control 21,306) and every
pre-treatment covariate is balanced against control: the largest standardized
mean difference anywhere is **0.014** (`zip_code=Rural`, Mens arm), far inside the
conventional 0.1 concern threshold.

**Decision:** the average treatment effect is identified by a plain **difference
in means** — no propensity model, no outcome model, no identification assumptions
beyond the randomization itself. Everything else in the project (CUPED,
regression adjustment, uplift models) is there for *precision* or *targeting*, not
for identification. This is the deliberate contrast with the sibling
`demand_forecasting` promo-lift study, where the treatment was *chosen* by the
retailer and the write-up has to argue identification.

## 2. The outcome is a steep, zero-inflated funnel

Response rates fall by more than an order of magnitude down the funnel, and the
money outcome is almost all zeros:

| Outcome | Control | Mens | Womens | Nonzero share |
|---|---|---|---|---|
| `visit` | 10.6% | 18.3% | 15.1% | 14.7% |
| `conversion` | 0.57% | 1.25% | 0.88% | 0.90% |
| `spend` ($) | 0.65 | 1.42 | 1.08 | 0.90% |

**Decisions:**
- **Report all three outcomes, but treat them differently.** `visit` is the
  statistically powered outcome (effects at z ≈ 14–23); `spend` is the outcome the
  business cares about but is dominated by the ~99% zeros, so its effects are real
  but noisy (z ≈ 3–5) and are always shown *with* their wide CIs.
- **Unpooled (Welch) standard errors everywhere** — appropriate for both the
  binary rates and the heavy-tailed spend, and honest about the differing arm
  variances.
- **Uplift models target `visit`.** Fitting an individual-effect model on a 0.9%
  base rate (conversion/spend) is mostly fitting noise; `visit` gives a stable
  ranking that is then *translated* into revenue via an experiment-wide
  dollars-per-visit figure, rather than modelled on the noisy spend directly.

## 3. Men's e-mail has the bigger effect; women's e-mail has the bigger *heterogeneity*

The men's e-mail wins on the average (visit lift +7.7pp vs +4.5pp for women's).
But the *interesting* structure is in the women's arm. Interacting the treatment
with prior purchase history (a pre-treatment covariate) on `visit`:

- women's e-mail × prior **women's** buyer: **+0.062** (p < 0.001)
- women's e-mail × prior **men's** buyer: **−0.052** (p < 0.001)

i.e. the women's e-mail strongly helps customers who previously bought women's
merchandise and is actively counter-productive for prior men's buyers. The men's
arm shows only weak heterogeneity by comparison.

**Decision:** the **primary uplift contrast is Women's-vs-control**, not because
it is the bigger effect but because it is the bigger *targeting problem* — there
is a real sign-flip for the model to find (whom to mail, and whom to leave alone).
A uplift model on the near-homogeneous men's effect would have little to learn.

## 4. Pre-period covariates barely predict a two-week response

The covariates are pre-treatment and safe to condition on, but they are weak
predictors of what someone does in the next fortnight. Full-covariate R² on the
outcome: **`visit` 2.7%**, `conversion` 0.2%, `spend` 0.1%. The single obvious
CUPED covariate — `history` (prior-year spend) — correlates with `visit` at only
ρ ≈ 0.065.

**Decision:** CUPED and regression adjustment are included and reported
**honestly**. On `visit`, multi-covariate regression adjustment buys ~2.9%
variance reduction (≈ 1.03× effective sample size); on `spend`/`conversion` it
buys essentially nothing — because variance reduction is bounded by the squared
covariate–outcome correlation, and here that is near zero. The *mechanism* is
demonstrated where it can be controlled — on synthetic data, CUPED variance
reduction tracks ρ² exactly (see `tests/test_adjustment.py`). This is the honest
version of a technique that is often cargo-culted: it is a covariance, not magic,
and this dataset is where it politely declines to help.

## 5. Data quirks handled at load time

- `zip_code` is spelled **`Surburban`** in the raw file; normalized to `Suburban`
  in the loader (documented, non-destructive).
- `history_segment` is a redundant bucketing of the continuous `history` and is
  **dropped from the feature set** (kept in the raw frame for reference).
- Integrity checks fail loudly if any `conversion` has non-positive `spend` or
  occurs without a `visit`, and if any covariate that should be binary is not.
