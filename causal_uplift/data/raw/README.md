# Dataset: Hillstrom MineThatData E-Mail Analytics Challenge

A **randomized** e-mail marketing experiment. 64,000 customers who had purchased
in the previous 12 months were randomly assigned to one of three arms; outcomes
were measured over the following two weeks.

| | |
|---|---|
| **File** | `hillstrom_email.csv` (3.96 MB, 64,000 rows × 12 columns, bundled in this repository) |
| **Design** | 1/3 sent the **Men's** e-mail, 1/3 sent the **Women's** e-mail, 1/3 sent **no** e-mail (control). Assignment is randomized. |
| **Source** | Kevin Hillstrom, *The MineThatData E-Mail Analytics And Data Mining Challenge* (2008). Data: <http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv> · Announcement: <https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html> |
| **Retrieved** | 2026-07-10, sha256 `0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece` |
| **License** | No formal license text accompanies the file. Kevin Hillstrom released it publicly for the open challenge and it has circulated for 15+ years as the canonical uplift-modelling benchmark (it ships inside `scikit-uplift` and is used throughout the `causalml`/`econml` literature). It is bundled here for **non-commercial, educational** use with provenance fully attributed. If you are the rights holder and want it removed, open an issue. |

## Columns

**Pre-treatment covariates** (measured *before* the campaign — safe to condition on):

| Column | Meaning |
|---|---|
| `recency` | Months since last purchase |
| `history_segment` | Bucketed prior-year spend (`1) $0 - $100` … `7) $1,000 +`) — a redundant discretisation of `history` |
| `history` | Actual prior-year spend, in dollars (continuous) |
| `mens` | 1 if the customer purchased **men's** merchandise in the past year |
| `womens` | 1 if the customer purchased **women's** merchandise in the past year |
| `zip_code` | Customer area type: `Urban` / `Suburban` / `Rural` (spelled `Surburban` in the raw file — sic) |
| `newbie` | 1 if the customer is new in the past 12 months |
| `channel` | Channels the customer purchased through: `Phone` / `Web` / `Multichannel` |

**Treatment assignment:**

| Column | Meaning |
|---|---|
| `segment` | Randomized arm: `Mens E-Mail`, `Womens E-Mail`, or `No E-Mail` (control) |

**Outcomes** (measured over the two weeks *after* the campaign):

| Column | Meaning |
|---|---|
| `visit` | 1 if the customer visited the website |
| `conversion` | 1 if the customer made a purchase |
| `spend` | Dollars spent (continuous; zero for the ~99% who did not convert) |

## Why this dataset

It is the canonical **randomized** benchmark for uplift / heterogeneous-treatment-
effect modelling: assignment is genuinely random (so treatment effects are
identified without modelling assumptions), it carries pre-treatment covariates
rich enough to make *targeting* a real question, and it is small enough to bundle
directly — clone and run, no accounts, no API keys, no downloads. The clean
randomization is exactly what lets this project contrast an honest experimental
analysis with the observational caveats of the sibling `demand_forecasting`
promotion-lift study.

## Citation

Hillstrom, K. (2008). *The MineThatData E-Mail Analytics And Data Mining
Challenge.* MineThatData.
<https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html>
