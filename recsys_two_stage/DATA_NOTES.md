# Data notes: EDA findings and the design decisions they forced

The exploratory pass on RetailRocket drove several decisions that shaped the whole
project — most consequentially the switch from user-based to session-based
evaluation. This documents what was found and what each finding forced, in the
order the decisions had to be made.

All figures below are reproducible with `reclab eda`; the numbers in §1–§2 come
from the raw event log before any filtering.

## 1. The prediction target: all event types, because it barely matters

RetailRocket records three event types over 137 days (2015-05-03 to 2015-09-18):

| Event | Count | Share |
|---|---|---|
| view | 2,664,312 | 96.7% |
| addtocart | 69,332 | 2.5% |
| transaction | 22,457 | 0.8% |

The obvious commercial instinct is to predict *transactions* — that is what a shop
earns on. The data forbids it: after a temporal split, a 14-day test window
contains only **1,857 transaction pairs from 1,073 buyers**. There is nothing to
evaluate on. Transaction-only modelling is dead on arrival here.

The next instinct is to weight the event types. But the choice turns out to be
nearly free: taking views alone yields 2,132,127 distinct visitor-item pairs;
taking all three yields 2,145,179 — a **0.6% difference**, because a purchase is
almost always preceded by a view of the same item, and views already dominate at
99.4% of the signal. So all three event types are treated as implicit positives,
and the "which event is the target" question that looks important is, on this data,
not. The one place the distinction earns its keep is confidence weighting (§7).

## 2. Users do not persist — 79.6% appear exactly once

The finding that reshaped the project. `visitorid`s are cookies, not accounts:

- **79.6% of visitors view exactly one item, ever.** Median distinct items per
  visitor: **1**. Even the 99th percentile is only 8.
- Mean **1.25 sessions per visitor** — most people show up once and never return.

A user-based next-item protocol needs users with history *before* a cutoff and
activity *after* it. Run honestly — split by time, fit the k-core filter on the
training window alone — that leaves roughly **1,000–2,000 evaluable users**, and
the only filter setting reaching ~2,000 pushes the catalogue past EASE's memory
budget (§4). Too thin to conclude anything from, and concluding things is the
entire point.

## 3. The pivot: the session is the unit that carries signal

A recommender does not need repeat *customers*; it needs **co-occurrence** —
evidence that people who viewed A also viewed B — and that lives *inside* a single
session. A visitor who arrives once, views four items and never returns has still
revealed how those four items relate.

Reframed around the session (30-minute inactivity gap, the session-based
convention), the numbers become workable:

| | Value |
|---|---|
| Total sessions | 1,761,675 |
| Sessions with ≥2 distinct items | 263,899 (15.0%) |
| Mean items per multi-item session | ~3.2 |
| **Evaluable test sessions** (28-day window) | **21,053** |

This is not a workaround to a limitation — it is the right question for this data,
and it matches the session-based recommendation literature (Ludewig & Jannach
2018). It also predicts a specific result: because next-item behaviour is driven by
what you *just* did rather than what is globally popular, a popularity baseline
should be weak here — which it is (§ README, HR@20 ≈ 2.6%), the opposite of its
often-competitive showing in user-based benchmarks.

A pre-registered viability bar was set *before* any model was run: a personalised
model must beat popularity on NDCG@20 with non-overlapping bootstrap CIs, or
RetailRocket is the wrong dataset. It passed at **37×** (popularity 0.009,
ItemKNN 0.319).

The protocol predicts each test session's **last item from its preceding items** —
the standard session-based next-item task. One honest consequence: with a single
held-out target per session, Recall@k collapses to HitRate@k, so the reported
metric set is HitRate / NDCG / MRR rather than Recall.

## 4. Filter thresholds, and the memory wall that sets them

Iterative k-core filtering (drop sessions with <2 items and items in <N sessions,
repeat to a fixed point) trades data for density. The upper bound on catalogue
size is not a modelling preference — it is **EASE's memory**, which grows with the
square of the item count:

| Catalogue | Dense item×item matrix (float64) | Feasible on a laptop? |
|---|---|---|
| 10,000 | 0.8 GB | comfortable |
| ~13,750 (chosen) | 1.5 GB | comfortable |
| 20,000 | 3.2 GB | workable |
| 40,000 | 12.8 GB | marginal |
| 235,061 (unfiltered) | 442 GB | impossible |

`min_item_sessions = 10` (with `min_session_items = 2`) lands the training
catalogue at **13,754 items / 124,638 sessions / 396,008 interactions** — dense
enough to model, small enough that EASE runs, and large enough that the evaluation
is not thin. That the benchmark-winning model is what *forces* us to discard data
is not an inconvenience to hide; it is the project's thesis appearing a stage
early, and it is measured rather than asserted (see the catalogue-scaling sweep
planned for Stage 3).

## 5. The filter is fit on the training window only

Fitting the k-core on the *whole* dataset — which most published preprocessing
does — lets test-period activity decide which items existed during training. That
is a subtle leak: an item survives the filter because of views that happen after
the cutoff, which the model could not have known about. So the filter is fit on
the training window alone, and its item vocabulary is applied to both sides. Test
interactions involving items absent from training are dropped — those are the
cold-start question, deferred to Stage 2. The leaky `--filter-scope global`
variant is retained so the difference can be reported, not because it is
defensible.

## 6. Time span, activity, and the split cutoff

Daily event volume is stable (median ~20,600) across the 137 days, except the
final three days, which decline sharply (11,495 / 10,128 / 1,528 vs the median) —
consistent with partial collection at the end of the window. A 28-day test window
places the cutoff at 2015-08-21, well clear of that tail. 28 days also gives the
largest evaluable-session count among the windows tried (7/14/21/28) without the
cutoff biting into the low-volume tail.

Sessions that straddle the cutoff (start before, end after) are **dropped, not
truncated** — assigning their post-cutoff items to training would leak the future,
and truncating them would evaluate on a session the model half-saw. Only a handful
straddle a single instant against a 28-day window, so dropping them costs nothing
and keeps the leakage guarantee exact.

## 7. Repeat views carry signal — which is what confidence weighting is for

Within a session, 85.6% of visitor-item view pairs occur once, but the repeat
count is not noise:

| Times an item was viewed | P(it was later purchased) |
|---|---|
| 1 | 0.4% |
| 3 | 4.9% |
| 6+ | 12.4% |

A 30× gradient. Binarising the interaction matrix throws this away; ALS's
confidence weighting (`c = 1 + α · count`, Hu et al. 2008) is exactly the mechanism
that uses it, so the training matrix passes the counts through rather than
collapsing them to 0/1.

## 8. Data quirks handled at load time

- **460 exact-duplicate events** (identical timestamp, visitor, item, event) are
  dropped on load.
- **`transactionid` is null for 99.2% of rows** — it is populated only on
  transaction events (22,457 of them, exactly matching the transaction count). Not
  used as a signal here.
- **Timestamps are millisecond Unix epoch**; validated on load to fall within the
  published collection window, so a wrong or corrupted file fails loudly.

## 9. What is not in this data

Stated plainly, because it bounds what the project can claim:

- **No prices.** Revenue-weighted ranking and value models are impossible; this is
  the main reason the Stage 3 reranker's feature set is thin.
- **No user demographics or identities that persist** — the pivot in §2.
- **Item properties are hashed.** Categories are usable (integer ids), but property
  *values* are anonymised tokens with no readable text, so there is no natural
  language to encode — which rules out pretrained text encoders in Stage 2.
- **One retailer, one 4.5-month window.** No seasonality beyond the window, no
  cross-market generalisation.
