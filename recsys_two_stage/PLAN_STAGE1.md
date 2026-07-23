# Stage 1 Build Plan — Two-Stage Recommender Systems Project

**Status:** proposed, not started
**Audience:** a developer picking this up cold, with no prior context on the project
**Directory name (`recsys_two_stage/`) is provisional** and can be renamed freely before work starts.

---

## 0. What this project is, and why it exists

This is the sixth project in a public ML portfolio. The other five (`rag_pipeline`,
`end2end_churn`, `config_driven_ml`, `demand_forecasting`, `causal_uplift`) share a
consistent house style: bundled data, Docker, a `make reproduce` that regenerates
every number in the README, a `make check-readme` that *verifies* those numbers,
comprehensive tests, and — most importantly — an **honest headline**, usually a
negative or counterintuitive result that changed a decision.

This project applies that stance to recommender systems, a field that had a public
reproducibility reckoning worth building on:

- **Dacrema, Cremonesi & Jannach (2019)**, *"Are we really making much progress?"* —
  of 18 neural recommenders from top venues, most lost to properly tuned classical
  baselines (ItemKNN, SLIM, EASE).
- **Rendle et al. (2020)**, *Neural Collaborative Filtering vs. Matrix Factorization
  Revisited* — a learned MLP similarity loses to a plain dot product.
- **Krichene & Rendle (2020)**, *On Sampled Metrics for Item Recommendation* —
  evaluating against ~100 sampled negatives (the field's long-standing default)
  produces model rankings that **disagree with full-catalogue ranking**. Published
  leaderboards were partly an artifact of the sampling shortcut.

**The project's thesis:**

> A recommender is a two-stage system whose reported quality depends more on the
> evaluation protocol than on the model.

**The intended final headline** (to be confirmed empirically, not assumed):

> EASE may well win on accuracy. It also cannot run on a real catalogue — its memory
> grows with the square of the number of items — and it cannot recommend a product it
> has never seen co-purchased. The two-tower model scores *worse on the benchmark* and
> is nevertheless what industry deploys, because it scales linearly and handles new
> items. The benchmark and the business optimise different things.

That conclusion needs Stages 2 and 3 to land. **Stage 1 builds the foundation it
rests on: the data, the honest evaluation harness, and the classical baselines.**

### The three-stage roadmap (for context; only Stage 1 is specified here)

| Stage | Contents |
|---|---|
| **1 (this document)** | Data acquisition & bundling, EDA/DATA_NOTES, temporal split harness, full-catalogue **and** sampled metrics, beyond-accuracy metrics, four classical models (Popularity, ItemKNN, EASE, ALS), tuning protocol, tests, README |
| 2 | Two-tower retrieval model, SASRec sequential model, retrieval-ceiling analysis, cold-start evaluation using item content features |
| 3 | LightGBM reranker (second stage), approximate-nearest-neighbour index with a measured recall-vs-latency curve, minimal `/recommend` FastAPI endpoint with per-stage timings |

**Stage 1 must be shippable on its own.** At the end of Stage 1 the README is
complete and honest — it describes an evaluation study of classical recommenders,
with the two-stage/neural work named in "Future Work" rather than left visibly
half-built.

---

## 1. Scope of Stage 1

### In scope

1. Acquire RetailRocket, bundle it in the repo, document provenance and licence.
2. Distil the large item-properties files into a small bundled item table.
3. EDA → `DATA_NOTES.md`, which resolves two open design decisions (below).
4. Loading + iterative k-core filtering, derived from the training window only.
5. Global temporal split, plus a leave-one-out comparator.
6. Metrics: full-catalogue ranking, sampled-negatives ranking, beyond-accuracy.
7. Four models: Popularity, ItemKNN, EASE, ALS — all properly tuned.
8. Tuning protocol on a temporal validation window carved from training.
9. CLI, Makefile, Dockerfile, `make reproduce`, `make check-readme`.
10. Test suite.
11. `README.md`, `DATA_NOTES.md`, `data/raw/README.md`.

### Explicitly out of scope for Stage 1

Two-tower, SASRec, reranker, ANN index, serving endpoint, cold-start evaluation,
content features in models (the item table is *bundled* in Stage 1 but only *used*
in Stage 2).

### Two decisions Stage 1 must make from the data, not from assumption

These are deliberately unresolved. They get decided during Phase A and written up in
`DATA_NOTES.md` with the reasoning on record.

**Decision 1 — which event type is the prediction target.** RetailRocket records
`view` (~96%), `addtocart` (~2.5%), `transaction` (~1%). Views are plentiful but weak
signal; transactions are what the business cares about but may be too sparse to
model. Options: views only; views+addtocart as implicit positives; transactions only;
or a weighted scheme. Decide from the observed counts after filtering, and state the
commercial reasoning.

**Decision 2 — the k-core filter thresholds.** Aggressive filtering yields denser,
easier, less realistic data. The catalogue size is additionally constrained by EASE
(see §7.3). Choose thresholds that leave roughly **15,000–20,000 items**, and justify
them from the EDA rather than defaulting to "5-core because everyone does".

---

## 2. Repository layout

Mirrors `causal_uplift/` and `demand_forecasting/` exactly.

```
recsys_two_stage/
├── README.md                       # Written LAST, in Phase I
├── DATA_NOTES.md                   # EDA findings → design decisions (Phase A)
├── PLAN_STAGE1.md                  # This file
├── pyproject.toml
├── requirements.lock               # pip-compile output, incl. dev extra
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── setup.sh / setup.ps1
├── .gitignore                      # __pycache__/ *.pyc .venv/ outputs/ .pytest_cache/ *.egg-info/
├── data/
│   ├── raw/
│   │   ├── events.csv.gz           # bundled, full, uncut (~15–25 MB)
│   │   ├── item_properties.csv.gz  # bundled, DISTILLED (see §4.2)
│   │   ├── category_tree.csv.gz    # bundled, tiny
│   │   └── README.md               # provenance, licence, checksums, column docs
│   └── prepare_item_properties.py  # one-off distillation script (documented, not run in CI)
├── scripts/
│   └── check_readme.py             # stdlib only, no project deps
├── src/reclab/
│   ├── __init__.py
│   ├── main.py                     # CLI entry point
│   ├── data/
│   │   ├── __init__.py
│   │   ├── load.py                 # loading + validation
│   │   ├── filtering.py            # iterative k-core
│   │   └── synthetic.py            # generators for tests
│   ├── splitting/
│   │   ├── __init__.py
│   │   └── protocols.py            # temporal split + leave-one-out comparator
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py              # recall / ndcg / mrr / hitrate primitives
│   │   ├── full_catalogue.py       # honest evaluation loop
│   │   ├── sampled.py              # sampled-negatives evaluation loop
│   │   └── beyond_accuracy.py      # coverage, Gini, popularity bias
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py                 # Recommender protocol
│   │   ├── popularity.py
│   │   ├── itemknn.py
│   │   ├── ease.py
│   │   └── als.py
│   └── tuning/
│       ├── __init__.py
│       └── grid.py                 # validation-window grid search
├── tests/
│   ├── test_data.py
│   ├── test_splitting.py
│   ├── test_metrics.py
│   ├── test_sampled_bias.py
│   ├── test_models.py
│   └── test_tuning.py
└── outputs/                        # gitignored, created host-side by make targets
```

**Package name `reclab`; CLI command `reclab`.** (Precedent: `upliftlab`,
`demandcast`, `mlctl`.)

---

## 3. Conventions to match

Copy these from `causal_uplift/` — do not reinvent:

- **Dockerfile:** `python:3.12-slim`; non-root user created from build args `UID`/`GID`
  (default 1000) so files written to the mounted `outputs/` are host-owned; deps
  installed from `requirements.lock` *before* the source copy so code edits don't
  invalidate the dependency layer; `ENTRYPOINT ["reclab"]`. Install `libgomp1`
  (LightGBM/OpenMP) only if a Stage 1 dependency needs it — otherwise omit.
- **docker-compose.yml:** single service, `./outputs:/app/outputs` bind mount, UID/GID
  build args.
- **Makefile:** `help` target using the `## comment` grep pattern; an `outputs:` target
  that `mkdir -p outputs` host-side (the daemon otherwise creates it root-owned);
  every run target depends on `outputs`.
- **requirements.lock:** generated by
  `pip-compile --extra=dev --output-file=requirements.lock pyproject.toml`.
- **`scripts/check_readme.py`:** stdlib only, parses headline numbers out of README
  markdown and compares against `outputs/*.csv` within rounding tolerance, exits
  non-zero listing mismatches.
- **CLI:** `argparse` with subcommands, each writing a CSV to `--out` (default
  `outputs/`) and printing a human-readable summary. An `all` subcommand runs
  everything and reproduces the README.
- **Type hints throughout; `from __future__ import annotations`.**
- Comments explain *why*, not *what* — matching the density in `upliftlab/main.py`.

### Makefile targets for Stage 1

| Target | Purpose |
|---|---|
| `setup` | `./setup.sh` — build image, run a quick smoke evaluation |
| `build` | `docker compose build` |
| `eda` | Dataset summary statistics feeding DATA_NOTES |
| `tune` | Grid search on the validation window (slowest step) |
| `evaluate` | Full-catalogue evaluation of all four models |
| `sampled` | Sampled-negatives evaluation + the disagreement table |
| `protocols` | Temporal vs leave-one-out comparison |
| `beyond` | Coverage / Gini / popularity-bias metrics |
| `reproduce` | Everything, end to end |
| `check-readme` | `python3 scripts/check_readme.py` |
| `test` | `docker compose run --rm --entrypoint pytest reclab tests/ -v` |
| `shell`, `clean` | As in `causal_uplift` |

---

## 4. Phase A(i) — Data acquisition and bundling

### 4.1 Source

RetailRocket e-commerce dataset, published on Kaggle
(`retailrocket/ecommerce-dataset`). Four files:

| File | Approx. size | Contents |
|---|---|---|
| `events.csv` | ~92 MB | `timestamp` (ms epoch), `visitorid`, `event` (`view`/`addtocart`/`transaction`), `itemid`, `transactionid` (mostly null). ~2.76M rows, ~1.4M visitors, ~235k items, spanning roughly 2015-05-03 to 2015-09-18 |
| `item_properties_part1.csv`, `part2.csv` | ~470 MB combined | `timestamp`, `itemid`, `property`, `value`. Timestamped snapshots of every property change. Values are hashed tokens except numerics, which are prefixed `n`. Properties include `categoryid` and `available` |
| `category_tree.csv` | tiny | `categoryid`, `parentid` |

> **⚠ First task, before anything else: verify the licence directly from the Kaggle
> dataset page.** It is believed to be **CC BY-NC-SA 4.0**, which permits
> redistribution with attribution for non-commercial use — the entire bundling plan
> depends on this. If it turns out to be otherwise, **stop and escalate**; the
> fallback is MovieLens-1M with a download-at-setup step, which changes the project
> materially (no content features, so no cold-start story in Stage 2).

### 4.2 What gets bundled

**`events.csv` → bundled whole, uncut, as `events.csv.gz`.**
The file is almost entirely integers and compresses hard: ~92 MB should become
roughly 15–25 MB. `pandas.read_csv` reads `.gz` natively with no code changes. No
Git LFS, no file chunking, no GitHub size warnings.

Crucially, **filtering happens in code at load time, not by shipping a pre-filtered
file.** This makes the filter an inspectable, testable, adjustable decision that a
reader can change and re-run, rather than a mystery subset baked into the data. This
is a deliberate departure from what most repos do and is worth one sentence in the
README.

**The item-properties files cannot be bundled** at ~470 MB. `data/prepare_item_properties.py`
distils them:

- Keep only rows where `property` is in `{"categoryid", "available"}`.
- Keep the `timestamp` column — do **not** collapse to a single current value.
  Property values change over time, and taking a post-cutoff category snapshot would
  leak future information into training. Downstream code must be able to take an
  as-of-cutoff view.
- Optionally restrict to items that survive filtering, if size demands it. Prefer not
  to — keeping all items leaves Stage 2's cold-start work unconstrained.
- Write `data/raw/item_properties.csv.gz`.

The script is committed and documented but **not** run in CI, because it needs the
raw download. `data/raw/README.md` must record the source URL, retrieval date, and
sha256 of each raw file so the derived table is reproducible in principle. This is a
real, small compromise on the portfolio's "no downloads" promise and should be stated
plainly in `DATA_NOTES.md` rather than glossed over.

`category_tree.csv` is tiny — bundle as-is (gzipped for consistency).

### 4.3 `data/raw/README.md`

Follow the format of `causal_uplift/data/raw/README.md` exactly: a summary table
(file / size / rows / source URL / retrieval date + sha256 / licence), full column
documentation for every bundled file, a "Why this dataset" section, and a citation.

The "Why this dataset" section should say, in substance:

> RetailRocket was chosen over MovieLens — the field's usual benchmark — for three
> reasons. Its licence permits redistribution, so the data ships in the repository
> and the project keeps the portfolio's clone-and-run promise. It is real e-commerce
> behaviour (views, add-to-carts, purchases) with genuine timestamps, so a
> time-based split reflects how a deployed system actually works. And it carries
> product attributes, which is the only reason the cold-start comparison in Stage 2
> is possible at all. The cost is comparability: MovieLens results can be checked
> against published numbers, and these cannot. That trade was made deliberately,
> in favour of realism and a licence that allows the data to be shipped.

---

## 5. Phase A(ii) — EDA → `DATA_NOTES.md`

Follow the established format: `# Data notes: EDA findings and the design decisions
they forced`, then numbered sections, each stating a **finding** and the **decision
it forced**. See `demand_forecasting/DATA_NOTES.md` and
`causal_uplift/DATA_NOTES.md` for tone and length.

Sections to produce (adjust as the data dictates):

1. **Event-type distribution and the choice of prediction target** — resolves
   Decision 1 (§1). Report counts and per-user/per-item distributions for each event
   type; state which is the target and the commercial reasoning.
2. **Interaction sparsity: most visitors appear once** — motivates filtering at all.
   Report the distribution of events per visitor and per item.
3. **Filter thresholds and what they cost** — resolves Decision 2. Show a table of
   candidate threshold pairs against resulting users / items / interactions /
   density, and justify the chosen point. State explicitly that EASE's quadratic
   memory (§7.3) is what sets the upper bound on catalogue size.
4. **The filter is derived from the training window only** — see §6.2. Explain why,
   and quantify how much the numbers move if it is derived from the whole dataset
   (the way most papers do it).
5. **Time span, activity over time, and the split cutoff** — justify the test-window
   length from observed daily volume.
6. **Data quirks handled at load time** — duplicate events, sessions crossing
   midnight, the `transactionid` null pattern, any timestamp anomalies.
7. **What is not in this data** — no user demographics, no prices, no explicit
   ratings, anonymised/hashed item properties. Sets up limitations honestly.

---

## 6. Phase B — Loading, filtering, splitting

### 6.1 `data/load.py`

```python
def load_events(
    path: Path | None = None,
    event_types: tuple[str, ...] = ...,   # from Decision 1
) -> pd.DataFrame
```

Returns columns `user`, `item`, `timestamp` (pandas datetime, UTC), `event`. Validate
on load and raise loudly on: missing columns, null user/item, non-monotonic or
out-of-range timestamps, unexpected event values. Deduplicate exact
(user, item, timestamp, event) repeats and report how many were dropped.

### 6.2 `data/filtering.py`

```python
def k_core_filter(
    df: pd.DataFrame,
    min_user_interactions: int,
    min_item_interactions: int,
    max_iterations: int = 50,
) -> tuple[pd.DataFrame, FilterReport]
```

Iterative: drop users below threshold, then items below threshold, repeat until no
change (a fixed point) or `max_iterations`. `FilterReport` records per-iteration
counts for DATA_NOTES.

**Critical design point — the filter is fit on the training window only.**
Deriving the filter from the full dataset lets test-period activity decide which
items exist during training, which is a subtle leak. Most published work does exactly
this. The correct sequence is:

1. Split by time first, on the raw data.
2. Compute the k-core filter on the **training** interactions alone.
3. Apply the resulting user/item sets to both train and test.
4. Drop test interactions involving users or items that did not survive.

Step 4 discards genuinely new items — which is the cold-start question, deferred to
Stage 2. Note this explicitly in DATA_NOTES.

Provide `--filter-scope {train,global}` on the CLI so the leaky variant can be run
and the difference reported. That comparison is a finding, not just a safety rail.

### 6.3 `splitting/protocols.py`

```python
@dataclass(frozen=True)
class Split:
    train: pd.DataFrame
    test: pd.DataFrame
    cutoff: pd.Timestamp
    eval_users: np.ndarray          # users with >=1 train AND >=1 test interaction
    protocol: str                   # "temporal" | "leave_one_out"

def temporal_split(df, test_days: int = ...) -> Split
def leave_one_out_split(df, seed: int = 0) -> Split
```

- **`temporal_split`** — the headline protocol. Everything before `cutoff` trains;
  everything after tests. Matches what a deployed system knows.
- **`leave_one_out_split`** — the comparator, present *only* to be shown up. Holds
  out one interaction per user at random, which is the field's default and which
  trains on interactions that occurred after other users' held-out items. Same
  models, same metrics, different protocol.

Both must expose an identical interface so the evaluation loop is protocol-agnostic.

---

## 7. Phase C — Models

### 7.0 `models/base.py`

```python
class Recommender(Protocol):
    name: str
    def fit(self, train: sp.csr_matrix) -> Self: ...
    def score_users(self, user_indices: np.ndarray) -> np.ndarray: ...
        # returns (n_users, n_items) dense scores, chunked internally if needed
```

Interactions are held as a sparse CSR user×item matrix built from the training split,
with explicit `user_index` / `item_index` mappings owned by the split object.

**Every model must exclude items the user already interacted with in training**
before ranking. This is done once, centrally, in the evaluation loop — not
per-model — so it cannot be forgotten. A test asserts it.

### 7.1 `popularity.py`

Score = training interaction count per item, identical for every user. The floor:
if a model cannot beat this, it is worth nothing. Expect it to be uncomfortably
competitive, especially on the sampled metrics.

### 7.2 `itemknn.py`

Cosine similarity over the binary user×item matrix, with a shrinkage term
`sim(i,j) = c_ij / (sqrt(c_ii · c_jj) + shrink)`, retaining the top-`k` neighbours per
item. User score for item `j` = sum of similarities from `j` to the items in that
user's training history. Hyperparameters: `k`, `shrink`. This is the baseline that
beat the neural models in Dacrema et al. — it must be genuinely tuned, not
stubbed.

### 7.3 `ease.py`

Steck (2019), *Embarrassingly Shallow Autoencoders*. Closed form:

```
G = XᵀX + λI
P = G⁻¹
B = I − P · diagMat(1 / diag(P))
B[diagonal] = 0
scores = X · B
```

One hyperparameter, `λ`. About fifteen lines. Routinely competitive with far more
complex models.

**Memory is the binding constraint on the whole project.** `G` is items×items, dense:

| Catalogue | float64 `G` | Feasible on a laptop? |
|---|---|---|
| 10,000 | ~0.8 GB | Comfortable |
| 20,000 | ~3.2 GB | Workable, slow |
| 40,000 | ~12.8 GB | Marginal at best |
| 235,000 (unfiltered) | ~440 GB | No |

Use float64 for the inversion (stability), and document the requirement. This
constraint is not an inconvenience to hide — it is **evidence for the project's
central argument** and belongs in the README: the method that wins on accuracy is the
one that cannot scale.

### 7.4 `als.py`

Implicit-feedback alternating least squares (Hu, Koren & Volinsky 2008).
**Recommendation: use the `implicit` library** rather than hand-rolling — it is the
standard, it is fast, and Stage 1's depth signal comes from EASE and the evaluation
harness rather than from reimplementing ALS. If `implicit` causes install friction in
the slim image, a ~50-line NumPy implementation is an acceptable fallback; note
whichever was chosen in the README.

Hyperparameters: `factors`, `regularization`, `iterations`, `alpha`.

### 7.5 `tuning/grid.py`

**Non-negotiable.** The entire premise of Dacrema et al. is that baselines lose only
when under-tuned. Every model gets a real grid search.

Protocol: carve a **temporal** validation window from the end of the training period
(same shape as the train/test split, one level in). Tune on it, select by NDCG@20,
refit on the full training period with the winning settings, evaluate once on test.
The test split is never used for selection.

Write `outputs/tuning_<model>.csv` with every grid point and its validation score, so
the tuning is auditable rather than asserted.

---

## 8. Phase D — Evaluation

### 8.1 `evaluation/metrics.py`

Primitives operating on a ranked item list and a relevant-item set:
`recall_at_k`, `ndcg_at_k`, `mrr`, `hit_rate_at_k`. Report at k ∈ {10, 20, 50}.

**Define Recall@k precisely and document it** — with multiple held-out positives per
user, `|hits| / |relevant|` and `|hits| / min(k, |relevant|)` differ, and papers
disagree. Pick one, state it, test it against hand-computed values.

### 8.2 `evaluation/full_catalogue.py`

The honest loop. For each evaluation user: score all items, mask out training-history
items, rank, compute metrics against that user's held-out items. Chunk over users to
bound memory. Output `outputs/metrics_full.csv` with columns
`model, protocol, k, recall, ndcg, mrr, hit_rate, n_eval_users`.

### 8.3 `evaluation/sampled.py`

The shortcut, implemented in order to be discredited. For each held-out positive,
sample `n_negatives` (default 100) items the user has not interacted with, rank the
positive among them, compute the same metrics.

Support two samplers — `uniform` and `popularity` — because the choice changes the
answer, which is itself part of the point.

Output `outputs/metrics_sampled.csv` plus **the key artifact of Stage 1**:
`outputs/protocol_disagreement.csv`, a table showing each model's rank under
full-catalogue evaluation versus its rank under sampled evaluation, with a rank
correlation. If the orderings disagree — as the literature predicts — that table is
the README's headline.

### 8.4 `evaluation/beyond_accuracy.py`

Accuracy is not the business question. A recommender that shows only bestsellers can
score well and be commercially worthless.

- **Coverage@k** — fraction of the catalogue appearing in at least one user's top-k.
- **Gini coefficient** of recommendation frequency across items — concentration.
- **Mean popularity percentile** of recommended items — popularity bias.

Output `outputs/metrics_beyond.csv`. Intra-list diversity needs item categories and
is deferred to Stage 2.

---

## 9. Phase E — CLI, outputs, reproducibility

### 9.1 Subcommands

```
reclab eda           # dataset summary + filter-threshold table for DATA_NOTES
reclab tune          # grid search on the validation window   [slowest]
reclab evaluate      # full-catalogue evaluation, all models
reclab sampled       # sampled-negatives evaluation + disagreement table
reclab protocols     # temporal vs leave-one-out comparison
reclab beyond        # coverage / Gini / popularity bias
reclab all           # everything, in order — reproduces the README
```

Global options: `--out` (default `outputs/`), `--seed` (default 0),
`--filter-scope {train,global}`, `--test-days`.

### 9.2 Output files

| File | Contents |
|---|---|
| `data_summary.csv` | Pre/post-filter counts, density, date range |
| `filter_thresholds.csv` | Candidate threshold grid → resulting dataset shape |
| `tuning_<model>.csv` | Every grid point with validation score |
| `metrics_full.csv` | Full-catalogue metrics, all models × k |
| `metrics_sampled.csv` | Sampled metrics, all models × k × sampler |
| `protocol_disagreement.csv` | Full vs sampled model rankings + rank correlation |
| `protocol_comparison.csv` | Temporal vs leave-one-out, all models |
| `metrics_beyond.csv` | Coverage, Gini, popularity percentile |

### 9.3 Budgets

- **`make reproduce` under ~20 minutes** on a modern laptop, `make tune` excluded (it
  may be slower; if so, ship the tuned hyperparameters as defaults and have
  `reproduce` use them, with `tune` as a separate, documented target).
- **Peak memory under ~8 GB**, set by EASE. State the requirement in the README
  prerequisites, as the other projects do.

---

## 10. Phase F — Tests

Target 25–35 tests, in the spirit of the existing suites. The ones that matter:

**`test_data.py`**
- k-core filtering reaches a genuine fixed point; a second application is a no-op.
- The filter derived from the training window differs from the global one, and
  training-window filtering never consults post-cutoff rows.
- Loader validation rejects null users, unknown event types, malformed timestamps.

**`test_splitting.py`**
- No test interaction predates the cutoff; train and test are disjoint.
- `eval_users` all have both train and test interactions.
- **Leakage test** (mirroring `demand_forecasting`): corrupt every post-cutoff
  interaction, retrain, assert the trained model artifacts and all training-derived
  statistics are bit-identical.

**`test_metrics.py`**
- Recall / NDCG / MRR / hit-rate against hand-computed toy cases.
- Edge cases: no relevant items, k larger than the catalogue, a user with a single
  held-out item, perfect and worst-possible rankings.

**`test_sampled_bias.py`** — the distinctive one.
- On synthetic data with a known true model ordering, assert that full-catalogue
  evaluation recovers the true order and that sampled evaluation **misranks** it.
  This turns Krichene & Rendle from a citation into a property the repo verifies.

**`test_models.py`**
- **EASE closed form** checked against a brute-force ridge solve on a tiny matrix.
- Items in a user's training history never appear in that user's recommendations.
- Popularity is deterministic and matches hand-counted frequencies.
- ItemKNN similarity is symmetric; shrinkage monotonically damps low-support pairs.
- Every model is reproducible under a fixed seed.

**`test_tuning.py`**
- The validation window is strictly inside the training period and never touches test.
- Grid search selects the best validation score.

---

## 11. Phase G — Documentation

Written **last**, once the numbers exist.

**`README.md`** — follow `causal_uplift/README.md` structure precisely: title with a
one-line framing; Summary; **the honest headline** with result tables; Quick Start
(~N minutes) with prerequisites, one-command setup, "Try It Out", and a no-Docker
local alternative; "What This Project Demonstrates" broken into themes; Testing, with
the two or three tests worth a reviewer's eye called out; Project Structure; Honest
Limitations & Future Work; Licence / Dataset Licence / Citation; Troubleshooting.

Required content beyond the standard shape:

- The dataset-choice rationale from §4.3.
- The EASE memory constraint, framed as evidence rather than apology.
- **Arguments for each deliberate omission**, not a bare list:
  - *LLM-based recommenders* — no settled evaluation methodology; including one would
    dilute a project whose entire point is honest measurement.
  - *Bandits / learning from live traffic* — genuinely important, and genuinely a
    separate project; off-policy evaluation done badly is worse than not done, and
    this dataset carries no logged propensities.
  - *Graph neural networks (LightGCN etc.)* — already covered by the "tuned baselines
    win" literature this project builds on; adding one would restate the finding, not
    extend it.
  - *Kubernetes / monitoring stack* — already demonstrated in the sibling
    `end2end_churn` project; repeating it would add volume, not evidence.
- A Stage 2/3 roadmap so the reader knows what is coming rather than what is missing.

**Portfolio root `README.md`** — add the project as entry 6, matching the existing
entry format (Status / Domain / Key Findings / Tech Stack / prose / Highlights /
link), extend the Repository Structure tree, update "Technologies Used So Far" with
ranking metrics and recommender methods, update the project count and Last Updated
date, and remove the recommender bullet from "Future Projects".

---

## 12. Definition of done for Stage 1

1. `make setup` works from a clean clone on Linux, macOS and Windows.
2. `make reproduce` regenerates every number in the README inside the time budget.
3. `make check-readme` exits zero.
4. `make test` passes; every test in §10 exists.
5. `DATA_NOTES.md` resolves both open decisions with reasoning on record.
6. `data/raw/README.md` documents provenance, licence, checksums and every column.
7. The README's honest headline is backed by `outputs/` and stated plainly, whichever
   way the results fell.
8. The portfolio root README is updated.
9. No API keys, no accounts, no runtime downloads.

---

## 13. Risks and open questions

| Risk | Mitigation |
|---|---|
| **Licence is not what we believe** | Verify first, before any other work. Fallback is MovieLens-1M with download-at-setup, which sacrifices the Stage 2 cold-start story. Escalate rather than proceed. |
| **Filtering leaves too little data** | RetailRocket is sparse; aggressive k-core could collapse it. Explore the threshold grid in Phase A *before* committing to the pipeline. If the data cannot support the study, that is a Phase A finding, not a Phase E surprise. |
| **The expected findings don't materialise** | The sampled-metric disagreement and the temporal-vs-LOO gap are predicted by the literature but not guaranteed on this data. **Report whatever is found.** A well-executed null is on-brand for this portfolio and is explicitly preferable to a massaged result. |
| **EASE exhausts memory** | The catalogue-size ceiling is known in advance (§7.3). Pick filter thresholds accordingly in Phase A. |
| **Tuning dominates runtime** | Ship tuned defaults; keep `tune` as a separate target from `reproduce`. |
| **Views vs transactions is the wrong call** | Decision 1 is data-driven and reversible — the CLI takes `event_types`, so both can be run and compared if it is genuinely close. |

---

## 14. Suggested working order

1. Verify the licence. **Stop here if it fails.**
2. Download raw data; write `prepare_item_properties.py`; produce the bundled files;
   write `data/raw/README.md` with checksums.
3. Scaffold the repo: `pyproject.toml`, Dockerfile, compose, Makefile, setup scripts,
   `.gitignore`, empty package.
4. `data/load.py` + `data/filtering.py` + their tests.
5. `reclab eda`; explore the threshold grid; **write `DATA_NOTES.md` and resolve both
   open decisions.**
6. `splitting/protocols.py` + tests, including the leakage test.
7. `evaluation/metrics.py` + hand-computed tests.
8. Models in ascending order of complexity: Popularity → ItemKNN → EASE → ALS, each
   with its tests as it lands.
9. `tuning/grid.py`; run the grid; record the winners.
10. `evaluation/full_catalogue.py`, then `sampled.py`, then `beyond_accuracy.py`.
11. `test_sampled_bias.py` — the synthetic misranking demonstration.
12. Wire up `main.py` and all Makefile targets; verify `reproduce` end to end.
13. Write the README from the actual outputs; write `scripts/check_readme.py`;
    run `make check-readme` until it is green.
14. Update the portfolio root README.

Work on a branch off the portfolio repo's default branch; do not commit to the
default branch directly.
