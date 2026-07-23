# Stage 3 Build Plan — Ranking, Approximate Search, and Serving

**Status:** proposed, not started
**Prerequisites:** Stages 1 and 2 complete and merged
**Audience:** a developer picking this up cold

---

## 0. What Stage 3 is for

Stages 1 and 2 built an evaluation study of *retrieval*: eight models, honestly
measured, plus a retrieval-ceiling analysis showing how often the right item appears
anywhere in a candidate set of a given size.

Stage 3 builds the second half of the actual architecture and closes the project's
argument:

1. **A LightGBM reranker** that reorders the retrieved candidates using features
   retrieval cannot see — turning the repo from a model comparison into a working
   two-stage system.
2. **An approximate-nearest-neighbour index** with a measured recall-versus-latency
   curve, showing what the standard serving shortcut actually costs.
3. **A minimal `/recommend` endpoint** that composes retrieval → ranking → filtering,
   with latency measured *per stage*.

### The closing argument

The project's thesis is that reported quality depends more on the evaluation protocol
than on the model. Stages 1 and 2 established the first two beats. Stage 3 supplies
the third and completes it:

> **Stage 1:** a 15-line linear model (EASE) wins the benchmark.
> **Stage 2:** it cannot scale past ~20k items, and cannot score a product it has never
> seen. The neural retrieval model loses the benchmark and solves both.
> **Stage 3:** the two-stage architecture that makes large catalogues tractable is a
> *scalability device*, and on a catalogue small enough to score exhaustively it may
> buy no accuracy at all — here is the measured size of that trade.

Read together: much of what industrial recommender architecture looks like is not
sophistication chasing accuracy, it is a tax paid for scale and freshness. Measuring
the size of that tax is a more useful contribution than another leaderboard row.

**This argument survives every plausible outcome of Stage 3, including — especially —
the reranker failing to improve accuracy.** See §6.

---

## 1. Scope

### In scope

1. Ranker training-data construction with a nested temporal design (§3).
2. Feature engineering for the ranker, with a strict pre-label-time rule.
3. LightGBM `lambdarank` reranker, plus a pointwise-binary ablation.
4. End-to-end two-stage evaluation, comparable to the single-stage numbers.
5. HNSW index over the embedding-based retrievers; recall/latency Pareto curve;
   end-to-end metric impact of approximation.
6. FastAPI `/recommend` + `/health`, with per-stage timings and a small latency
   harness.
7. Model artifact persistence and load-at-startup.
8. Tests, README, DATA_NOTES updates.

### Out of scope, with reasons for the README

- **Prometheus / Grafana / Kubernetes** — already demonstrated in the sibling
  `end2end_churn` project. Repeating them here would add volume, not evidence. The
  endpoint exists to *measure per-stage latency in a realistic composition*, not to
  re-demonstrate serving skills.
- **A feature store** — feature assembly at request time is served from memory, which
  is correct at this scale. What production would change is documented (§5.4) rather
  than built.
- **Online / A-B evaluation and bandits** — no logged propensities exist in this
  dataset, so off-policy evaluation would be fiction. Named as future work.

---

## 2. New dependencies and budgets

`lightgbm` (already a portfolio dependency, and the Dockerfile's `libgomp1` install
from Stage 1 covers it), `hnswlib`, `fastapi`, `uvicorn`, `httpx` (test client).

**`hnswlib` over `faiss-cpu`:** smaller image, no BLAS/OpenMP complications, and the
HNSW algorithm is the one being demonstrated either way. Note `faiss` as the
larger-scale industrial alternative in the README rather than depending on it.

| Budget | Target |
|---|---|
| Ranker feature build | ≤ 5 min |
| Ranker training (incl. tuning) | ≤ 10 min |
| ANN sweep (build × query grid) | ≤ 5 min |
| Latency harness | ≤ 2 min |
| Full three-stage `reproduce` | ≤ 60 min total |

---

## 3. Ranker training data — the part that goes wrong quietly

### 3.1 The nested temporal design

A reranker is trained on *candidates produced by a retriever*, labelled by what the
user actually did. The failure mode is subtle and common:

> If the retriever that generates candidates for ranker **training** was itself
> trained on the interactions being used as **labels**, then retrieval scores are
> in-sample during ranker training and out-of-sample at serving time. The ranker
> learns to trust a signal that will be systematically weaker in production. Nothing
> crashes; the model is simply miscalibrated in a way no shape assertion catches.

The correct structure is nested and temporal:

```
|<---------- A: retrieval fit ---------->|<-- B: ranker labels -->|<-- C: test -->|
start                                    T1                       T2            end
```

- **Period A `[start, T1)`** — fit retrieval models used to generate ranker training
  candidates.
- **Period B `[T1, T2)`** — generate candidates for each active user from the
  period-A retrievers; label an item `1` if the user interacted with it during B,
  else `0`. Train the ranker on these.
- **Period C `[T2, end)`** — the Stage 1 test window, untouched. For final
  evaluation, retrieval is **refit on `[start, T2)`** and the trained ranker is
  applied on top.

Refitting retrieval for the final pass while keeping the ranker fixed mirrors
production, where retrieval is retrained frequently and the ranker less often. It
does introduce a mild train/serve mismatch in the retrieval-score features.
**Document this explicitly**, and provide `--retrieval-refit/--no-retrieval-refit` so
the alternative (retrieval frozen at period A throughout) can be run and the
difference reported. If the gap is large, that is a finding about retraining cadence
worth a paragraph.

### 3.2 Candidate-set size

Take `N` from Stage 2's retrieval-ceiling curve — the point where recall flattens
relative to cost, likely 500 or 1000. Justify the choice from the measured curve
rather than picking a round number, and state the ceiling it implies: the reranker
cannot exceed it.

Use the best blended retriever if Stage 2 showed blending raises the ceiling;
otherwise the best single retriever. Record which, and why.

### 3.3 Label definition and class balance

Positives are rare — most of 500 candidates are negatives. Record the positive rate;
if it is extreme, consider negative downsampling **with a documented rate** and note
that downsampling changes score calibration but not ranking. Ranking is all we need,
so this is acceptable — but say so rather than leaving it implicit.

---

## 4. The reranker

### 4.1 Features

Every feature must be computable from data **strictly before the label timestamp**.
Feature leakage in rankers is rampant and is the single most likely way to produce a
spectacular, meaningless result.

| Group | Features |
|---|---|
| **Retrieval signals** | Score and rank from *each* retriever (so the ranker learns to blend), plus a flag for which retrievers surfaced the item |
| **Item** | Train-window popularity, recency of last interaction, item age since first appearance, category, parent category, availability |
| **User** | Interaction count, distinct items, distinct categories, session count, recency of last event, mean session length |
| **User × item** | Prior interactions with this item's category, time since last interaction in that category, share of the user's history in that category, whether the item was viewed but not purchased earlier |
| **Context** | Rank position within the candidate set, candidate-set size |

RetailRocket has no prices, no text, and anonymised property hashes — **so the feature
set is genuinely thin, and that limits how much a reranker can add.** State this in
the README as a property of the data, not as an excuse discovered afterwards. It is
also the main reason §6 treats "the reranker does not help" as a likely outcome.

### 4.2 Model

LightGBM with `objective="lambdarank"`, grouped by user, `ndcg_eval_at=[10, 20]`.
Tuned on a validation slice carved temporally from period B — never on period C.

**Ablation: pointwise binary vs. pairwise LambdaMART.** Train an `LGBMClassifier` on
the same features and rank by predicted probability. Pointwise often performs close
to LambdaMART in practice, and knowing whether the listwise objective earns its
complexity on this data is exactly the kind of question this portfolio asks. One
extra flag, one extra row.

### 4.3 End-to-end evaluation

The two-stage system is evaluated on the **same test users, same metrics, same
protocol** as every single-stage model, so the numbers drop straight into Stage 1's
comparison table. Report:

- Best single-stage model (from Stages 1–2).
- Two-stage: retrieval@N → rerank → top-k.
- The ceiling from §3.2, and **how much of the Stage 2 gap the reranker closed**,
  against the target Stage 2 committed to numerically.
- Beyond-accuracy metrics for the two-stage system — a reranker trained on engagement
  labels tends to concentrate on popular items, and Stage 1's Gini and popularity
  measures will show it. If the reranker improves NDCG while collapsing catalogue
  coverage, **that trade is the finding**, and it is a more commercially interesting
  one than the NDCG delta.

---

## 5. Approximate nearest neighbours and serving

### 5.1 What can be indexed, and what that says

ANN search applies only to models that produce an embedding space: **ALS and the
two-tower model**. EASE and ItemKNN produce item-item similarity structures, not a
metric space over items, so fast vector search does not apply to them.

That is not a footnote — it is part of the argument. **The architecture that scales in
memory is also the one that admits sublinear retrieval.** EASE's disadvantages
compound: quadratic memory, no cold start, and no fast approximate serving path. Say
this once, plainly, in the README.

### 5.2 The sweep

Index the item embeddings with HNSW. Sweep build parameters (`M`,
`ef_construction`) and query parameter (`ef_search`), and for each configuration
measure:

- **Recall of the ANN top-k against exact brute-force top-k** (the approximation's own
  fidelity).
- **Query latency**, p50 / p95 / p99 over many queries, single-threaded.
- **Index build time and memory footprint.**
- **End-to-end NDCG@10 of the full two-stage system** using ANN retrieval versus exact
  retrieval.

That last row is the one that matters. The expected result — small recall loss, *no
measurable end-to-end metric change*, large speedup — is the honest answer to "what
did the approximation cost you", and it is a much better interview answer than a
recall number alone.

Outputs: `outputs/ann_sweep.csv`, `assets/ann_recall_latency.png` (Pareto curve,
recall on y, p95 latency on x, annotated with `ef_search`).

### 5.3 Honesty about scale

**At ~20k items, exact search is entirely affordable and ANN is not necessary.** Do
not imply otherwise. The correct framing:

> The approximation is unnecessary at this catalogue size — exact search over 20,000
> items takes single-digit milliseconds. It is built and measured here because the
> shape of the trade-off is what matters, and because the crossover is a property of
> catalogue size, not of this dataset. The curve shows what you buy and what you pay;
> extrapolating the operation count to a 10-million-item catalogue is arithmetic, and
> is stated as arithmetic rather than measured as a claim.

Overselling the necessity of ANN here would undercut a project whose entire premise is
not overselling things.

### 5.4 The endpoint

FastAPI, deliberately minimal:

- `POST /recommend` — accepts a `user_id`, **or** a list of recent `item_ids` for an
  anonymous or unseen visitor (the history-based user tower from Stage 2 makes this
  possible, and exercising it is worth more than a user-id lookup). Returns ranked
  items with scores and per-stage timings.
- `GET /health` — model artifacts loaded, index built, catalogue size.

Pipeline inside the handler, timed at each step: **candidate retrieval (ANN) →
feature assembly → ranking → filter already-seen → top-k**.

Report p50/p95 **per stage**, not just end to end. The expected and interesting result
is that feature assembly and ranking dominate retrieval — which is exactly why
production systems retrieve cheaply and rank expensively, and why the two-stage split
exists at all. A small script issuing N sequential requests and reporting percentiles
is sufficient; no Locust, no load-testing infrastructure.

**What production would change** (document, do not build): features served from a
feature store rather than memory; the ANN index rebuilt on a schedule with atomic
swap; model artifacts versioned in a registry; the ranker retrained on a different
cadence from retrieval; request-level logging of served candidates and their scores,
which is the prerequisite for the off-policy evaluation this project declines to fake.

---

## 6. The most likely outcome, and why it is fine

**The reranker may not beat the best single-stage model.** On a 20k-item catalogue
with no prices, no text, and hashed properties, the feature set available to the
ranker is thin, and the retrieval models already encode most of the available signal.

This must be pre-committed as a reportable result, not treated as a bug to be tuned
away. If it happens, the README says:

> The two-stage architecture did not improve accuracy on this dataset. That is the
> expected result once you see what it is for: two-stage exists so that a catalogue
> too large to score exhaustively can be served in milliseconds. This catalogue is
> not too large — 20,000 items can be scored exactly in single-digit milliseconds — so
> the architecture is being asked to pay for a problem the data does not have. Its
> cost is measurable here; its benefit only appears at catalogue sizes this dataset
> cannot demonstrate. That is worth knowing, and it is the same lesson as the rest of
> the project: the benchmark and the deployment optimise different things.

That paragraph is a *stronger* ending than "the reranker improved NDCG by 4%". It is
also the third instance of the portfolio's recurring finding — sophistication
declined where it does not pay — following the RAG project's query rewriting and the
uplift project's CUPED.

Tuning the reranker until it wins, or quietly reporting only the metric on which it
happens to lead, would be the one genuine failure available in Stage 3.

---

## 7. Tests

Add roughly 15–20 tests.

**`test_ranker_data.py`** — the leakage-critical group.
- **Nested-window integrity:** period A precedes B precedes C, with no overlap; the
  ranker's training candidates come from a retriever fit only on A.
- **Feature leakage:** corrupt all interactions at or after each label timestamp,
  rebuild features, assert every feature value is unchanged. This is the Stage 1
  leakage test applied at the feature level, and it is the most important test in
  Stage 3.
- Labels match a hand-computed fixture; positive rate is recorded, not silently
  handled.

**`test_ranker.py`**
- The ranker reorders candidates — output order differs from input order — and never
  introduces an item absent from the candidate set.
- Grouping is by user: no group spans two users, and group sizes sum to the row count.
- Both objectives (lambdarank, binary) train and produce finite scores.
- Reproducible under a fixed seed.

**`test_ann.py`**
- ANN recall against exact brute-force exceeds a threshold at the documented
  `ef_search`.
- Results contain no duplicates and respect `k`.
- Already-seen items are absent from the final output **after the full pipeline** —
  the classic production bug, tested at system level here as well as at model level
  in Stage 1.
- Exact and ANN paths return identically shaped, schema-compatible results.

**`test_api.py`**
- Response schema validates; scores are monotonically non-increasing.
- Unknown `user_id` is handled gracefully via a documented fallback rather than a 500.
- Empty history and single-item history both produce valid responses.
- `k` bounds are enforced; invalid input returns 422, not a stack trace.
- **Timing fields exist and the per-stage timings sum to the reported total** — assert
  *structure*, never wall-clock thresholds. Timing assertions are flaky in CI and
  would be the one test in this repo that fails for reasons unrelated to correctness.

---

## 8. CLI and Makefile additions

```
reclab rank-data       # build the nested-window ranker training set
reclab train-ranker    # train + tune the reranker (both objectives)
reclab evaluate-e2e    # end-to-end two-stage evaluation, comparable to Stage 1
reclab ann-sweep       # HNSW recall/latency sweep + Pareto figure
reclab serve           # launch the FastAPI app (uvicorn)
reclab latency         # per-stage latency harness against a running server
```

Makefile: `serve` (with port mapping added to `docker-compose.yml`), `latency`,
`ann-sweep`, `evaluate-e2e`, plus extending `reproduce` and `refresh-assets` for the
new figure.

New options: `--candidates N`, `--ranker-objective {lambdarank,binary}`,
`--retrieval {exact,ann}`, `--retrieval-refit/--no-retrieval-refit`.

---

## 9. Documentation

**`README.md`** — final form. Add:

- The two-stage system diagram and the nested-window training design from §3.1, which
  is the most technically interesting thing in Stage 3 and should be explained, not
  just implemented.
- End-to-end comparison table: best single-stage vs two-stage, with beyond-accuracy
  columns alongside, and the ceiling-gap closure against Stage 2's stated target.
- The lambdarank-vs-pointwise ablation.
- The ANN Pareto figure, the end-to-end metric impact of approximation, and the
  scale-honesty paragraph from §5.3.
- Per-stage latency table.
- The closing argument from §0, and — if it applies — the §6 paragraph.
- "What would change for production" (§5.4).
- Reasons for the omissions in §1.

**Portfolio root `README.md`** — final update: Key Findings, Tech Stack (FastAPI,
HNSW, LightGBM ranking), Technologies Used So Far (learning-to-rank, approximate
nearest-neighbour search), project count, Last Updated.

---

## 10. Definition of done

1. `make reproduce` runs all three stages within budget and regenerates every number.
2. `make check-readme` exits zero across all tables.
3. All Stage 1 and 2 tests still pass; every Stage 3 test in §7 exists and passes.
4. The feature-leakage test is green and was written **before** the ranker was trained.
5. The end-to-end table reports beyond-accuracy metrics next to accuracy.
6. The ANN section states plainly that approximation is unnecessary at this catalogue
   size.
7. Latency is reported per stage.
8. The reranker result is reported whichever way it fell, with §6's framing if it did
   not help.
9. The README's closing argument is supported by numbers in `outputs/`.

---

## 11. Risks

| Risk | Mitigation |
|---|---|
| **Reranker does not improve accuracy** | Most likely outcome. Pre-committed in §6 as a reportable finding with its framing already written. Do not tune until it wins. |
| **Ranker feature leakage produces a spectacular result** | Write `test_ranker_data.py` first. Treat any large, sudden gain as a leak until the test proves otherwise — that instinct is the point of the project. |
| **In-sample retrieval scores during ranker training** | The nested design in §3.1 exists solely to prevent this. It is the subtlest failure in Stage 3 and warrants review attention. |
| **ANN oversold** | §5.3's framing is mandatory. Necessity is not claimed at this scale; the trade-off curve and the arithmetic extrapolation are. |
| **Latency numbers read as production claims** | Label them as single-threaded laptop measurements inside Docker, and report per-stage shares — which are informative — rather than absolute numbers, which are not. |
| **Scope creep into a second monitoring stack** | §1's exclusions are deliberate and argued in the README. The endpoint is a measurement instrument, not a product. |
| **Thin features limit the ranker** | Known in advance from the data (§4.1). Documented as a property of RetailRocket, not discovered as a disappointment. |

---

## 12. Suggested working order

1. `test_ranker_data.py` and the nested-window splitter — **before any ranker code.**
2. Feature builder; run the leakage test until green.
3. Ranker training with `lambdarank`; end-to-end evaluation wired into Stage 1's
   comparison table.
4. Pointwise ablation.
5. Beyond-accuracy metrics for the two-stage system (the popularity-concentration
   check).
6. HNSW index, recall-vs-exact test, then the sweep and the Pareto figure.
7. End-to-end metric impact of ANN versus exact.
8. Artifact persistence and load-at-startup.
9. FastAPI app; contract tests; per-stage timing harness.
10. Final README rewrite from the actual outputs; `make check-readme` until green.
11. Root README update; final full-suite run of all three stages from a clean clone.

Work on a branch; do not commit to the portfolio's default branch directly.
