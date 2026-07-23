# Stage 2 Build Plan — Neural Retrieval, Sequential Models, and Cold Start

**Status:** proposed, not started
**Prerequisite:** Stage 1 complete and merged (see `PLAN_STAGE1.md`)
**Audience:** a developer picking this up cold

---

## 0. Where Stage 1 left off, and what Stage 2 adds

Stage 1 built the foundation: bundled RetailRocket data, an iterative k-core filter
derived from the training window only, a global temporal split with a leave-one-out
comparator, full-catalogue **and** sampled-negative metrics, beyond-accuracy metrics,
and four tuned classical models (Popularity, ItemKNN, EASE, ALS).

Stage 2 adds the modern neural half of the comparison and the two analyses that turn
a model bake-off into a systems study:

1. **Two-tower retrieval model** — the dual-encoder architecture that industry uses
   for candidate generation, trained with in-batch sampled softmax and a logQ
   sampling-bias correction.
2. **SASRec** — the self-attentive sequential model, the standard strong baseline for
   "what will this person do next".
3. **Retrieval-ceiling analysis** — how much of the achievable recall each retriever
   captures at candidate-set sizes a production system would actually use. This is
   the ceiling Stage 3's reranker will be measured against.
4. **Cold-start evaluation** — the axis on which the classical winners fail
   structurally, using the item content features bundled (but unused) in Stage 1.

**The design test for Stage 2:** if Stage 1 was built correctly, Stage 2 should add
*models* and reuse the evaluation harness untouched. Any need to modify
`evaluation/` to accommodate a neural model is a signal that Stage 1's abstraction
leaked, and should be fixed there rather than worked around here. The one legitimate
exception is §5 (cold start), which deliberately requires a second evaluation track.

### The argument Stage 2 is assembling

Stage 1 likely showed a classical method — most plausibly EASE — winning on accuracy.
Stage 2 exists to show why that finding does not mean what it appears to mean:

> EASE cannot score an item it has never seen co-purchased, and its memory grows with
> the square of the catalogue. The two-tower model may score worse on the benchmark
> and is nevertheless what gets deployed, because it scales linearly and handles new
> products. The benchmark and the business optimise different things.

The first half of that claim is established by Stage 1's memory table; Stage 2
establishes the second half by measurement. **If the measurements do not support it,
the claim gets rewritten, not the measurements.**

---

## 1. Scope

### In scope

1. Item content-feature extraction from the bundled `item_properties.csv.gz`,
   snapshotted as-of the training cutoff.
2. Two-tower model (PyTorch, CPU-trainable), with the logQ correction and a
   correction on/off ablation.
3. SASRec (PyTorch, CPU-trainable), with a full-softmax vs sampled-BCE loss ablation.
4. Retrieval-ceiling analysis, including multi-retriever candidate blending.
5. Cold-start evaluation track with its own data path and metrics.
6. Tests for all of the above, including two leakage tests specific to neural models.
7. README and DATA_NOTES updates.

### Out of scope (Stage 3)

Reranker, approximate-nearest-neighbour index, latency measurement, serving endpoint.

### Explicitly not built, with reasons for the README

- **BERT4Rec** — its reported advantage over SASRec did not survive reproduction once
  SASRec was trained with a comparable loss (Klenitskiy & Vasilev, 2023). Building it
  would restate a finding this project already makes in §4.3, at several times the
  cost.
- **Graph neural recommenders (LightGCN and relatives)** — covered by the "tuned
  baselines win" literature the project already builds on.
- **Pretrained text/image encoders for item content** — RetailRocket's property values
  are anonymised hashes, so there is no natural language to encode. Stating this is
  more useful than pretending otherwise.

---

## 2. New dependencies and compute budget

**PyTorch, CPU-only.** Install from the CPU wheel index to keep the image from
ballooning with CUDA libraries:

```
--index-url https://download.pytorch.org/whl/cpu
```

Note this in `requirements.lock` generation and in the Dockerfile. Expect the image
to grow by roughly 200–300 MB; state the new disk requirement in the README
prerequisites.

**Everything must remain CPU-trainable.** The portfolio's promise is clone-and-run on
an ordinary laptop, and that constraint drives the model sizing below. This is not a
compromise to hide — "the models are small enough to train on a laptop, and here is
what that costs" is a legitimate and honest scoping decision that belongs in the
README.

| Budget | Target |
|---|---|
| Two-tower training | ≤ 5 min |
| SASRec training | ≤ 15 min |
| Full Stage 2 `reproduce` increment | ≤ 30 min |
| Peak memory | ≤ 8 GB (still set by EASE from Stage 1, not by the neural models) |

If SASRec exceeds budget, reduce `max_seq_len` before reducing embedding dimension —
RetailRocket sessions are short, so long sequences buy little.

Add `--device` (default `cpu`) and honour `RECLAB_DEVICE` so a GPU can be used if
present, but never require one, and generate all committed numbers on CPU.

---

## 3. Item content features

### 3.1 `src/reclab/features/item_features.py`

```python
def build_item_features(
    cutoff: pd.Timestamp,
    item_index: pd.Index,
) -> ItemFeatures
```

Reads the bundled `item_properties.csv.gz` and `category_tree.csv.gz` and returns
aligned feature arrays for every item in `item_index`.

**Features:**

| Feature | Source | Encoding |
|---|---|---|
| `categoryid` | `item_properties`, property `categoryid` | Embedding over category vocabulary |
| `parent_category` | `category_tree` join | Embedding; coarser, denser signal |
| `available` | `item_properties`, property `available` | Binary |
| hashed property tokens | remaining properties, if retained in the distillation | Multi-hot → mean-pooled embedding bag |

### 3.2 The as-of-cutoff rule — the leakage hazard of this stage

`item_properties` is a **timestamped change log**, not a snapshot. Taking an item's
current category means taking a value that may have been set after the training
cutoff, which leaks future information into the item tower.

**Rule: for every item and every property, take the most recent value with
`timestamp <= cutoff`.** Items with no pre-cutoff value for a property get an explicit
"unknown" category index — not a silent default, because "we did not know this item's
category at training time" is real information and the model should be able to use it.

A test asserts that no feature value derives from a post-cutoff row (§7).

### 3.3 Feature coverage report

Before modelling, report and record in DATA_NOTES: what fraction of items have a
known category as-of cutoff, how many distinct categories survive, and the category
size distribution. If category coverage is poor, the cold-start story weakens and the
README must say so rather than overclaim. Write `outputs/item_feature_coverage.csv`.

---

## 4. Models

### 4.1 `models/two_tower.py`

Two small networks producing embeddings in a shared space; relevance is their dot
product.

**Item tower.** Item ID embedding **summed with** projected content features
(category, parent category, availability, property bag). The content path is what
makes cold-start scoring possible: an item with no ID-embedding signal still has
content. Provide `--item-tower {id_only,content_only,id_plus_content}` — `id_only`
is the ablation showing the content path is what buys cold-start capability.

**User tower.** Mean-pool (optionally recency-weighted) of the item embeddings in the
user's training history, then an MLP.

> **Deliberate design choice: no user ID embeddings.** RetailRocket users are mostly
> low-activity, and per-user parameters generalise to no one new. A history-based
> user tower generalises to any user with at least one interaction, including users
> unseen during training. Document this — it is the same reasoning that makes ALS
> awkward for new users while EASE and ItemKNN handle them naturally, and it is worth
> one line in the README because it is a genuine architectural trade-off rather than
> an implementation detail.

**Training.** In-batch sampled softmax: within a batch of (user, positive item)
pairs, every other item in the batch serves as a negative. Cross-entropy over the
batch, with a temperature parameter.

**The logQ correction.** In-batch sampling draws negatives in proportion to item
popularity, so popular items appear as negatives far more often than rare ones and
the model learns to suppress them beyond what the data warrants. The correction
subtracts `log(sampling_probability)` from each logit before the softmax (Yi et al.,
2019). Estimate sampling probability from empirical training frequency.

Implement with `--logq-correction/--no-logq-correction`. **The ablation is the
point** — run both and report the effect on accuracy *and* on the beyond-accuracy
popularity metrics from Stage 1. The expected result is that the correction shifts
recommendations toward less-popular items; whether it helps accuracy is an open
question on this data.

**Hyperparameters** (tuned on Stage 1's temporal validation window): embedding dim
(32/64/128), MLP width/depth, temperature, learning rate, batch size (which is also
the negative count — worth noting), epochs with early stopping on validation NDCG@20.

### 4.2 `models/sasrec.py`

Kang & McAuley (2018). A causally-masked transformer over each user's interaction
sequence, predicting the next item.

**Architecture.** Item embeddings + learned positional embeddings → N transformer
blocks (multi-head self-attention with causal mask, then a pointwise feed-forward
network, with residual connections, layer norm and dropout) → the final position's
hidden state scores all items via the shared item embedding matrix.

**Sizing for CPU:** `max_seq_len` 50, 2 blocks, 2 heads, embedding dim 64 as the
starting point. Tune within budget.

**Causal masking is the correctness-critical detail.** A bug there lets the model see
the future and produces spectacular, meaningless results. It gets a dedicated test
(§7).

### 4.3 The loss ablation — a finding worth reproducing

The original SASRec trains with binary cross-entropy against **one sampled negative**
per positive. Later work (Klenitskiy & Vasilev, 2023) showed that training the same
architecture with **full softmax cross-entropy over the catalogue** improves it
substantially — enough to erase BERT4Rec's reported advantage. With ~20k items after
Stage 1's filtering, full softmax is entirely affordable.

Implement `--loss {full_softmax,sampled_bce}` and report both.

This is a strong fit for the project's thesis: **the loss function mattered more than
the architecture, and a whole line of published comparisons was confounded by it.**
It costs one flag and is exactly the kind of result the portfolio is built around.

### 4.4 Protocol fairness for sequential models — read this carefully

Stage 1's protocol predicts *all* of a user's test-period items from their
training-period history alone. SASRec's natural evaluation is different: predict each
next item given everything before it, including earlier test items.

Scoring SASRec autoregressively while the other models predict once from frozen
history is **not a fair comparison** — it hands the sequential model information the
others never receive.

**Therefore:**

- **Primary table (comparable):** every model, including SASRec, predicts the user's
  full test set from training-period history only. Directly comparable to Stage 1.
- **Secondary table (clearly labelled):** SASRec and two-tower additionally evaluated
  autoregressively, with history updated as test interactions are revealed.

The secondary table is a genuine finding about deployment, not a cheat, **provided it
is labelled as answering a different question**: what a model gains from being
re-scored on every event rather than served from a nightly batch. Report the gap. If
it is large, that is an operational argument that belongs in the README, and it sets
up Stage 3's serving discussion.

---

## 5. Cold-start evaluation

### 5.1 Why this needs its own data path

Stage 1's filter drops test interactions involving items absent from training —
correctly, since those items are unscoreable by the classical models. Cold start is
precisely the study of those dropped interactions, so Stage 2 needs a second track
that retains them.

Add `--keep-cold-items` to the split, producing an additional evaluation set. The
warm evaluation from Stage 1 is unchanged; the cold evaluation is reported
separately and never mixed into the headline warm numbers.

### 5.2 Item-cold, not user-cold

Be precise, because the two are commonly conflated:

| | Classical (EASE / ItemKNN) | ALS | Two-tower | SASRec |
|---|---|---|---|---|
| **New item** (no training interactions) | Cannot score — structural | Cannot score | **Can score** via content | Cannot score |
| **New user** (unseen, has some history) | Scores fine from history | Needs fold-in | Scores fine (history-based tower) | Scores fine |

The interesting axis is **new items**. User cold start is worth one sentence noting
that ALS is the odd one out; it is not worth an experiment.

### 5.3 Definitions and metrics

- **Strictly cold:** test items with zero training interactions.
- **Near-cold:** fewer than 5 training interactions. Report as a separate bucket —
  the cliff is rarely sharp, and where it sits is informative.

Metrics: recall and NDCG restricted to cold items, plus a cold-start baseline of
**most-popular-within-predicted-category**, which is what a sensible engineer would
ship without a neural model. Beating that is the bar, not beating zero.

Models that cannot score cold items are reported as **0.0, explicitly, with a
footnote that this is structural rather than a tuning failure.** That row is the
point of the table.

### 5.4 The magnitude question — do not skip this

Report **what fraction of test-period interactions involve cold items at all.**

If only 2% of test interactions touch cold items, then the two-tower's cold-start
advantage is real but commercially marginal on this dataset, and the README must say
so. If it is 20%, the argument is strong. Either way the number is required — an
advantage on a slice nobody visits is not an advantage, and reporting the slice size
is what separates this from a demo.

Write `outputs/cold_start_share.csv` and `outputs/metrics_cold.csv`.

---

## 6. Retrieval-ceiling analysis

### 6.1 The idea

A two-stage system's ranker can only reorder what retrieval handed it. If the right
item is not in the candidate set, no ranker recovers it. So **Recall@N at
candidate-set sizes is a hard ceiling on the whole system**, and it is a more useful
description of a retriever than NDCG@10.

### 6.2 What to compute

For every retriever (all four Stage 1 models plus both Stage 2 models), compute
Recall@N for N ∈ {50, 100, 200, 500, 1000, 2000}.

Then compute the ceiling for **unions of retrievers** — every pair, plus the union of
all — at a fixed total budget. Production systems blend multiple retrieval sources
precisely because their misses are not identical, and showing whether that holds here
is cheap and directly informative for Stage 3.

### 6.3 Outputs

- `outputs/retrieval_ceiling.csv` — model × N → recall.
- `outputs/retrieval_blend.csv` — retriever combination × budget → recall.
- `assets/retrieval_ceiling.png` — recall vs N, log-x, one line per retriever, with
  the best single-model NDCG@10 marked to make the gap between "found it" and "ranked
  it first" visible at a glance.

Follow the Stage 1 asset convention: figures regenerate to gitignored `outputs/`,
with a committed copy in `assets/` refreshed via `make refresh-assets`.

### 6.4 What this sets up

The gap between Recall@1000 and NDCG@10 is the room available to Stage 3's reranker.
State it as an explicit, quantified target: *"retrieval finds the right item within
1000 candidates X% of the time; ranking currently surfaces it in the top 10 Y% of the
time; the reranker's job is to close some of the X−Y gap."* Ending Stage 2 with a
number that Stage 3 is measured against is much stronger than ending it with a
promise.

---

## 7. Tests

Add roughly 15–20 tests to Stage 1's suite.

**`test_item_features.py`**
- **As-of-cutoff correctness:** construct a property log where an item's category
  changes after the cutoff; assert the extracted feature is the pre-cutoff value.
- Items with no pre-cutoff value get the explicit "unknown" index, not a silent
  default or a null.
- Category and parent-category vocabularies are built from training data only.

**`test_two_tower.py`**
- **Cold-item scoring:** an item absent from training receives a finite, non-degenerate
  score under `id_plus_content`, and the `id_only` ablation demonstrably cannot do
  this. This is the architectural claim of §5, asserted as a property.
- **logQ correction direction:** with the correction enabled, the mean popularity
  percentile of recommended items is lower than without it. Tests the mechanism, not
  just that the code runs.
- Embedding shapes and the user tower's invariance to history ordering (it is a
  pooling operation — if order matters, something is wrong).

**`test_sasrec.py`**
- **Causal masking (the critical one):** modify an item at position `t` in an input
  sequence and assert that the model's outputs at all positions `< t` are unchanged.
  This is the transformer-shaped analogue of Stage 1's leakage test and the single
  most important test in Stage 2.
- Padding and short sequences are handled: a user with one interaction produces a
  valid prediction rather than a crash or a NaN.
- Both loss modes train without error and produce finite losses.

**`test_neural_sanity.py`** — cheap, and catches silent training bugs that no shape
assertion will:
- On tiny synthetic data with a trivially memorisable pattern (each user always
  interacts with items from exactly one cluster), both neural models must reach
  near-perfect recall. A model that trains without error but learns nothing fails
  here, and that failure mode is otherwise very hard to notice.
- Loss decreases monotonically over the first N steps on that data.

**`test_cold_start.py`**
- The cold split contains only items genuinely absent from training.
- Classical models return exactly 0.0 on cold items — the structural-failure claim,
  verified rather than assumed.
- The cold-share statistic matches a hand-computed value on a fixture.

**`test_retrieval_ceiling.py`**
- Recall@N is monotonically non-decreasing in N.
- The union of two retrievers has recall at least equal to the better of the two at
  the same total budget.

**Across all new models:** reproducibility under a fixed seed, on CPU.

---

## 8. CLI additions

```
reclab features        # build + report item content-feature coverage
reclab train-neural    # train two-tower and SASRec, persist to outputs/models/
reclab ceiling         # retrieval-ceiling analysis + blending + figure
reclab cold-start      # cold-item evaluation track
reclab ablations       # logQ on/off, id_only vs content, full-softmax vs sampled-BCE
```

`reclab evaluate`, `sampled`, `protocols` and `beyond` from Stage 1 must pick up the
new models with **no changes to their implementations** — they iterate over a model
registry. If they need editing, fix the registry, not the evaluation code.

New global options: `--device` (default `cpu`), `--epochs`, `--keep-cold-items`.

Extend `reclab all` to run the Stage 2 steps after the Stage 1 steps.

---

## 9. Documentation updates

**`DATA_NOTES.md`** — add sections on item-property coverage as-of cutoff, the
category vocabulary and its size distribution, and the cold-item share of test
interactions.

**`README.md`** — the substantive rewrite of the stage. Add:

- Two-tower and SASRec to the model table, with their tuned settings.
- The full comparison table, warm evaluation, all six models.
- The **ablation table** — logQ on/off, id_only vs id+content, full-softmax vs
  sampled-BCE. Three findings for the price of three flags.
- The **cold-start table**, with structural zeros for the classical models *and* the
  cold-share figure alongside it, so the advantage is presented at its true
  commercial size.
- The **retrieval-ceiling figure** and the quantified gap Stage 3 will target.
- The protocol-fairness note from §4.4, so a reader knows exactly what was compared
  with what.
- The reasons for not building BERT4Rec, graph models, or text encoders (§1).
- Updated Stage 3 roadmap.

**Portfolio root `README.md`** — update the entry's Key Findings and Tech Stack
(PyTorch, transformer-based sequential modelling, dual-encoder retrieval).

---

## 10. Definition of done

1. `make reproduce` runs Stage 1 + Stage 2 within budget and regenerates every number.
2. `make check-readme` exits zero, including all new tables.
3. All Stage 1 tests still pass; every Stage 2 test in §7 exists and passes.
4. Stage 1's evaluation modules are **unmodified** apart from the cold-start track.
5. The cold-start table reports both the advantage and the share of traffic it applies
   to.
6. The retrieval-ceiling figure is committed to `assets/` and the Stage 3 target gap
   is stated numerically.
7. Every ablation is reported whichever way it fell.
8. All committed numbers were generated on CPU.

---

## 11. Risks and open questions

| Risk | Mitigation |
|---|---|
| **The neural models simply lose** | Entirely possible, and on-brand. The project's thesis does not require them to win on accuracy — §0's argument is that they win on *scalability and cold start* while losing on the benchmark. Report the loss plainly; it strengthens the story rather than weakening it. |
| **Cold items are a negligible share of traffic** | Then the cold-start advantage is marginal on this data and the README says so, with the share reported next to the table. Do not bury the denominator. |
| **Category coverage as-of cutoff is poor** | Measure in §3.3 *before* building the item tower. If coverage is bad, the content path is weak and the cold-start claim must be softened — this is a Phase-3 finding, not a Phase-5 surprise. |
| **SASRec exceeds the CPU budget** | Reduce `max_seq_len` first (sessions are short), then blocks, then dimension. Do not silently move to GPU — committed numbers must be CPU-generated. |
| **The autoregressive comparison gets misread as the headline** | Keep it in a clearly labelled secondary table with an explicit statement that it answers a different question. |
| **Neural models train but learn nothing** | `test_neural_sanity.py` exists precisely for this. Write it *before* training on real data, not after. |
| **Stage 1's harness needs modification** | Treat as a Stage 1 design defect: fix it there, with its own test, rather than special-casing neural models in Stage 2. |

---

## 12. Suggested working order

1. Item feature extraction + as-of-cutoff tests + coverage report. **Check coverage
   before proceeding** — it gates the cold-start work.
2. `test_neural_sanity.py` and its synthetic fixtures, written first.
3. Two-tower: architecture → training loop → sanity test green → tune on the Stage 1
   validation window.
4. logQ ablation; verify the popularity-shift direction test.
5. SASRec: architecture → **causal-masking test green before any real training** →
   training loop → tune.
6. Loss ablation (full softmax vs sampled BCE).
7. Register both models; confirm Stage 1's evaluation commands pick them up unchanged.
8. Retrieval-ceiling analysis + blending + figure.
9. Cold-start track: split path, metrics, category-popularity baseline, share
   statistic.
10. Autoregressive secondary evaluation.
11. Update README, DATA_NOTES, root README; `make check-readme` until green.

Work on a branch; do not commit to the portfolio's default branch directly.
