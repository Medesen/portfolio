# Recommender Systems — Two-Stage Retrieval & Ranking, and What the Evaluation Protocol Hides

**Stages 1 & 2 of 3 complete.** An honest evaluation study of session-based
recommenders on real e-commerce data — classical baselines *and* neural retrieval —
built to show that, with recommenders, *how you measure changes the answer more than
what you build*, and that **the model that wins on ranking is not the one that wins on
retrieval**.

## Summary

Recommender systems had a public reproducibility reckoning that most portfolio
projects never touch, and this one is built on it:

- **Dacrema, Cremonesi & Jannach (2019)** — of 18 neural recommenders from top
  venues, most lost to properly *tuned* classical baselines.
- **Krichene & Rendle (2020)** — evaluating against ~100 sampled negatives (the
  field's default for years) produces model rankings that **disagree** with
  full-catalogue ranking. Leaderboards were partly an artifact of the shortcut.
- **Ludewig & Jannach (2018)** — on session data, simple neighbourhood methods beat
  the neural ones.

The thesis, and what this stage delivers evidence for:

> A recommender's reported quality depends more on the evaluation protocol than on
> the model.

Four tuned classical models — Popularity, ItemKNN, EASE, and implicit-feedback ALS —
are evaluated on RetailRocket (a real e-commerce click/cart/buy log, CC BY-NC-SA
4.0, bundled in this repo). The same models are scored three ways: the honest
full-catalogue protocol, the sampled-negative shortcut, and leave-one-out. **They
disagree on which model wins.** Stages 2 and 3 add neural retrieval (two-tower,
SASRec), a reranker, and the scalability trade-offs that explain *why* the industry
deploys the models that lose this benchmark — see [the roadmap](#stage-2--3-roadmap).

### The honest headline

**On the honest full-catalogue metric, the simplest models win; the latent-factor
model (ALS) is third of four. Switch to the sampled-negative shortcut every second
recommender paper used, and ALS jumps to *first*.** If you trusted the standard
sampled metric, you would ship the model that is actually the worst of the three at
finding the right item — and, as the beyond-accuracy table shows, the most
popularity-biased one too.

**Full-catalogue evaluation — the honest protocol** (21,053 test sessions, 13,754
items; 95% bootstrap CIs over sessions):

| Model | NDCG@20 | 95% CI | HitRate@20 | MRR@20 |
|---|---|---|---|---|
| **ItemKNN** | **0.319** | [0.315, 0.324] | 0.562 | 0.248 |
| **EASE** | **0.318** | [0.313, 0.323] | 0.557 | 0.248 |
| ALS | 0.264 | [0.260, 0.269] | 0.525 | 0.191 |
| Popularity | 0.009 | [0.008, 0.009] | 0.026 | 0.004 |

ItemKNN and EASE are a statistical tie (CIs overlap); ALS is clearly third;
popularity is nowhere. The pre-registered viability bar — *a personalised model must
beat popularity on NDCG@20 with non-overlapping CIs, or the dataset is wrong for the
study* — was set before any model ran and cleared at **37×**.

**The same models, ranked by the sampled-negative shortcut** (each true item scored
against 100 sampled negatives):

| Model | Full-catalogue rank | Sampled rank (uniform) | Sampled rank (popularity) |
|---|---|---|---|
| ItemKNN | **1** | 2 | 2 |
| EASE | **2** | 3 | 3 |
| **ALS** | **3** | **1** | **1** |
| Popularity | 4 | 4 | 4 |

ALS goes from **last of the real models to first**, under *both* negative samplers.
The rank correlation between the honest and the shortcut ordering is only **0.2**
(uniform) / **0.4** (popularity). Latent-factor models are good at separating the
target from random junk (the easy, sampled task) and worse at pushing it above all
13,754 items (the hard, real task); sparse co-occurrence models are the reverse. The
shortcut rewards the wrong skill — and even the *choice of sampler* reorders the
board.

**Beyond accuracy — what each model does to the catalogue** (top-20):

| Model | Coverage | Gini (exposure) | Mean popularity percentile |
|---|---|---|---|
| ItemKNN | 98.7% | 0.53 | 0.65 |
| EASE | 98.6% | 0.56 | 0.72 |
| ALS | 58.4% | 0.77 | 0.83 |
| Popularity | 0.2% | 1.00 | 1.00 |

The model the sampled metric flatters (ALS) is also the most concentrated and most
popularity-biased of the three — it surfaces less than 60% of the catalogue and
leans hard on already-popular items. The sampled shortcut and the commercial
downside point the same way, and the honest metric already caught it. (Popularity's
0.2% coverage — the same ~25 items shown to everyone — is why it is worthless
despite being a "baseline".)

**A third protocol, a third answer** — leave-one-out (the field's other default)
against the temporal split:

| Model | Temporal NDCG@20 | Leave-one-out NDCG@20 |
|---|---|---|
| ItemKNN | **0.319** | 0.271 |
| EASE | 0.318 | **0.277** |
| ALS | 0.264 | 0.248 |

Leave-one-out — which trains on interactions that occurred *after* the ones it
predicts — depresses every score and nudges EASE ahead of ItemKNN. A different
honest-looking protocol, a different leaderboard.

### One finding the project did not go looking for

ItemKNN and EASE tie on accuracy, but **ItemKNN fits in 0.3 seconds and EASE takes
56** — a 190× gap for no measurable accuracy difference, because EASE inverts a
dense 13,754² matrix and ItemKNN does not. That is the project's scalability thesis
showing up on a dimension it had not planned to measure, and it is what Stage 3's
catalogue-scaling sweep is built to make explicit.

## Stage 2 — Neural Retrieval, Sequential Models, and Cold Start

Stage 2 adds the modern neural half — a **two-tower** dual-encoder retriever and
**SASRec** (a causally-masked sequential transformer) — and the analysis that turns a
model bake-off into a *systems* study. All neural models are CPU-trainable (the whole
Stage 2 pipeline runs in ~15 minutes on a laptop); PyTorch is the only new dependency.

### The Stage 2 headline: ranking skill and retrieval skill are different skills

On the honest full-catalogue metric, the two-tower **loses** — it ranks the target in
the top 20 less well than the tuned classical models:

| Model | NDCG@20 | HitRate@20 |
|---|---|---|
| ItemKNN (classical) | **0.319** | 0.562 |
| EASE (classical) | 0.318 | 0.557 |
| ALS (classical) | 0.264 | 0.525 |
| **Two-tower** | 0.260 | 0.510 |
| SASRec (full-softmax) | 0.127 | 0.266 |
| SASRec (sampled-BCE) | 0.020 | 0.050 |

You could stop there and conclude the neural models simply lost. But **NDCG@20 is a
*ranking* metric, and retrieval is a different job.** A two-stage system's first stage
does not need to rank the target first — it needs to get it *somewhere* in a candidate
set of a few hundred to a couple of thousand, which a downstream ranker then reorders.
Measured that way — Recall@N, "is the target retrievable at all" — the order **flips**:

| Retriever | R@50 | R@100 | R@500 | R@1000 | **R@2000** |
|---|---|---|---|---|---|
| ItemKNN | 0.654 | 0.702 | 0.761 | 0.772 | 0.792 |
| EASE | 0.645 | 0.691 | 0.756 | 0.768 | 0.775 |
| ALS | 0.696 | 0.794 | 0.883 | 0.899 | 0.914 |
| **Two-tower** | 0.642 | 0.727 | 0.864 | 0.900 | **0.929** |
| SASRec | 0.379 | 0.482 | 0.726 | 0.812 | 0.881 |

![Retrieval ceiling](assets/retrieval_ceiling.png)

**The models that win on ranking (ItemKNN, EASE) are the *worst* retrievers.** They
peak early and saturate — ItemKNN finds 65% of targets in its top 50 but only 79% even
by 2000, because sparse co-occurrence has nothing to say about items a session has no
direct neighbour link to. The two-tower, third-worst on NDCG@20, is the **best
retriever** at depth (93% by 2000): its dense embedding space keeps finding relevant
items as the net widens. *This is precisely why industry builds two-stage systems* — a
high-recall embedding retriever to cast the net, then an expensive ranker to sort it.
The gap between "found it" (retrieval) and "ranked it first" (NDCG@10) is the room
Stage 3's reranker is built to work in.

Blending retrievers helps only modestly (the union of all five reaches 0.928 at a
500-item budget each, vs 0.914 for the best single retriever) — their misses are
correlated, which is itself worth knowing for Stage 3.

### Three ablations, three findings

| Ablation | Result |
|---|---|
| **Item tower: content path** | id+content **0.260** > id-only 0.230 > content-only 0.134 — the content features earn their place, adding ~13% NDCG over IDs alone |
| **logQ correction** | on **0.260** vs off 0.228: the correction *helps* accuracy (+14%) but concentrates recommendations on popular items (popularity percentile 0.55 → 0.70, coverage 0.97 → 0.94) — a real trade-off, and the *opposite* direction from a naive guess (it removes the in-batch sampler's incidental suppression of popular items) |
| **SASRec loss** | full-softmax **0.127** vs sampled-BCE 0.020 — a **6× gap** from the loss function alone, reproducing Klenitskiy & Vasilev (2023): the loss mattered far more than the architecture, and a line of published SASRec-vs-BERT4Rec comparisons was confounded by it |

SASRec underperforms here in absolute terms (0.127) — RetailRocket sessions average
~3 items, so a sequential model has little order to exploit, exactly the regime where
Ludewig & Jannach (2018) found simple methods win. It is also lightly trained (8
epochs, CPU budget). The robust finding is the *loss ablation*, which holds regardless
of the model's absolute level.

### Cold start: a real capability, still beaten by a simple heuristic

The classical models **cannot score an item with no training interactions** — the
recommendation is structurally impossible, so their cold-start recall is exactly 0.
The two-tower *can*, from an item's content (category, parent category, availability).
On the strictly-cold slice (targets with zero pre-cutoff interactions), ranking among
cold candidates:

| Model | Cold-start Recall@20 |
|---|---|
| category-popularity heuristic | **0.229** |
| Two-tower (content) | 0.157 |
| ItemKNN / EASE / ALS | 0.000 (structural) |

The two-tower breaks the structural barrier the classical models hit — but a simple
*most-popular-new-item-in-the-session's-category* heuristic still beats it. And this is
all on a **3.7% slice**: only 3.7% of evaluable test targets are genuinely new items
(9.8% have fewer than 5 prior interactions). So the neural cold-start capability is
real and the classical models genuinely lack it — but on this dataset its commercial
value is bounded by both the margin (a heuristic wins) and the slice size. Reporting
the advantage next to the 3.7% denominator is the honest framing; an advantage on a
slice few sessions reach is a small advantage.

## Quick Start (~5 minutes)

### Prerequisites

- **Docker Desktop** with Docker Compose V2 (`docker compose`, not `docker-compose`)
- ~6 GB free disk space (the image includes CPU-only PyTorch); **~4 GB RAM** free
  (EASE inverts a dense item×item matrix)
- No API keys, no accounts, no data downloads — the dataset is bundled
- No GPU required or used — every neural number is generated on CPU by design

### One-Command Setup

```bash
git clone https://github.com/Medesen/portfolio.git
cd portfolio/recsys_two_stage

make setup        # Linux/macOS/WSL2/Git Bash
.\setup.ps1       # Windows PowerShell
```

### Try It Out

```bash
# Stage 1 — classical models & evaluation protocols
make eda          # dataset summary + the filter-threshold grid
make evaluate     # full-catalogue evaluation, all four classical models (~4 min)
make sampled      # sampled-negative evaluation + the disagreement table (~4 min)
make beyond       # coverage / Gini / popularity-bias metrics
make protocols    # temporal vs leave-one-out
make tune         # re-run the validation-window grid search (regenerates TUNED_PARAMS)

# Stage 2 — neural retrieval, ablations, ceiling, cold start
make features     # item content-feature coverage (as-of cutoff)
make neural       # train + evaluate the two-tower and SASRec (~14 min, CPU)
make ablations    # logQ on/off, id vs content, full vs sampled loss
make ceiling      # retrieval-ceiling analysis + blending + figure
make cold-start   # cold-item evaluation vs the category-popularity baseline

make reproduce    # every number in this README, end to end (~30 min)
make check-readme # verify the README's headline numbers against outputs/
```

### Local Alternative (No Docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
reclab evaluate
```

## What This Project Demonstrates

### Evaluation discipline

- **The honest protocol is the headline, the shortcut is shown up.** Full-catalogue
  ranking is the default reported here; the sampled-negative metric is computed
  *specifically to demonstrate that it disagrees*, with the rank correlation stated.
  Most write-ups report only the shortcut.
- **A global temporal split, never leave-one-out as the headline.** Training is
  everything before a cutoff; testing is after. Leave-one-out is included only as a
  comparator — it trains on the future, and the numbers show what that buys.
- **The filter is fit on the training window only.** Deriving the k-core from the
  whole dataset lets test-period activity decide the training vocabulary — a subtle
  leak most preprocessing carries. The leaky variant is available behind
  `--filter-scope global` so the difference can be measured.
- **Beyond-accuracy metrics as first-class outputs.** Coverage, Gini, and popularity
  bias — because a recommender that only shows bestsellers can score adequately and
  be commercially worthless.
- **Uncertainty on every headline number** via bootstrap CIs over sessions, and a
  **pre-registered viability bar** fixed before any model was run.

### Modelling judgment

- **Baselines are tuned, not stubbed.** The entire Dacrema et al. result is that
  baselines lose only when under-tuned, so every model gets a real grid search on a
  nested temporal validation window — ItemKNN's shrinkage alone is worth +0.009
  NDCG. Winners are selected on validation, never on test.
- **EASE from the closed form**, one hyperparameter, validated in the tests against
  an independent column-wise ridge solve — and instrumented with the memory wall
  (13,754 items = 1.5 GB dense; 235k items = 442 GB) that is the whole point.
- **ALS folded in by hand.** `implicit` learns the item factors; the session-vector
  fold-in for unseen sessions is implemented directly (Hu et al. eq. 4), keeping ALS
  behind the same "score an unseen session from its history" contract as every other
  model — the constraint the single-visit data forces.
- **Neural models from scratch, CPU-only.** The two-tower (history-pooling user tower,
  no user-ID embeddings; item tower = ID + content; in-batch sampled softmax with logQ
  correction) and SASRec (causally-masked transformer, full-softmax vs sampled-BCE
  loss) are written in PyTorch and plug into the *unchanged* Stage 1 evaluation harness
  via a `HistoryBatch` that carries ordered sequences alongside the binary bag — the
  abstraction generalised, not special-cased.
- **Retrieval framed as a first-class, separate skill from ranking.** The
  retrieval-ceiling analysis measures Recall@N up to 2000 candidates, which is what
  exposes that the ranking winners are the retrieval losers — the finding that makes
  this a two-stage *systems* project rather than a leaderboard.

### Data judgment

- **The pivot is documented, not hidden.** RetailRocket users barely persist (79.6%
  view one item ever), so a user-based protocol was abandoned for a session-based one
  *before* modelling — see [DATA_NOTES.md](DATA_NOTES.md), worth reading first. The
  reasoning, the pre-registered test, and the dataset-choice trade-off (RetailRocket
  over MovieLens: bundleable licence, real behaviour, content features for Stage 2)
  are all on record.

## Testing

100 tests, all runnable in Docker:

```bash
make test
```

The ones worth a reviewer's eye:

- **The Krichene-Rendle reversal, as a test.** On synthetic data with a known true
  model ordering, full-catalogue evaluation recovers it and sampled evaluation
  **reverses** it — the paper's result turned into a property that fails if the
  effect ever stops reproducing (`test_sampled_bias.py`).
- **SASRec causal masking.** Perturb the item at a later sequence position and assert
  every earlier output is bit-identical — a causal-masking bug would let the model
  see the future it is predicting, and this is the transformer-shaped analogue of the
  temporal-leakage test (`test_sasrec.py`).
- **The two-tower's cold-start claim, as a property.** The content path lets a cold
  item find its cluster; the `id_only` ablation demonstrably cannot — the
  architectural claim asserted, not just asserted about (`test_two_tower.py`).
- **Neural sanity.** On a trivially learnable clustered world both neural models must
  reach near-perfect recall — catching the silent failure of a model that trains
  cleanly but learns nothing (`test_neural_sanity.py`).
- **EASE against a brute-force solve.** The closed form is checked against an
  independent, column-wise ridge regression — the same optimisation attacked a
  different way (`test_models.py`).
- **Temporal leakage.** Corrupt every post-cutoff interaction and assert the trained
  matrix is bit-identical; straddling sessions are dropped, not truncated
  (`test_splitting.py`).
- **Seen-item masking is central, not per-model**, and **k-core reaches a genuine
  fixed point** — a second application is a no-op.

## Project Structure

```
recsys_two_stage/
├── README.md                     # This file
├── DATA_NOTES.md                 # EDA findings → design decisions (read first)
├── PLAN_STAGE1/2/3.md            # The full three-stage build plan
├── data/
│   ├── raw/
│   │   ├── events.csv.gz          # The event log (32 MB, bundled, CC BY-NC-SA 4.0)
│   │   ├── item_properties.csv.gz # Distilled item properties (Stage 2 uses these)
│   │   ├── category_tree.csv.gz   # Category hierarchy
│   │   └── README.md              # Provenance, licence, checksums, columns
│   └── prepare_item_properties.py # Distillation script (documented, not run in CI)
├── src/reclab/
│   ├── data/                     # Loading, sessionization, iterative k-core
│   ├── splitting/                # Temporal + leave-one-out protocols (carry sequences)
│   ├── features/                 # Item content features, snapshotted as-of cutoff
│   ├── models/                   # Popularity, ItemKNN, EASE, ALS, TwoTower, SASRec
│   ├── evaluation/               # Metrics, full-catalogue, sampled, beyond-accuracy,
│   │                             #   retrieval-ceiling, cold-start
│   ├── tuning/                   # Nested temporal-validation grid search
│   ├── stage2.py                 # Stage 2 orchestration (neural, ablations, ceiling, cold)
│   └── main.py                   # CLI: eda/evaluate/sampled/…/features/neural/ceiling/cold-start/all
├── tests/                        # 100 tests
├── assets/retrieval_ceiling.png  # Committed retrieval-ceiling figure
├── scripts/check_readme.py       # Verifies README numbers against outputs/
├── Dockerfile / docker-compose.yml / Makefile / setup.sh / setup.ps1
└── outputs/                      # Result tables (gitignored)
```

## Honest Limitations & Future Work

### Stage 3 roadmap

Stages 1 and 2 are complete. Stage 3 closes the argument they set up:

- **Stage 3** — a **LightGBM reranker** (the second stage) working over the candidates
  the retrieval-ceiling analysis measured, an **approximate nearest-neighbour** index
  with a measured recall-vs-latency curve (which applies only to the embedding
  retrievers — ALS and the two-tower — another compounding disadvantage for EASE), a
  minimal `/recommend` endpoint with per-stage latency, and the **catalogue-scaling
  sweep** that turns EASE's memory wall from arithmetic into a measured curve.

### Known limitations of these stages

- **Single held-out target per session**, so Recall@k collapses to HitRate@k; the
  metric set is HitRate / NDCG / MRR by design, stated rather than papered over.
- **ALS is capped at 128 factors** to stay CPU-trainable; the neural models are
  likewise CPU-sized — SASRec in particular is lightly trained (8 epochs). Their
  *ranking* level would rise with more compute, but the qualitative findings (the
  two-tower's high retrieval recall, the loss ablation) are robust to it.
- **SASRec is weak in absolute NDCG** because RetailRocket sessions are ~3 items — a
  short-session regime where sequential order carries little, exactly where the
  literature expects simple methods to win. It is included to *make* that point and to
  carry the loss ablation, not because it was expected to top the table.
- **`make reproduce` refits models across subcommands** where a fully shared cache
  would not, so it runs ~30 min; the `stage2` and `all` paths already fit each unique
  neural model once, which is where the cost lives.

### Deliberately not built (and why)

- **LLM-based recommenders** — no settled evaluation methodology; including one would
  dilute a project whose entire point is honest measurement.
- **Bandits / off-policy evaluation** — genuinely important, and a separate project:
  this dataset carries no logged propensities, so off-policy estimates would be
  fiction.
- **Graph neural networks (LightGCN etc.)** — already covered by the "tuned baselines
  win" literature this project builds on; adding one would restate the finding.
- **A Kubernetes / monitoring stack** — already demonstrated in the sibling
  [`end2end_churn`](../end2end_churn/) project; repeating it would add volume, not
  evidence.

## License, Dataset License & Citation

The code in this project is MIT-licensed — see the repository [LICENSE](../LICENSE).

Bundled dataset: the **RetailRocket recommender system dataset**, **CC BY-NC-SA
4.0** — free to redistribute with attribution for non-commercial use, which is why
it ships in this repo. The distilled `item_properties.csv.gz` is an adaptation and
carries the same licence; the code does not. Provenance, checksums, and column
documentation are in [data/raw/README.md](data/raw/README.md). Please credit
Retailrocket / Roman Zykov if you reuse the data.

> **Why RetailRocket over MovieLens** (the field's usual benchmark): its licence
> permits redistribution, so the project keeps its clone-and-run promise; it is real
> e-commerce behaviour with genuine timestamps, so a temporal split reflects a
> deployed system; and it carries product attributes, without which the Stage 2
> cold-start comparison would be impossible. The cost is comparability with published
> MovieLens numbers — a trade made deliberately in favour of realism and a
> shippable licence.

## Troubleshooting

- **`docker-compose: command not found`** — this project needs Compose V2 (`docker
  compose`). Upgrade Docker, or see the portfolio root README.
- **Permission errors on `outputs/`** — run the `make` targets (they create the
  directory host-side first) or `mkdir outputs` before `docker compose run`.
- **`MemoryError` from EASE** — it needs a dense item×item matrix (~1.5 GB at the
  default catalogue). Free some RAM, or raise the item filter with a stricter
  `min_item_sessions`. The error prints the arithmetic — that constraint is the
  point, not a bug.
- **Slightly different last-digit ALS numbers** — `implicit`'s multithreaded ALS can
  perturb the factor init at the margin; the ItemKNN/EASE/popularity numbers are
  exact.
