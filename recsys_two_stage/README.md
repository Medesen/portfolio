# Recommender Systems — Two-Stage Retrieval & Ranking, and What the Evaluation Protocol Hides

**Stage 1 of 3.** An honest evaluation study of classical session-based recommenders
on real e-commerce data — built to show that, with recommenders, *how you measure
changes the answer more than what you build*.

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

## Quick Start (~5 minutes)

### Prerequisites

- **Docker Desktop** with Docker Compose V2 (`docker compose`, not `docker-compose`)
- ~4 GB free disk space; **~4 GB RAM** free (EASE inverts a dense item×item matrix)
- No API keys, no accounts, no data downloads — the dataset is bundled

### One-Command Setup

```bash
git clone https://github.com/Medesen/portfolio.git
cd portfolio/recsys_two_stage

make setup        # Linux/macOS/WSL2/Git Bash
.\setup.ps1       # Windows PowerShell
```

### Try It Out

```bash
make eda          # dataset summary + the filter-threshold grid
make evaluate     # full-catalogue evaluation, all four models (~4 min)
make sampled      # sampled-negative evaluation + the disagreement table (~4 min)
make beyond       # coverage / Gini / popularity-bias metrics
make protocols    # temporal vs leave-one-out
make tune         # re-run the validation-window grid search (regenerates TUNED_PARAMS)
make reproduce    # every number in this README, end to end (~15 min)
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

### Data judgment

- **The pivot is documented, not hidden.** RetailRocket users barely persist (79.6%
  view one item ever), so a user-based protocol was abandoned for a session-based one
  *before* modelling — see [DATA_NOTES.md](DATA_NOTES.md), worth reading first. The
  reasoning, the pre-registered test, and the dataset-choice trade-off (RetailRocket
  over MovieLens: bundleable licence, real behaviour, content features for Stage 2)
  are all on record.

## Testing

79 tests, all runnable in Docker:

```bash
make test
```

The ones worth a reviewer's eye:

- **The Krichene-Rendle reversal, as a test.** On synthetic data with a known true
  model ordering, full-catalogue evaluation recovers it and sampled evaluation
  **reverses** it — the paper's result turned into a property that fails if the
  effect ever stops reproducing (`test_sampled_bias.py`).
- **EASE against a brute-force solve.** The closed form is checked against an
  independent, column-wise ridge regression — the same optimisation attacked a
  different way (`test_models.py`).
- **Temporal leakage.** Corrupt every post-cutoff interaction and assert the trained
  matrix is bit-identical; straddling sessions are dropped, not truncated
  (`test_splitting.py`).
- **Seen-item masking is central, not per-model.** A model that forgets to exclude a
  session's own history scores spectacularly for nothing; the exclusion lives in the
  evaluation loop and a test asserts it holds for every model.
- **k-core reaches a genuine fixed point** — a second application is a no-op.

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
│   ├── splitting/                # Temporal + leave-one-out protocols
│   ├── models/                   # Popularity, ItemKNN, EASE, ALS
│   ├── evaluation/               # Metrics, full-catalogue, sampled, beyond-accuracy
│   ├── tuning/                   # Nested temporal-validation grid search
│   └── main.py                   # CLI: eda / tune / evaluate / sampled / protocols / beyond / all
├── tests/                        # 79 tests
├── scripts/check_readme.py       # Verifies README numbers against outputs/
├── Dockerfile / docker-compose.yml / Makefile / setup.sh / setup.ps1
└── outputs/                      # Result tables (gitignored)
```

## Honest Limitations & Future Work

### Stage 2 & 3 roadmap

This is Stage 1 — a complete evaluation study of classical models. The next stages
close the argument the headline sets up:

- **Stage 2** — a **two-tower** neural retriever and **SASRec** (sequential
  transformer), the retrieval-ceiling analysis, and a **cold-start** evaluation using
  the bundled item content features (6–7% of test interactions involve items with no
  training history). The expectation, consistent with the literature: the neural
  models may *lose* this full-catalogue benchmark and still be what industry deploys.
- **Stage 3** — a **LightGBM reranker** (the second stage), an **approximate
  nearest-neighbour** index with a measured recall-vs-latency curve, a minimal
  `/recommend` endpoint with per-stage latency, and the **catalogue-scaling sweep**
  that turns EASE's memory wall from arithmetic into a measured curve.

### Known limitations of this stage

- **Single held-out target per session**, so Recall@k collapses to HitRate@k; the
  metric set is HitRate / NDCG / MRR by design, stated rather than papered over.
- **ALS is capped at 128 factors** to stay CPU-trainable; it improves with more, but
  the qualitative result (third on the honest metric) is robust across the grid.
- **`make reproduce` refits each model per subcommand**, so it runs ~15 min rather
  than the ~5 the models themselves need; kept this way so each subcommand is
  independently runnable.

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
