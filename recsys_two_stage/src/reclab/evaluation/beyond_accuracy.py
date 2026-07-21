"""Beyond-accuracy metrics: what a recommender does to the catalogue.

Accuracy is not the business question. A recommender that shows everyone the same
twenty bestsellers can post a decent hit-rate and be commercially worthless — it
sells nothing a shop would not have sold anyway, and 99% of the catalogue is
never seen. So alongside accuracy we report:

- **Coverage@k** — fraction of the catalogue that appears in *any* session's
  top-k. Low coverage means most products are invisible.
- **Gini** of recommendation frequency — how concentrated exposure is across
  items. 0 = every item recommended equally often; near 1 = a handful of items
  soak up all the exposure.
- **Mean popularity percentile** of recommended items — popularity bias. A model
  that only surfaces items already popular in training is not discovering
  anything.

These are what the Stage 2 logQ-correction ablation moves, and what a reranker
trained on engagement tends to quietly wreck, so they are first-class outputs
from the start.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def gini(frequencies: np.ndarray) -> float:
    """Gini coefficient of an exposure-count vector over items.

    0.0 = perfectly even exposure; approaches 1.0 as exposure concentrates on a
    few items. Items never recommended (count 0) are included — they are exactly
    the concentration the metric should register.
    """
    counts = np.sort(np.asarray(frequencies, dtype=np.float64))
    n = len(counts)
    if n == 0:
        raise ValueError("empty frequency vector")
    total = counts.sum()
    if total == 0:
        return 0.0  # nothing recommended at all — no concentration to speak of
    # Standard order-statistic form: sum_i (2i - n - 1) x_i / (n * sum x).
    index = np.arange(1, n + 1)
    return float((np.sum((2 * index - n - 1) * counts)) / (n * total))


def beyond_accuracy_metrics(
    top_k: np.ndarray,
    n_items: int,
    item_popularity: np.ndarray,
    k: int,
) -> dict[str, float]:
    """Coverage, Gini and popularity bias for one model's top-k recommendations.

    ``top_k`` is (n_sessions, K) item indices; only the first ``k`` columns are
    used. ``item_popularity`` is the training interaction count per item, used to
    turn "how popular is what we recommend" into a percentile.
    """
    if k > top_k.shape[1]:
        raise ValueError(f"k={k} exceeds the {top_k.shape[1]} ranked items supplied")
    recommended = top_k[:, :k].ravel()

    exposure = np.bincount(recommended, minlength=n_items).astype(np.float64)
    coverage = float((exposure > 0).sum() / n_items)

    # Popularity percentile: rank items by training popularity, read off the
    # percentile of each recommended item, average. 1.0 = always recommending the
    # single most popular item; 0.5 = popularity-neutral.
    order = np.argsort(np.argsort(item_popularity))  # 0 = least popular
    percentile = order / max(n_items - 1, 1)
    mean_pop_percentile = float(percentile[recommended].mean())

    return {
        "k": k,
        "coverage": coverage,
        "gini": gini(exposure),
        "mean_pop_percentile": mean_pop_percentile,
    }


def evaluate_beyond_accuracy(
    model,
    split,
    item_popularity: np.ndarray,
    ks: tuple[int, ...] = (10, 20),
    chunk_size: int = 2048,
) -> pd.DataFrame:
    """Beyond-accuracy metrics for one model over all test sessions."""
    from reclab.evaluation.metrics import top_k_from_scores

    max_k = max(ks)
    collected = []
    for start in range(0, split.n_test_sessions, chunk_size):
        stop = min(start + chunk_size, split.n_test_sessions)
        histories = split.test_prefix[start:stop]
        scores = model.score(histories)
        seen = histories.toarray() > 0
        collected.append(top_k_from_scores(scores, max_k, mask=seen))
    top_k = np.vstack(collected)

    name = getattr(model, "name", type(model).__name__)
    rows = []
    for k in ks:
        metrics = beyond_accuracy_metrics(top_k, split.n_items, item_popularity, k)
        rows.append({"model": name, "protocol": split.protocol, **metrics})
    return pd.DataFrame(rows)
