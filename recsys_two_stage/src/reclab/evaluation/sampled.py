"""Sampled-negative evaluation — the shortcut, implemented to be discredited.

For years the field evaluated recommenders by ranking the true item against ~100
randomly sampled negatives instead of the full catalogue, because full ranking
was expensive. Krichene & Rendle (2020) showed the shortcut produces model
rankings that *disagree* with full-catalogue ranking: the metric is a biased,
high-variance estimator whose bias depends on the model, so which model "wins"
can flip.

This module computes the shortcut so it can be laid beside the honest
full-catalogue numbers from ``full_catalogue.py``. The disagreement between them
is the headline result of Stage 1, and ``test_sampled_bias.py`` proves the
reversal happens by construction on synthetic data.

Two samplers are provided because the choice itself changes the answer: uniform
negatives flatter models that already beat popularity, popularity-proportional
negatives make the task harder in a more realistic way.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from reclab.evaluation.metrics import hit_rate_at_k, ndcg_at_k, reciprocal_rank
from reclab.splitting.protocols import SessionSplit

METRICS = ("hit_rate", "ndcg", "mrr")


def sampled_metrics_from_scores(
    scores: np.ndarray,
    targets: np.ndarray,
    seen: np.ndarray,
    n_negatives: int,
    ks: tuple[int, ...],
    sampler: str = "uniform",
    item_popularity: np.ndarray | None = None,
    seed: int = 0,
) -> dict[tuple[str, int], list[float]]:
    """Core loop: rank each target against ``n_negatives`` sampled items.

    ``scores`` is (n_sessions, n_items); ``targets[i]`` is the held-out item for
    session i; ``seen[i]`` is its boolean history mask. Negatives are drawn from
    items that are neither seen nor the target.
    """
    n_sessions, n_items = scores.shape
    if sampler == "popularity" and item_popularity is None:
        raise ValueError("popularity sampler needs item_popularity")
    if max(ks) > n_negatives + 1:
        raise ValueError(
            f"k={max(ks)} exceeds the {n_negatives + 1}-item candidate set"
        )

    rng = np.random.default_rng(seed)
    if sampler == "popularity":
        # Smoothed so zero-popularity items remain sampleable; squared-free, the
        # raw counts are enough to bias toward the head.
        base_weights = item_popularity.astype(np.float64) + 1.0
    elif sampler != "uniform":
        raise ValueError(f"sampler must be 'uniform' or 'popularity', got {sampler!r}")

    per: dict[tuple[str, int], list[float]] = {
        (m, k): [] for m in METRICS for k in ks
    }
    all_items = np.arange(n_items)

    for row in range(n_sessions):
        target = int(targets[row])
        forbidden = seen[row].copy()
        forbidden[target] = True
        allowed = all_items[~forbidden]

        if sampler == "uniform":
            negatives = rng.choice(allowed, size=n_negatives, replace=False)
        else:
            w = base_weights[allowed]
            w = w / w.sum()
            negatives = rng.choice(allowed, size=n_negatives, replace=False, p=w)

        candidates = np.concatenate(([target], negatives))
        cand_scores = scores[row, candidates]
        # Break ties randomly so a model that scores many candidates equal to the
        # target is not silently rewarded by a stable argsort putting the target
        # first. Descending sort of (score + tiny noise).
        order = np.argsort(-(cand_scores + rng.random(len(candidates)) * 1e-12))
        ranked = candidates[order]

        relevant = {target}
        for k in ks:
            per[("hit_rate", k)].append(hit_rate_at_k(ranked, relevant, k))
            per[("ndcg", k)].append(ndcg_at_k(ranked, relevant, k))
            per[("mrr", k)].append(reciprocal_rank(ranked, relevant, k))
    return per


def evaluate_sampled(
    model,
    split: SessionSplit,
    item_popularity: np.ndarray,
    n_negatives: int = 100,
    ks: tuple[int, ...] = (10, 20),
    sampler: str = "uniform",
    seed: int = 0,
    chunk_size: int = 2048,
) -> pd.DataFrame:
    """Sampled-negative metrics for one model over all test sessions."""
    per: dict[tuple[str, int], list[float]] = {
        (m, k): [] for m in METRICS for k in ks
    }
    for start in range(0, split.n_test_sessions, chunk_size):
        stop = min(start + chunk_size, split.n_test_sessions)
        histories = split.test_prefix[start:stop]
        scores = model.score(histories)
        seen = histories.toarray() > 0
        chunk = sampled_metrics_from_scores(
            scores,
            split.test_target[start:stop],
            seen,
            n_negatives=n_negatives,
            ks=ks,
            sampler=sampler,
            item_popularity=item_popularity,
            seed=seed + start,  # decorrelate chunks, stay deterministic
        )
        for key, values in chunk.items():
            per[key].extend(values)

    rows = []
    name = getattr(model, "name", type(model).__name__)
    for (metric, k), values in per.items():
        rows.append(
            {
                "model": name,
                "protocol": f"sampled_{sampler}",
                "metric": metric,
                "k": k,
                "value": float(np.mean(values)),
                "n_sessions": len(values),
                "n_negatives": n_negatives,
            }
        )
    return pd.DataFrame(rows)


def protocol_disagreement(
    full: pd.DataFrame, sampled: pd.DataFrame, metric: str = "ndcg", k: int = 20
) -> pd.DataFrame:
    """Rank models under full-catalogue vs sampled evaluation, side by side.

    Returns one row per model with both ranks and both scores. A Spearman
    correlation below 1.0 between the two rank columns *is* the Krichene-Rendle
    finding on this data: the shortcut reorders the leaderboard.
    """
    def ranked(df, protocol_value):
        sub = df[(df["metric"] == metric) & (df["k"] == k)].copy()
        sub = sub.sort_values("value", ascending=False).reset_index(drop=True)
        sub["rank"] = np.arange(1, len(sub) + 1)
        return sub.set_index("model")

    f = ranked(full, "full")
    s = ranked(sampled, "sampled")
    models = list(f.index)
    table = pd.DataFrame(
        {
            "model": models,
            "full_value": [f.loc[m, "value"] for m in models],
            "full_rank": [int(f.loc[m, "rank"]) for m in models],
            "sampled_value": [s.loc[m, "value"] for m in models],
            "sampled_rank": [int(s.loc[m, "rank"]) for m in models],
        }
    )
    table["rank_changed"] = table["full_rank"] != table["sampled_rank"]
    return table.sort_values("full_rank").reset_index(drop=True)
