"""Ranking metrics for top-N recommendation.

Each primitive scores **one user**: a ranked array of item indices (best first)
against the set of items that user actually interacted with in the held-out
period. Relevance is binary — this dataset has implicit feedback, so there are no
grades to weight by.

The Recall@k normalisation is a real choice, not a detail
-------------------------------------------------------
With more than one held-out item per user, two definitions circulate and they do
not agree:

``normalize="min_k"``     hits / min(k, |relevant|)   <- default
``normalize="relevant"``  hits / |relevant|

The second cannot reach 1.0 whenever a user has more than ``k`` held-out items,
so it silently penalises the most active users and makes scores depend on how
long the test window is. The first is what Liang et al. (2018) and Steck (2019,
EASE) use, which is the literature this project compares itself against — so it
is the default here, and every reported Recall@k in the README uses it.

Both are implemented because the gap between them is worth being able to show:
on a dataset where most users have a single held-out item they coincide exactly,
and on one where users have many they diverge sharply. Which regime you are in
is a property of the split, not of the models.

Empty relevant sets raise rather than returning NaN. A user with nothing held out
should never have reached a metric function — the evaluation loop defines its
evaluation users as those with at least one training *and* one test interaction —
so an empty set means an upstream bug, and a loud failure beats a silent NaN that
averages away to a plausible-looking number.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "recall_at_k",
    "ndcg_at_k",
    "reciprocal_rank",
    "hit_rate_at_k",
    "evaluate_rankings",
    "top_k_from_scores",
    "order_by_score",
]

# Discount weights are reused across every user and every metric call; caching
# them keeps the per-user loop from rebuilding the same log2 array N times.
_DISCOUNT_CACHE: dict[int, np.ndarray] = {}


def _discounts(n: int) -> np.ndarray:
    """1 / log2(rank + 1) for ranks 1..n, cached."""
    cached = _DISCOUNT_CACHE.get(n)
    if cached is None:
        cached = 1.0 / np.log2(np.arange(2, n + 2))
        _DISCOUNT_CACHE[n] = cached
    return cached


def _check(ranked: np.ndarray, relevant: set[int], k: int) -> int:
    """Validate inputs and return the effective cut-off."""
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if not relevant:
        raise ValueError(
            "empty relevant set — evaluation users must have at least one "
            "held-out item; this indicates an upstream split bug"
        )
    # k may exceed the ranked list (a short catalogue, or a user whose seen-item
    # mask left fewer than k candidates). Truncate rather than pad: scoring
    # positions that do not exist would credit or penalise a model for nothing.
    return min(k, len(ranked))


def recall_at_k(
    ranked: np.ndarray,
    relevant: set[int],
    k: int,
    normalize: str = "min_k",
) -> float:
    """Fraction of held-out items recovered in the top k. See module docstring."""
    cut = _check(ranked, relevant, k)
    hits = sum(1 for item in ranked[:cut] if item in relevant)
    if normalize == "min_k":
        denominator = min(k, len(relevant))
    elif normalize == "relevant":
        denominator = len(relevant)
    else:
        raise ValueError(
            f"normalize must be 'min_k' or 'relevant', got {normalize!r}"
        )
    return hits / denominator


def ndcg_at_k(ranked: np.ndarray, relevant: set[int], k: int) -> float:
    """Normalised discounted cumulative gain with binary relevance.

    The ideal ranking places min(k, |relevant|) hits in the top positions, so
    NDCG reaches 1.0 exactly when every held-out item that *could* fit in the
    top k is there.
    """
    cut = _check(ranked, relevant, k)
    discounts = _discounts(k)
    gains = np.fromiter(
        (1.0 if item in relevant else 0.0 for item in ranked[:cut]),
        dtype=float,
        count=cut,
    )
    dcg = float(gains @ discounts[:cut])
    ideal_hits = min(k, len(relevant))
    idcg = float(discounts[:ideal_hits].sum())
    return dcg / idcg


def reciprocal_rank(
    ranked: np.ndarray, relevant: set[int], k: int | None = None
) -> float:
    """1 / (rank of the first hit), or 0.0 if there is no hit within the cut-off.

    ``k=None`` searches the whole ranked list. Passing a k makes this MRR@k,
    which is what the README reports — an unbounded MRR rewards a model for
    burying the right answer at position 900 instead of 2000, which no user will
    ever see.
    """
    cut = len(ranked) if k is None else _check(ranked, relevant, k)
    if k is None and not relevant:
        raise ValueError("empty relevant set")
    for position, item in enumerate(ranked[:cut], start=1):
        if item in relevant:
            return 1.0 / position
    return 0.0


def hit_rate_at_k(ranked: np.ndarray, relevant: set[int], k: int) -> float:
    """1.0 if any held-out item appears in the top k, else 0.0."""
    cut = _check(ranked, relevant, k)
    return float(any(item in relevant for item in ranked[:cut]))


def order_by_score(scores: np.ndarray, items: np.ndarray) -> np.ndarray:
    """Order indices by ``scores`` descending, ties broken by ascending ``items`` (item id).

    The project's single tie policy, shared by the sampled-negative and reranked paths so a
    model that ties the target against other candidates is neither rewarded nor penalised by
    an implicit, implementation-dependent order. Two stable passes: sort by item id, then a
    stable sort by ``-score`` preserves that id order within each score tie. (The
    full-catalogue path in ``top_k_from_scores`` gets the same policy for free, because its
    columns are already laid out in item-id order.)
    """
    scores = np.asarray(scores, dtype=float)
    items = np.asarray(items)
    tie = np.argsort(items, kind="stable")
    return tie[np.argsort(-scores[tie], kind="stable")]


def top_k_from_scores(
    scores: np.ndarray, k: int, mask: np.ndarray | None = None
) -> np.ndarray:
    """Top-k item indices per row of ``scores``, best first.

    ``mask`` is a boolean array marking positions to exclude (already-seen
    items). Masking happens here, once, rather than inside each model — a model
    that forgets to drop a user's training history scores spectacularly and
    meaninglessly, so the exclusion is the evaluation loop's job and a test
    asserts it holds for every model.

    Ties are broken deterministically by ascending item index (the project's single tie
    policy): a stable sort of ``-scores`` leaves equal-scoring items in column — i.e. item
    id — order. Sparse recommenders produce many exact ties, so without a fixed policy the
    boundary of the top-k would be implementation-dependent and protocol comparisons could
    reflect the tie order rather than the models.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2:
        raise ValueError(f"scores must be 2-D (users x items), got {scores.shape}")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    if mask is not None:
        if mask.shape != scores.shape:
            raise ValueError(
                f"mask shape {mask.shape} does not match scores {scores.shape}"
            )
        # Copy before mutating: callers reuse score matrices across k values.
        scores = np.where(mask, -np.inf, scores)

    cut = min(k, scores.shape[1])
    # Stable sort of -scores keeps equal scores in ascending column (item-id) order — the
    # one tie policy. O(n log n) per row, but full-catalogue evaluation is not the pipeline
    # bottleneck and cross-run/protocol determinism is worth more than argpartition's O(n).
    order = np.argsort(-scores, axis=1, kind="stable")
    return order[:, :cut]


def evaluate_rankings(
    top_k: np.ndarray,
    relevant: Sequence[set[int]],
    ks: Iterable[int],
    normalize: str = "min_k",
) -> pd.DataFrame:
    """Aggregate metrics over users.

    ``top_k`` is (n_users, K) with K >= max(ks); ``relevant[i]`` is the held-out
    item set for row i. Returns one row per k with the mean of each metric and
    the user count behind it.

    Standalone, unit-tested primitive: the production path (``full_catalogue.evaluate``)
    keeps per-session values for bootstrap CIs and does not call this. It is retained
    because it is the only place that exercises the ``normalize`` (min_k vs |relevant|)
    recall convention against multi-item relevant sets.
    """
    if len(top_k) != len(relevant):
        raise ValueError(
            f"{len(top_k)} ranking rows but {len(relevant)} relevant sets"
        )
    if len(top_k) == 0:
        raise ValueError("no evaluation users")

    ks = sorted(ks)
    if max(ks) > top_k.shape[1]:
        raise ValueError(
            f"requested k={max(ks)} but only {top_k.shape[1]} ranked items per "
            "user were supplied"
        )

    rows = []
    for k in ks:
        recalls, ndcgs, rrs, hits = [], [], [], []
        for ranked, rel in zip(top_k, relevant):
            recalls.append(recall_at_k(ranked, rel, k, normalize=normalize))
            ndcgs.append(ndcg_at_k(ranked, rel, k))
            rrs.append(reciprocal_rank(ranked, rel, k))
            hits.append(hit_rate_at_k(ranked, rel, k))
        rows.append(
            {
                "k": k,
                "recall": float(np.mean(recalls)),
                "ndcg": float(np.mean(ndcgs)),
                "mrr": float(np.mean(rrs)),
                "hit_rate": float(np.mean(hits)),
                "n_eval_users": len(top_k),
            }
        )
    return pd.DataFrame(rows)
