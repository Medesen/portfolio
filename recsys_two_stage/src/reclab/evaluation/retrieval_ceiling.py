"""Retrieval-ceiling analysis — how much a two-stage system *could* get right.

A two-stage recommender's ranker can only reorder the candidates retrieval handed
it. If the target is not in the candidate set, no ranker recovers it. So Recall@N at
the candidate-set sizes a system actually retrieves (hundreds to a couple of
thousand) is a hard *ceiling* on the whole system — a more useful description of a
retriever than NDCG@10, and the number Stage 3's reranker will be measured against.

This also computes the ceiling of **blended** retrievers: the union of two
retrievers' candidate sets at a fixed budget. Production systems blend sources
precisely because their misses differ; whether that holds here is cheap to check and
directly informs Stage 3.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from reclab.evaluation.full_catalogue import history_chunk


def _top_n_per_session(model, split, max_n: int, chunk_size: int = 1024) -> np.ndarray:
    """Rank each test session's items and return the top ``max_n`` per session.

    Seen items are masked exactly as in the accuracy evaluation, so the candidate
    sets are what a deployed retriever would actually return.
    """
    from reclab.evaluation.metrics import top_k_from_scores

    out = np.empty((split.n_test_sessions, max_n), dtype=np.int64)
    for start in range(0, split.n_test_sessions, chunk_size):
        stop = min(start + chunk_size, split.n_test_sessions)
        history = history_chunk(split, start, stop)
        scores = model.score(history)
        seen = history.matrix.toarray() > 0
        out[start:stop] = top_k_from_scores(scores, max_n, mask=seen)
    return out


def _recall_at_ns(top_n: np.ndarray, targets: np.ndarray, ns: list[int]) -> dict[int, float]:
    """Recall@N per N: with a single held-out target, the fraction of sessions
    whose target appears in the top N (i.e. hit-rate@N)."""
    # Position of each target within the ranking, or max_n if absent.
    hit_rank = np.full(len(targets), top_n.shape[1] + 1, dtype=np.int64)
    for row in range(len(targets)):
        found = np.where(top_n[row] == targets[row])[0]
        if len(found):
            hit_rank[row] = found[0] + 1  # 1-indexed rank
    return {n: float((hit_rank <= n).mean()) for n in ns}


def retrieval_ceiling(
    models: dict[str, object], split, ns: tuple[int, ...] = (50, 100, 200, 500, 1000, 2000)
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Recall@N for every retriever across candidate-set sizes.

    Returns the long-form table and, for reuse in blending, each model's top-max(N)
    per session.
    """
    ns = sorted(ns)
    max_n = min(max(ns), split.n_items)
    ns = [n for n in ns if n <= split.n_items]

    rankings, rows = {}, []
    for name, model in models.items():
        top_n = _top_n_per_session(model, split, max_n)
        rankings[name] = top_n
        recalls = _recall_at_ns(top_n, split.test_target, ns)
        for n in ns:
            rows.append({"model": name, "n": n, "recall": recalls[n]})
    return pd.DataFrame(rows), rankings


def blend_ceiling(
    rankings: dict[str, np.ndarray], targets: np.ndarray, budget: int
) -> pd.DataFrame:
    """Recall of single retrievers, every pair, and the union of all, at a per-source
    budget. A blend takes the top ``budget`` from each of its sources and unions them,
    so a pair can retrieve up to ``2 * budget`` distinct candidates.
    """
    names = list(rankings)

    def union_recall(members: list[str]) -> float:
        hits = 0
        for row in range(len(targets)):
            candidates = set()
            for m in members:
                candidates.update(rankings[m][row, :budget].tolist())
            hits += int(targets[row] in candidates)
        return hits / len(targets)

    rows = [{"blend": name, "sources": 1, "budget": budget, "recall": union_recall([name])}
            for name in names]
    for a, b in itertools.combinations(names, 2):
        rows.append({"blend": f"{a}+{b}", "sources": 2, "budget": budget,
                     "recall": union_recall([a, b])})
    rows.append({"blend": "all", "sources": len(names), "budget": budget,
                 "recall": union_recall(names)})
    return pd.DataFrame(rows).sort_values("recall", ascending=False).reset_index(drop=True)
