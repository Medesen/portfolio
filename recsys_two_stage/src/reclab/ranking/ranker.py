"""The reranker: LightGBM over retrieved candidates, plus end-to-end evaluation.

Two objectives are offered so the listwise-vs-pointwise question can be answered on
this data rather than assumed:

- ``lambdarank`` — the pairwise/listwise LambdaMART objective, grouped by session.
- ``binary`` — a plain classifier on the same features, ranked by predicted
  probability. Pointwise ranking is often close to LambdaMART in practice; whether the
  listwise objective earns its complexity here is one ablation row.

The end-to-end evaluation reorders each session's retrieved candidates by the ranker's
score and scores the result with the *same* HitRate/NDCG/MRR@k used for every
single-stage model, so the two-stage number drops straight into the Stage 1 table.
The retrieval-only baseline (rank by the retriever's own score) is reported alongside,
because the honest question is whether reranking beats simply trusting the retriever.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from reclab.evaluation.metrics import (
    hit_rate_at_k,
    ndcg_at_k,
    order_by_score,
    reciprocal_rank,
)
from reclab.ranking.dataset import FEATURES, RankerFrame


class Reranker:
    name = "reranker"

    def __init__(self, objective: str = "lambdarank", n_estimators: int = 300,
                 learning_rate: float = 0.05, num_leaves: int = 31,
                 min_child_samples: int = 50, seed: int = 0) -> None:
        if objective not in ("lambdarank", "binary"):
            raise ValueError(f"objective must be lambdarank or binary, got {objective}")
        self.objective = objective
        self.n_estimators = n_estimators
        # Native LightGBM params (the sklearn wrapper would pull in scikit-learn).
        self.params = {
            "objective": objective,
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "min_data_in_leaf": min_child_samples,
            "seed": seed,
            "num_threads": 4,
            "verbosity": -1,
        }
        if objective == "lambdarank":
            self.params["label_gain"] = [0, 1]        # binary relevance
            self.params["ndcg_eval_at"] = [10, 20]
        self.model_ = None

    def fit(self, frame: RankerFrame) -> "Reranker":
        import lightgbm as lgb

        dataset = lgb.Dataset(frame.X, label=frame.y, feature_name=list(FEATURES),
                              free_raw_data=False)
        if self.objective == "lambdarank":
            dataset.set_group(frame.groups)
        self.model_ = lgb.train(self.params, dataset, num_boost_round=self.n_estimators)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Reranker not fitted")
        return self.model_.predict(X)  # binary -> probability; lambdarank -> raw score

    def feature_importance(self) -> pd.Series:
        gains = self.model_.feature_importance(importance_type="gain")
        return pd.Series(gains, index=FEATURES).sort_values(ascending=False)


def _score_blocks(scores: np.ndarray, frame: RankerFrame):
    """Yield (session_index, candidate_items, candidate_scores) per group."""
    offset = 0
    for g in frame.groups:
        sl = slice(offset, offset + g)
        yield frame.session_row[offset], frame.candidate_items[sl], scores[sl]
        offset += g


def evaluate_e2e(reranker: Reranker, frame: RankerFrame, targets: np.ndarray,
                 ks=(10, 20), label: str = "two_stage") -> pd.DataFrame:
    """End-to-end metrics: rerank each session's candidates, score the ordering.

    Reported against the retrieval-only ordering (rank by the retriever's own score,
    which is feature column 0) so the reranker's contribution is isolated.
    """
    reranked = reranker.predict(frame.X)
    retr_score = frame.X[:, FEATURES.index("retr_score")]

    rows = []
    for name, scores in [(label, reranked), ("retrieval_only", retr_score)]:
        per = {("hit_rate", k): [] for k in ks}
        per.update({("ndcg", k): [] for k in ks})
        per.update({("mrr", k): [] for k in ks})
        for sess, items, sc in _score_blocks(scores, frame):
            ranked = items[order_by_score(sc, items)]
            relevant = {int(targets[sess])}
            for k in ks:
                per[("hit_rate", k)].append(hit_rate_at_k(ranked, relevant, k))
                per[("ndcg", k)].append(ndcg_at_k(ranked, relevant, k))
                per[("mrr", k)].append(reciprocal_rank(ranked, relevant, k))
        for (metric, k), vals in per.items():
            rows.append({"model": name, "metric": metric, "k": k,
                         "value": float(np.mean(vals)), "n_sessions": len(vals)})
    return pd.DataFrame(rows)


def e2e_per_session(
    scores: np.ndarray, frame: RankerFrame, targets: np.ndarray, metric: str, k: int
) -> np.ndarray:
    """Per-session metric values for one candidate ordering (given its ``scores``).

    Iterates ``frame``'s session blocks in a fixed order, so arrays returned for
    different score vectors (reranked vs retrieval-only, lambdarank vs pointwise) are
    session-aligned and can be fed to a *paired* bootstrap — the honest way to ask
    whether reranking, or the listwise objective, actually resolves a difference.
    """
    metric_fn = {"hit_rate": hit_rate_at_k, "ndcg": ndcg_at_k, "mrr": reciprocal_rank}[metric]
    vals = []
    for sess, items, sc in _score_blocks(scores, frame):
        ranked = items[order_by_score(sc, items)]
        vals.append(metric_fn(ranked, {int(targets[sess])}, k))
    return np.asarray(vals, dtype=float)


def e2e_top_k(reranker: Reranker, frame: RankerFrame, k: int) -> dict[int, np.ndarray]:
    """Per session, the reranked top-k item indices — for beyond-accuracy metrics."""
    scores = reranker.predict(frame.X)
    out = {}
    for sess, items, sc in _score_blocks(scores, frame):
        out[sess] = items[order_by_score(sc, items)[:k]]
    return out
