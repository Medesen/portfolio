"""Evaluation: ranking metrics, full-catalogue / sampled / beyond-accuracy protocols,
plus the Stage 2 retrieval-ceiling and cold-start tracks."""

from reclab.evaluation.metrics import (
    evaluate_rankings,
    hit_rate_at_k,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    top_k_from_scores,
)

__all__ = [
    "evaluate_rankings",
    "hit_rate_at_k",
    "ndcg_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "top_k_from_scores",
]
