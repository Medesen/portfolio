"""Full-catalogue evaluation: rank every item, for every test session.

This is the honest protocol. The shortcut the field used for years — score the
true item against ~100 sampled negatives — is implemented separately, in
``sampled.py``, so the two can be compared. They do not always agree, which is
the point.

Per-session metric values are retained rather than averaged away immediately,
because a mean without an interval cannot support a claim that one model beats
another. Bootstrap CIs come from resampling those per-session values.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from reclab.evaluation.metrics import (
    hit_rate_at_k,
    ndcg_at_k,
    reciprocal_rank,
    top_k_from_scores,
)
from reclab.models.base import HistoryBatch
from reclab.splitting.protocols import SessionSplit


def history_chunk(split: SessionSplit, start: int, stop: int) -> HistoryBatch:
    """Build a HistoryBatch for test rows [start, stop): the binary slice plus,
    when the split carries them, the matching ordered sequences."""
    sequences = None
    if split.test_prefix_sequences is not None:
        sequences = split.test_prefix_sequences[start:stop]
    return HistoryBatch(matrix=split.test_prefix[start:stop], sequences=sequences)

METRICS = ("hit_rate", "ndcg", "mrr")


@dataclass
class EvalResult:
    """Per-session metric values for one model, plus aggregation helpers."""

    model: str
    protocol: str
    per_session: dict[tuple[str, int], np.ndarray]
    n_sessions: int

    def summary(self) -> pd.DataFrame:
        rows = []
        for (metric, k), values in sorted(self.per_session.items(), key=lambda x: (x[0][1], x[0][0])):
            rows.append(
                {
                    "model": self.model,
                    "protocol": self.protocol,
                    "metric": metric,
                    "k": k,
                    "value": float(values.mean()),
                    "n_sessions": self.n_sessions,
                }
            )
        return pd.DataFrame(rows)

    def bootstrap_ci(
        self, metric: str, k: int, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05
    ) -> tuple[float, float, float]:
        """Percentile bootstrap over sessions. Returns (mean, low, high)."""
        values = self.per_session[(metric, k)]
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(values), size=(n_boot, len(values)))
        means = values[idx].mean(axis=1)
        return (
            float(values.mean()),
            float(np.quantile(means, alpha / 2)),
            float(np.quantile(means, 1 - alpha / 2)),
        )


def evaluate(
    model,
    split: SessionSplit,
    ks: tuple[int, ...] = (10, 20, 50),
    chunk_size: int = 2048,
) -> EvalResult:
    """Score every test session against the full catalogue.

    Sessions are processed in chunks: a dense (n_sessions, n_items) score block
    for 21k sessions and 14k items would be 2.3 GB, which is pointless to
    materialise all at once.
    """
    max_k = max(ks)
    if max_k > split.n_items:
        raise ValueError(
            f"k={max_k} exceeds the catalogue ({split.n_items:,} items)"
        )

    per_session: dict[tuple[str, int], list[float]] = {
        (metric, k): [] for metric in METRICS for k in ks
    }

    for start in range(0, split.n_test_sessions, chunk_size):
        stop = min(start + chunk_size, split.n_test_sessions)
        history = history_chunk(split, start, stop)

        scores = model.score(history)
        # Items already seen in the session are not predictions. Masking is done
        # here, once, for every model — a model that forgets scores its own
        # input back and looks spectacular for no reason.
        seen = history.matrix.toarray() > 0
        ranked = top_k_from_scores(scores, max_k, mask=seen)

        for offset, row in enumerate(ranked):
            relevant = {int(split.test_target[start + offset])}
            for k in ks:
                per_session[("hit_rate", k)].append(hit_rate_at_k(row, relevant, k))
                per_session[("ndcg", k)].append(ndcg_at_k(row, relevant, k))
                per_session[("mrr", k)].append(reciprocal_rank(row, relevant, k))

    return EvalResult(
        model=getattr(model, "name", type(model).__name__),
        protocol=split.protocol,
        per_session={key: np.asarray(v, dtype=float) for key, v in per_session.items()},
        n_sessions=split.n_test_sessions,
    )
