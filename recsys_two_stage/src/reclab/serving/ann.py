"""Approximate nearest-neighbour retrieval over item embeddings (HNSW / hnswlib).

ANN applies **only to embedding retrievers** — ALS and the two-tower — which score by
a dot product in a shared space. EASE and ItemKNN produce item-item similarity
structures, not a metric space over items, so fast vector search does not apply to
them. That is not a footnote: the architecture that scales in memory (linear
embeddings) is also the one that admits sublinear retrieval, so EASE's disadvantages
compound — quadratic memory, no cold start, and no ANN serving path.

`hnswlib` over `faiss-cpu`: smaller, no BLAS/OpenMP entanglement, and HNSW is the
algorithm being demonstrated either way. `faiss` is the larger-scale industrial
alternative, noted rather than depended on.

Honesty about scale: at ~14k items exact search takes single-digit milliseconds and
ANN is **unnecessary**. It is measured here because the *shape* of the trade-off is
what matters, and the crossover is a property of catalogue size — extrapolating the
operation count to a 10-million-item catalogue is arithmetic, stated as arithmetic.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


@dataclass
class ANNStats:
    ef_search: int
    recall_at_k: float       # ANN top-k vs exact brute-force top-k
    p50_ms: float
    p95_ms: float
    p99_ms: float
    build_s: float


class ANNIndex:
    """HNSW index over item embeddings; queries by user embedding (inner product)."""

    def __init__(self, item_vectors: np.ndarray, M: int = 16,
                 ef_construction: int = 200, seed: int = 0) -> None:
        self.item_vectors = np.ascontiguousarray(item_vectors, dtype=np.float32)
        self.n_items, self.dim = self.item_vectors.shape
        self.M = M
        self.ef_construction = ef_construction
        self.seed = seed
        self.index = None
        self.build_s = 0.0

    def build(self) -> "ANNIndex":
        import hnswlib

        # 'ip' = inner product, matching the dot-product scoring of the retrievers.
        index = hnswlib.Index(space="ip", dim=self.dim)
        index.init_index(max_elements=self.n_items, ef_construction=self.ef_construction,
                         M=self.M, random_seed=self.seed)
        t = time.perf_counter()
        index.add_items(self.item_vectors, np.arange(self.n_items))
        self.build_s = time.perf_counter() - t
        self.index = index
        return self

    def query(self, user_vectors: np.ndarray, k: int, ef_search: int) -> np.ndarray:
        """Approximate top-k item ids per user vector."""
        self.index.set_ef(max(ef_search, k))
        labels, _ = self.index.knn_query(
            np.ascontiguousarray(user_vectors, dtype=np.float32), k=k
        )
        return labels

    def exact_top_k(self, user_vectors: np.ndarray, k: int) -> np.ndarray:
        """Brute-force top-k by inner product — the ground truth ANN is measured against."""
        scores = user_vectors @ self.item_vectors.T
        part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
        rows = np.arange(len(user_vectors))[:, None]
        return part[rows, np.argsort(-scores[rows, part], axis=1)]


def recall_at_k(approx: np.ndarray, exact: np.ndarray) -> float:
    """Mean fraction of each user's exact top-k recovered by the ANN top-k."""
    k = exact.shape[1]
    return float(np.mean([len(set(a) & set(e)) / k for a, e in zip(approx, exact)]))


def sweep_ann(item_vectors: np.ndarray, user_vectors: np.ndarray, k: int = 200,
              ef_searches=(16, 32, 64, 128, 256), M: int = 16,
              ef_construction: int = 200, n_latency: int = 2000,
              seed: int = 0) -> tuple[list[ANNStats], np.ndarray]:
    """Build one index, sweep ``ef_search``, measuring recall and per-query latency.

    Returns the per-config stats and the exact top-k (so the caller can measure the
    end-to-end metric impact of ANN vs exact retrieval without recomputing it).
    """
    index = ANNIndex(item_vectors, M=M, ef_construction=ef_construction, seed=seed).build()
    exact = index.exact_top_k(user_vectors, k)

    rng = np.random.default_rng(seed)
    latency_rows = user_vectors[rng.integers(0, len(user_vectors), size=n_latency)]

    stats = []
    for ef in ef_searches:
        approx = index.query(user_vectors, k, ef_search=ef)
        rec = recall_at_k(approx, exact)
        # Single-threaded per-query latency over many individual queries.
        index.index.set_ef(max(ef, k))
        times = []
        for row in latency_rows:
            t = time.perf_counter()
            index.index.knn_query(row[None, :].astype(np.float32), k=k)
            times.append((time.perf_counter() - t) * 1000.0)
        times = np.array(times)
        stats.append(ANNStats(
            ef_search=ef, recall_at_k=rec,
            p50_ms=float(np.percentile(times, 50)),
            p95_ms=float(np.percentile(times, 95)),
            p99_ms=float(np.percentile(times, 99)),
            build_s=index.build_s,
        ))
    return stats, exact
