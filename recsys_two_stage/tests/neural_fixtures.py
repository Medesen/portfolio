"""Synthetic fixtures for the neural-model tests.

A clustered world: items belong to clusters and each item's category *is* its
cluster. Sessions are **ordered contiguous runs** within a cluster (item k tends to
be followed by k+1), which gives the two-tower model cluster co-occurrence to pool
*and* the sequential model a genuine order to exploit. A model that trains without
error but learns nothing fails the sanity test built on this.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from reclab.features.item_features import ItemFeatures
from reclab.splitting.protocols import FilterReport, SessionSplit


def clustered_split(
    n_clusters: int = 5,
    items_per: int = 8,
    train_per_cluster: int = 120,
    test_per_cluster: int = 20,
    seed: int = 0,
) -> tuple[SessionSplit, ItemFeatures]:
    rng = np.random.default_rng(seed)
    n_items = n_clusters * items_per

    def make_sessions(count):
        seqs, cluster_of = [], []
        for c in range(n_clusters):
            base = c * items_per
            for _ in range(count):
                length = rng.integers(3, items_per)  # contiguous ascending run
                start = rng.integers(0, items_per - length + 1)
                seqs.append(np.arange(base + start, base + start + length).astype(np.int64))
                cluster_of.append(c)
        return seqs, np.array(cluster_of)

    train_seqs, _ = make_sessions(train_per_cluster)
    test_seqs, test_cluster = make_sessions(test_per_cluster)

    def to_matrix(seqs):
        rows, cols = [], []
        for r, s in enumerate(seqs):
            rows += [r] * len(s)
            cols += list(s)
        return sp.csr_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)),
            shape=(len(seqs), n_items),
        )

    train = to_matrix(train_seqs)
    prefix_seqs = [s[:-1] for s in test_seqs]
    targets = np.array([s[-1] for s in test_seqs])
    prefix = to_matrix(prefix_seqs)

    split = SessionSplit(
        train=train,
        test_prefix=prefix,
        test_target=targets,
        item_ids=np.arange(n_items),
        cutoff=None,
        protocol="temporal",
        filter_scope="train",
        filter_report=FilterReport(2, 1),
        train_sequences=train_seqs,
        test_prefix_sequences=prefix_seqs,
    )

    # Each item's category is its cluster (1-indexed; 0 reserved for unknown).
    category_ids = np.array([i // items_per + 1 for i in range(n_items)], dtype=np.int64)
    features = ItemFeatures(
        item_ids=np.arange(n_items),
        category_ids=category_ids,
        parent_ids=category_ids.copy(),
        available=np.ones(n_items, dtype=np.float32),
        category_vocab={c + 1: c + 1 for c in range(n_clusters)},
        parent_vocab={c + 1: c + 1 for c in range(n_clusters)},
    )
    return split, features, test_cluster, items_per
