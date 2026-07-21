"""HNSW approximate nearest-neighbour index: recall against exact, and hygiene."""

from __future__ import annotations

import numpy as np
import pytest

from reclab.serving.ann import ANNIndex, recall_at_k, sweep_ann


def _vectors(n_items=2000, n_users=200, dim=32, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.normal(size=(n_items, dim)).astype(np.float32),
            rng.normal(size=(n_users, dim)).astype(np.float32))


class TestANNIndex:
    def test_recall_against_exact_is_high(self):
        items, users = _vectors()
        index = ANNIndex(items, M=16, ef_construction=200, seed=0).build()
        exact = index.exact_top_k(users, k=50)
        approx = index.query(users, k=50, ef_search=128)
        assert recall_at_k(approx, exact) > 0.9  # HNSW recovers most of exact top-k

    def test_higher_ef_search_does_not_reduce_recall(self):
        items, users = _vectors()
        index = ANNIndex(items, seed=0).build()
        exact = index.exact_top_k(users, k=50)
        r_low = recall_at_k(index.query(users, 50, ef_search=16), exact)
        r_high = recall_at_k(index.query(users, 50, ef_search=256), exact)
        assert r_high >= r_low - 1e-9  # more search never hurts recall

    def test_results_have_no_duplicates_and_respect_k(self):
        items, users = _vectors()
        index = ANNIndex(items, seed=0).build()
        approx = index.query(users, k=20, ef_search=64)
        assert approx.shape == (len(users), 20)
        for row in approx:
            assert len(set(row.tolist())) == 20  # no duplicate items

    def test_exact_matches_a_manual_dot_product(self):
        items, users = _vectors(n_items=100, n_users=5, dim=8)
        index = ANNIndex(items, seed=0).build()
        exact = index.exact_top_k(users, k=3)
        for u in range(len(users)):
            manual = np.argsort(-(users[u] @ items.T))[:3]
            np.testing.assert_array_equal(exact[u], manual)


class TestSweep:
    def test_sweep_returns_a_stat_per_config_with_monotone_recall(self):
        items, users = _vectors()
        stats, exact = sweep_ann(items, users, k=50,
                                 ef_searches=(16, 64, 256), n_latency=100, seed=0)
        assert len(stats) == 3
        assert exact.shape == (len(users), 50)
        recalls = [s.recall_at_k for s in stats]
        assert recalls == sorted(recalls)  # non-decreasing in ef_search
        for s in stats:
            assert s.p50_ms > 0 and s.p95_ms >= s.p50_ms
