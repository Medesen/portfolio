"""Two-tower correctness: the cold-start claim and the logQ mechanism.

These assert the two architectural claims the model exists to make — that the content
path buys cold-start scoring the ID path cannot, and that the logQ correction moves
the popularity distribution of recommendations in a definite, measurable direction.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from reclab.features.item_features import ItemFeatures
from reclab.models import TwoTower
from reclab.models.base import HistoryBatch
from reclab.splitting.protocols import FilterReport, SessionSplit
from tests.neural_fixtures import clustered_split


def _cold_item(category: int, vocab, parent_vocab) -> ItemFeatures:
    return ItemFeatures(
        item_ids=np.array([999_999]),
        category_ids=np.array([category]),
        parent_ids=np.array([category]),
        available=np.array([1.0], dtype=np.float32),
        category_vocab=vocab,
        parent_vocab=parent_vocab,
    )


class TestColdStart:
    def test_content_path_lets_a_cold_item_find_its_cluster(self):
        # A brand-new item, absent from training, embedded from content alone must
        # sit closer to its own cluster's items than to the rest. Averaged over all
        # clusters so the per-item margin's noise cancels.
        split, feats, _, ipp = clustered_split(seed=1)
        model = TwoTower(n_items=split.n_items, item_features=feats, emb_dim=32,
                         epochs=15, seed=0, item_tower_mode="id_plus_content").fit(split)
        n_clusters = split.n_items // ipp
        margins = []
        for c in range(n_clusters):
            cold = _cold_item(c + 1, feats.category_vocab, feats.parent_vocab)
            emb = model.embed_cold_items(cold)[0]
            assert np.isfinite(emb).all() and np.abs(emb).sum() > 0  # non-degenerate
            sims = model.item_emb_ @ emb
            own = sims[c * ipp : (c + 1) * ipp].mean()
            others = (sims.sum() - sims[c * ipp : (c + 1) * ipp].sum()) / (
                split.n_items - ipp
            )
            margins.append(own - others)
        assert np.mean(margins) > 0  # content places cold items with their cluster

    def test_id_only_ablation_cannot_place_cold_items(self):
        # With the content path removed, the content embeddings are never trained,
        # so a cold item carries no cluster signal — the ablation that proves the
        # content path is what buys cold-start capability.
        split, feats, _, ipp = clustered_split(seed=1)
        model = TwoTower(n_items=split.n_items, item_features=feats, emb_dim=32,
                         epochs=15, seed=0, item_tower_mode="id_only").fit(split)
        margins = []
        n_clusters = split.n_items // ipp
        for c in range(n_clusters):
            emb = model.embed_cold_items(_cold_item(c + 1, feats.category_vocab, feats.parent_vocab))[0]
            sims = model.item_emb_ @ emb
            own = sims[c * ipp : (c + 1) * ipp].mean()
            others = (sims.sum() - sims[c * ipp : (c + 1) * ipp].sum()) / (split.n_items - ipp)
            margins.append(abs(own - others))
        # No meaningful discrimination: the id_plus_content margin dwarfs this.
        assert np.mean(margins) < 0.05


def _skewed_split(seed=0):
    """Sessions where a few head items dominate — so logQ has something to move."""
    rng = np.random.default_rng(seed)
    n_items, n_head = 40, 4
    seqs = []
    for _ in range(1500):
        s = []
        if rng.random() < 0.8:
            s.append(int(rng.integers(0, n_head)))
        s += list(rng.choice(np.arange(n_head, n_items), size=rng.integers(2, 4), replace=False))
        seqs.append(np.array(s, dtype=np.int64))

    def mat(ss):
        r, c = [], []
        for i, s in enumerate(ss):
            r += [i] * len(s)
            c += list(s)
        return sp.csr_matrix((np.ones(len(r), dtype=np.float32), (r, c)), shape=(len(ss), n_items))

    feats = ItemFeatures(np.arange(n_items), np.ones(n_items, dtype=np.int64),
                         np.ones(n_items, dtype=np.int64), np.ones(n_items, dtype=np.float32),
                         {1: 1}, {1: 1})
    te = seqs[:300]
    split = SessionSplit(
        train=mat(seqs), test_prefix=mat([s[:-1] for s in te]),
        test_target=np.array([s[-1] for s in te]), item_ids=np.arange(n_items),
        cutoff=None, protocol="temporal", filter_scope="train",
        filter_report=FilterReport(2, 1), train_sequences=seqs,
        test_prefix_sequences=[s[:-1] for s in te],
    )
    return split, feats


class TestLogQCorrection:
    def test_correction_shifts_recommendation_popularity(self):
        # The logQ correction removes the in-batch sampler's incidental *suppression*
        # of popular items (they appear as negatives most often), so enabling it
        # moves recommendations toward MORE popular items — the opposite of a naive
        # guess, and the theoretically correct direction. This asserts the mechanism
        # fires, not merely that the code runs.
        from reclab.evaluation.beyond_accuracy import evaluate_beyond_accuracy

        split, feats = _skewed_split(seed=0)
        pop = np.asarray(split.train.sum(axis=0)).ravel()

        def pop_percentile(logq):
            m = TwoTower(n_items=split.n_items, item_features=feats, emb_dim=32,
                         epochs=12, seed=0, logq_correction=logq).fit(split)
            return evaluate_beyond_accuracy(m, split, pop, ks=(10,)).iloc[0]["mean_pop_percentile"]

        assert pop_percentile(True) > pop_percentile(False) + 0.03


class TestInterface:
    def test_score_shape_and_finiteness(self):
        split, feats, _, _ = clustered_split(seed=0)
        model = TwoTower(n_items=split.n_items, item_features=feats, emb_dim=16,
                         epochs=2, seed=0).fit(split)
        scores = model.score(HistoryBatch(matrix=split.test_prefix[:5],
                                          sequences=split.test_prefix_sequences[:5]))
        assert scores.shape == (5, split.n_items)
        assert np.isfinite(scores).all()

    def test_is_order_agnostic_uses_the_bag_not_the_sequence(self):
        # The two-tower pools a bag, so it must ignore sequence order entirely:
        # same matrix + differently-ordered sequences -> identical scores.
        split, feats, _, _ = clustered_split(seed=0)
        model = TwoTower(n_items=split.n_items, item_features=feats, emb_dim=16,
                         epochs=2, seed=0).fit(split)
        row = split.test_prefix[:1]
        seq = split.test_prefix_sequences[0]
        a = model.score(HistoryBatch(matrix=row, sequences=[seq]))
        b = model.score(HistoryBatch(matrix=row, sequences=[seq[::-1].copy()]))
        np.testing.assert_allclose(a, b)

    def test_rejects_bad_item_tower_mode(self):
        split, feats, _, _ = clustered_split(seed=0)
        with pytest.raises(ValueError, match="item_tower_mode"):
            TwoTower(n_items=split.n_items, item_features=feats, item_tower_mode="bogus")
