"""SASRec correctness — causal masking is the one that matters.

A causal-masking bug lets the model attend to the item it is trying to predict and
produces spectacular, meaningless accuracy. This is the transformer-shaped analogue
of Stage 1's temporal-leakage test, and the single most important test in Stage 2.
"""

from __future__ import annotations

import numpy as np
import torch

from reclab.models import SASRec
from reclab.models.base import HistoryBatch
from reclab.models.sasrec import _SASRecNet
from tests.neural_fixtures import clustered_split


class TestCausalMasking:
    def test_future_positions_cannot_affect_earlier_outputs(self):
        # Feed a sequence, then change the item at position t; every output at a
        # position < t must be bit-identical, because causal attention forbids
        # position s < t from seeing position t.
        torch.manual_seed(0)
        net = _SASRecNet(n_items=50, emb_dim=32, max_len=8, n_blocks=2, n_heads=2, dropout=0.0)
        net.eval()

        seq = torch.tensor([[3, 7, 12, 25, 41, 5, 9, 2]])  # all real (right-padded full)
        t = 5
        with torch.no_grad():
            out_a = net(seq)
            altered = seq.clone()
            altered[0, t] = 44  # change a later position
            out_b = net(altered)

        # Positions before t: unchanged. Position t and after: allowed to change.
        torch.testing.assert_close(out_a[0, :t], out_b[0, :t])
        assert not torch.allclose(out_a[0, t], out_b[0, t])

    def test_changing_last_position_leaves_all_earlier_unchanged(self):
        torch.manual_seed(1)
        net = _SASRecNet(n_items=50, emb_dim=16, max_len=6, n_blocks=1, n_heads=2, dropout=0.0)
        net.eval()
        seq = torch.tensor([[1, 2, 3, 4, 5, 6]])
        with torch.no_grad():
            a = net(seq)
            seq2 = seq.clone()
            seq2[0, -1] = 40
            b = net(seq2)
        torch.testing.assert_close(a[0, :-1], b[0, :-1])


class TestPaddingAndShortSequences:
    def test_single_item_history_produces_valid_prediction(self):
        split, _, _, _ = clustered_split(seed=0)
        model = SASRec(n_items=split.n_items, emb_dim=16, max_len=10,
                       epochs=2, seed=0).fit(split)
        # A session with a one-item prefix must score without crashing or NaN.
        one = HistoryBatch(matrix=split.test_prefix[:1],
                           sequences=[np.array([3], dtype=np.int64)])
        scores = model.score(one)
        assert scores.shape == (1, split.n_items)
        assert np.isfinite(scores).all()

    def test_empty_history_is_handled(self):
        split, _, _, _ = clustered_split(seed=0)
        model = SASRec(n_items=split.n_items, emb_dim=16, epochs=1, seed=0).fit(split)
        empty = HistoryBatch(matrix=split.test_prefix[:1],
                             sequences=[np.array([], dtype=np.int64)])
        scores = model.score(empty)
        assert scores.shape == (1, split.n_items)
        assert np.isfinite(scores).all()


class TestBothLossesTrain:
    def test_both_losses_produce_finite_scores(self):
        split, _, _, _ = clustered_split(seed=0)
        for loss in ("full_softmax", "sampled_bce"):
            model = SASRec(n_items=split.n_items, emb_dim=16, epochs=2,
                           loss=loss, seed=0).fit(split)
            scores = model.score(
                HistoryBatch(matrix=split.test_prefix[:4],
                             sequences=split.test_prefix_sequences[:4])
            )
            assert np.isfinite(scores).all()

    def test_rejects_unknown_loss(self):
        import pytest
        with pytest.raises(ValueError, match="loss must be"):
            SASRec(n_items=10, loss="hinge")
