"""Neural models must actually learn — the failure mode no shape assertion catches.

A model that trains without error but learns nothing is very hard to notice later,
so this is written and run before any training on real data. On a trivially
learnable clustered world both neural models must reach near-perfect recall and
show a decreasing loss.
"""

from __future__ import annotations

import pytest

from reclab.evaluation.full_catalogue import evaluate
from reclab.models import SASRec, TwoTower
from tests.neural_fixtures import clustered_split


def _recall_at_k(split, model, k=5):
    result = evaluate(model, split, ks=(k,))
    return result.per_session[("hit_rate", k)].mean()


class TestTwoTowerLearns:
    def test_reaches_high_recall_on_clustered_data(self):
        split, features, _, items_per = clustered_split(seed=1)
        model = TwoTower(
            n_items=split.n_items, item_features=features, emb_dim=32,
            epochs=15, batch_size=128, seed=0,
        ).fit(split)
        # The right item shares the session's cluster; recall@items_per should be
        # far above the popularity floor (items_per / n_items).
        floor = items_per / split.n_items
        assert _recall_at_k(split, model, k=items_per) > 2 * floor


class TestSASRecLearns:
    @pytest.mark.parametrize("loss", ["full_softmax", "sampled_bce"])
    def test_reaches_high_recall_on_clustered_data(self, loss):
        split, _, _, items_per = clustered_split(seed=2)
        model = SASRec(
            n_items=split.n_items, emb_dim=32, max_len=10, n_blocks=2, n_heads=2,
            dropout=0.1, loss=loss, epochs=30, batch_size=128, seed=0,
        ).fit(split)
        floor = items_per / split.n_items
        assert _recall_at_k(split, model, k=items_per) > 2 * floor


def test_two_tower_loss_decreases():
    # A crude but decisive check that optimisation is actually happening: recall
    # after training beats recall from an untrained model.
    split, features, _, items_per = clustered_split(seed=3)
    untrained = TwoTower(n_items=split.n_items, item_features=features,
                         emb_dim=32, epochs=0, seed=0).fit(split)
    trained = TwoTower(n_items=split.n_items, item_features=features,
                       emb_dim=32, epochs=15, seed=0).fit(split)
    r_untrained = evaluate(untrained, split, ks=(items_per,)).per_session[("hit_rate", items_per)].mean()
    r_trained = evaluate(trained, split, ks=(items_per,)).per_session[("hit_rate", items_per)].mean()
    assert r_trained > r_untrained
