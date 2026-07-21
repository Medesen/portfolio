"""Beyond-accuracy metrics and the tuning harness."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from reclab.evaluation.beyond_accuracy import beyond_accuracy_metrics, gini
from reclab.tuning.grid import grid_search, validation_split


class TestGini:
    def test_perfectly_even_is_zero(self):
        assert gini(np.array([5, 5, 5, 5])) == pytest.approx(0.0)

    def test_maximally_concentrated_approaches_one(self):
        # All exposure on one of 1000 items.
        counts = np.zeros(1000)
        counts[0] = 1000
        assert gini(counts) > 0.99

    def test_nothing_recommended_is_zero(self):
        assert gini(np.zeros(10)) == pytest.approx(0.0)

    def test_known_value(self):
        # Two items, one with all exposure: Gini = (n-1)/n = 1/2 for n=2.
        assert gini(np.array([0.0, 10.0])) == pytest.approx(0.5)


class TestBeyondAccuracy:
    def test_full_coverage_when_every_item_recommended(self):
        # 4 sessions, top-2 each, together covering all 8 items.
        top_k = np.array([[0, 1], [2, 3], [4, 5], [6, 7]])
        pop = np.ones(8)
        m = beyond_accuracy_metrics(top_k, n_items=8, item_popularity=pop, k=2)
        assert m["coverage"] == pytest.approx(1.0)

    def test_bestseller_only_model_has_low_coverage_high_popularity(self):
        # Every session recommends the same two most-popular items.
        top_k = np.tile([0, 1], (50, 1))
        pop = np.array([100.0, 99.0] + [1.0] * 8)  # items 0,1 are the head
        m = beyond_accuracy_metrics(top_k, n_items=10, item_popularity=pop, k=2)
        assert m["coverage"] == pytest.approx(2 / 10)
        assert m["mean_pop_percentile"] > 0.9  # recommends the popular head

    def test_k_beyond_supplied_raises(self):
        with pytest.raises(ValueError, match="exceeds"):
            beyond_accuracy_metrics(np.array([[0, 1]]), 5, np.ones(5), k=5)


def dense_log(n_sessions, n_items, days, seed=0):
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2015-06-01", tz="UTC")
    rows = []
    for s in range(n_sessions):
        day = rng.integers(0, days)
        start = base + pd.Timedelta(days=int(day), minutes=int(rng.integers(0, 300)))
        for it in rng.choice(n_items, size=rng.integers(2, 6), replace=False):
            rows.append({"session": s, "itemid": int(it),
                         "ts": start + pd.Timedelta(minutes=int(it))})
    return pd.DataFrame(rows)


class TestValidationSplit:
    def test_validation_window_sits_inside_training_period(self):
        log = dense_log(500, 30, days=60, seed=1)
        real_cutoff = log["ts"].max().normalize() - pd.Timedelta(days=14)
        val = validation_split(log, test_days=14, val_days=14, min_item_sessions=5)
        # The validation cutoff must be strictly before the real test cutoff, so
        # no test-period information can reach the tuning.
        assert val.cutoff < real_cutoff


class TestGridSearch:
    def test_selects_the_best_validation_score(self):
        # A dummy model whose validation NDCG is engineered to peak at a known
        # parameter value, so we can assert the search finds the argmax.
        class DummySplit:
            protocol = "temporal"
            n_items = 3
            train = None  # Dummy.fit ignores it

        class Dummy:
            name = "dummy"

            def __init__(self, strength):
                self.strength = strength

            def fit(self, X):
                return self

            def score(self, histories):
                pass

        target = 7

        def fake_evaluate(model, split, ks):
            # Score is highest when strength == target; expose it via the metric.
            from reclab.evaluation.full_catalogue import EvalResult

            value = -abs(model.strength - target)
            return EvalResult(
                model="dummy",
                protocol="temporal",
                per_session={("ndcg", 20): np.array([float(value)])},
                n_sessions=1,
            )

        import reclab.tuning.grid as grid_mod

        original = grid_mod.evaluate
        grid_mod.evaluate = fake_evaluate
        try:
            result = grid_search(
                Dummy, {"strength": [1, 5, 7, 9, 12]}, DummySplit(), metric="ndcg", k=20
            )
        finally:
            grid_mod.evaluate = original

        assert result.best_params == {"strength": target}
        assert len(result.table) == 5
