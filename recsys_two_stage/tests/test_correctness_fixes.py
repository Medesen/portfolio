"""Unit tests for the synthesis-remediation fixes: two-tower in-batch false-negative
masking (P1-2), SASRec negative exclusion (P2-1), the one shared tie policy (P2-2), and
the operational next-item metric (P0-1)."""

from __future__ import annotations

import numpy as np
import torch

from reclab.evaluation.full_catalogue import operational_bootstrap_ci
from reclab.evaluation.metrics import order_by_score, top_k_from_scores
from reclab.models.sasrec import _sampled_negatives
from reclab.models.two_tower import _collision_mask


class TestCollisionMask:
    def test_duplicate_targets_are_masked_off_diagonal(self):
        # Rows 0 and 2 share target item 5; row 1 targets 9. The off-diagonal (0,2)/(2,0)
        # cells are the accidental hits that must be masked; the diagonal never is.
        mask = _collision_mask(torch.tensor([5, 9, 5]))
        assert bool(mask[0, 2]) and bool(mask[2, 0])          # duplicate -> masked
        assert not bool(mask[0, 1]) and not bool(mask[1, 2])  # genuine negative -> kept
        assert not bool(mask.diagonal().any())                # each row keeps its positive

    def test_no_duplicates_masks_nothing(self):
        assert not bool(_collision_mask(torch.tensor([1, 2, 3, 4])).any())


class TestSasrecNegatives:
    def test_negative_is_never_the_positive(self):
        rng = np.random.default_rng(0)
        pos = np.array([0, 1, 2, 3, 4] * 200)     # every id, many draws
        neg = _sampled_negatives(rng, n_items=5, pos=pos)
        assert (neg != pos).all()                 # exclusion holds
        assert neg.min() >= 0 and neg.max() <= 4  # stays in range

    def test_excluded_id_leaves_the_rest_uniform(self):
        rng = np.random.default_rng(1)
        pos = np.zeros(30000, dtype=int)          # always exclude id 0
        counts = np.bincount(_sampled_negatives(rng, n_items=4, pos=pos), minlength=4)
        assert counts[0] == 0                     # never the positive
        assert all(8000 < counts[i] < 12000 for i in (1, 2, 3))  # ~1/3 each


class TestTiePolicy:
    def test_ties_break_by_ascending_item_id_on_both_paths(self):
        # Three items tied at the top score, one clear winner. The full-catalogue path and
        # the candidate path must resolve the tie to ascending item id, identically.
        top = top_k_from_scores(np.array([[0.0, 5.0, 5.0, 5.0, 1.0]]), k=3)[0]
        assert list(top) == [1, 2, 3]             # tied 5s in id order, then the 1
        items = np.array([4, 1, 2, 3, 0])
        order = order_by_score(np.array([1.0, 5.0, 5.0, 5.0, 0.0]), items)
        assert list(items[order][:3]) == [1, 2, 3]

    def test_top_k_is_deterministic_across_calls(self):
        rng = np.random.default_rng(0)
        s = rng.integers(0, 3, size=(50, 40)).astype(float)  # heavy ties
        np.testing.assert_array_equal(
            top_k_from_scores(s, k=10), top_k_from_scores(s.copy(), k=10)
        )


class TestOperationalMetric:
    def test_cold_targets_drag_the_mean_down_by_the_warm_share(self):
        # 8 warm sessions all hit (1.0), 2 cold-target forced misses -> mean 8/10 = 0.8.
        mean, lo, hi = operational_bootstrap_ci(np.ones(8), n_forced_miss=2)
        assert mean == 0.8
        assert lo <= mean <= hi

    def test_no_forced_misses_is_the_plain_mean(self):
        vals = np.array([0.0, 0.5, 1.0, 0.25])
        mean, _, _ = operational_bootstrap_ci(vals, n_forced_miss=0)
        assert abs(mean - vals.mean()) < 1e-12
