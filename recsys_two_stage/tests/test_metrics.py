"""Metric correctness against hand-computed values.

Expected values are written as arithmetic expressions rather than decimal
literals so a reader can check the formula, not just the number.
"""

from __future__ import annotations

import numpy as np
import pytest

from reclab.evaluation.metrics import (
    evaluate_rankings,
    hit_rate_at_k,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    top_k_from_scores,
)

# Worked example used across several tests:
#   ranked   = [3, 1, 4, 2, 0]   (best first)
#   relevant = {1, 4}
#   -> item 1 sits at rank 2, item 4 at rank 3
RANKED = np.array([3, 1, 4, 2, 0])
RELEVANT = {1, 4}

D2 = 1.0 / np.log2(3)  # discount at rank 2
D3 = 1.0 / np.log2(4)  # discount at rank 3


class TestRecall:
    def test_partial_hit(self):
        # top-2 = [3, 1] -> one hit; min(k, |relevant|) = min(2, 2) = 2
        assert recall_at_k(RANKED, RELEVANT, 2) == pytest.approx(1 / 2)

    def test_all_hits_within_k(self):
        # top-3 = [3, 1, 4] -> both held-out items recovered
        assert recall_at_k(RANKED, RELEVANT, 3) == pytest.approx(2 / 2)

    def test_no_hits(self):
        assert recall_at_k(RANKED, RELEVANT, 1) == pytest.approx(0.0)

    def test_normalisations_agree_when_relevant_fits_in_k(self):
        # |relevant| = 2 <= k = 3, so min(k, |R|) == |R| and the two coincide.
        assert recall_at_k(RANKED, RELEVANT, 3, normalize="min_k") == recall_at_k(
            RANKED, RELEVANT, 3, normalize="relevant"
        )

    def test_normalisations_diverge_when_relevant_exceeds_k(self):
        # The choice documented in metrics.py, made visible: with three held-out
        # items and k=2, a model that puts two of them in the top 2 has done
        # everything possible. min_k credits that as 1.0; the other definition
        # caps it at 2/3 purely because the user was active.
        ranked = np.array([0, 1, 2, 3])
        relevant = {0, 1, 2}
        assert recall_at_k(ranked, relevant, 2, normalize="min_k") == pytest.approx(1.0)
        assert recall_at_k(ranked, relevant, 2, normalize="relevant") == pytest.approx(
            2 / 3
        )

    def test_rejects_unknown_normalisation(self):
        with pytest.raises(ValueError, match="normalize"):
            recall_at_k(RANKED, RELEVANT, 2, normalize="whatever")


class TestNDCG:
    def test_partial_hit(self):
        # DCG@2 = 0/log2(2) + 1/log2(3);  IDCG@2 = 1/log2(2) + 1/log2(3)
        assert ndcg_at_k(RANKED, RELEVANT, 2) == pytest.approx(D2 / (1.0 + D2))

    def test_both_hits(self):
        # DCG@3 = 1/log2(3) + 1/log2(4);  ideal still places 2 hits at ranks 1-2
        assert ndcg_at_k(RANKED, RELEVANT, 3) == pytest.approx((D2 + D3) / (1.0 + D2))

    def test_perfect_ranking_scores_one(self):
        assert ndcg_at_k(np.array([0, 1, 2, 3]), {0, 1}, 2) == pytest.approx(1.0)

    def test_ideal_is_capped_by_k(self):
        # Three held-out items but k=2: the best achievable is two hits in the
        # top two, which must score exactly 1.0 rather than 2/3.
        assert ndcg_at_k(np.array([0, 1, 2, 3]), {0, 1, 2}, 2) == pytest.approx(1.0)

    def test_no_hits_scores_zero(self):
        assert ndcg_at_k(RANKED, RELEVANT, 1) == pytest.approx(0.0)


class TestReciprocalRank:
    def test_first_hit_position(self):
        assert reciprocal_rank(RANKED, RELEVANT, 5) == pytest.approx(1 / 2)

    def test_zero_when_first_hit_is_beyond_k(self):
        assert reciprocal_rank(RANKED, RELEVANT, 1) == pytest.approx(0.0)

    def test_unbounded_search_finds_late_hit(self):
        # Without a cut-off the hit at rank 4 counts; with k=2 it would not.
        ranked = np.array([0, 1, 2, 3])
        assert reciprocal_rank(ranked, {3}) == pytest.approx(1 / 4)
        assert reciprocal_rank(ranked, {3}, 2) == pytest.approx(0.0)


class TestHitRate:
    def test_hit(self):
        assert hit_rate_at_k(RANKED, RELEVANT, 2) == 1.0

    def test_miss(self):
        assert hit_rate_at_k(RANKED, RELEVANT, 1) == 0.0


class TestEdgeCases:
    def test_empty_relevant_set_raises_loudly(self):
        # A user with nothing held out should never reach a metric function.
        # Returning NaN here would average away into a plausible-looking score.
        for fn in (recall_at_k, ndcg_at_k, hit_rate_at_k):
            with pytest.raises(ValueError, match="empty relevant set"):
                fn(RANKED, set(), 2)

    def test_non_positive_k_raises(self):
        with pytest.raises(ValueError, match="k must be positive"):
            recall_at_k(RANKED, RELEVANT, 0)

    def test_k_larger_than_ranked_list_truncates(self):
        # Two candidates, k=10: scoring the eight positions that do not exist
        # would credit the model for nothing. Denominator is min(10, |R|) = 1.
        ranked = np.array([0, 1])
        assert recall_at_k(ranked, {1}, 10) == pytest.approx(1.0)
        assert ndcg_at_k(ranked, {1}, 10) == pytest.approx(D2 / 1.0)
        assert hit_rate_at_k(ranked, {1}, 10) == 1.0

    def test_worst_possible_ranking(self):
        ranked = np.array([0, 1, 2, 3])
        assert recall_at_k(ranked, {3}, 2) == pytest.approx(0.0)
        assert ndcg_at_k(ranked, {3}, 2) == pytest.approx(0.0)
        assert hit_rate_at_k(ranked, {3}, 2) == 0.0


class TestTopKFromScores:
    def test_orders_best_first(self):
        scores = np.array([[0.1, 0.9, 0.5]])
        np.testing.assert_array_equal(top_k_from_scores(scores, 2), [[1, 2]])

    def test_mask_excludes_seen_items(self):
        # Item 1 has the top score but is masked, so it must not appear at all.
        scores = np.array([[0.1, 0.9, 0.5]])
        mask = np.array([[False, True, False]])
        np.testing.assert_array_equal(top_k_from_scores(scores, 2, mask), [[2, 0]])

    def test_mask_does_not_mutate_caller_scores(self):
        # Score matrices get reused across k values; masking in place would
        # silently corrupt every subsequent call.
        scores = np.array([[0.1, 0.9, 0.5]])
        original = scores.copy()
        top_k_from_scores(scores, 2, np.array([[False, True, False]]))
        np.testing.assert_array_equal(scores, original)

    def test_multiple_users(self):
        scores = np.array([[0.1, 0.9, 0.5], [0.8, 0.2, 0.3]])
        np.testing.assert_array_equal(top_k_from_scores(scores, 2), [[1, 2], [0, 2]])

    def test_k_beyond_catalogue_truncates(self):
        scores = np.array([[0.1, 0.9]])
        assert top_k_from_scores(scores, 10).shape == (1, 2)

    def test_shape_validation(self):
        with pytest.raises(ValueError, match="2-D"):
            top_k_from_scores(np.array([0.1, 0.9]), 1)
        with pytest.raises(ValueError, match="mask shape"):
            top_k_from_scores(np.zeros((1, 3)), 1, np.zeros((1, 2), dtype=bool))


class TestEvaluateRankings:
    def test_aggregates_across_users(self):
        # User A: hit at rank 1. User B: hit at rank 2.
        top_k = np.array([[0, 1], [1, 0]])
        relevant = [{0}, {0}]
        result = evaluate_rankings(top_k, relevant, ks=[1, 2])

        at_1 = result[result["k"] == 1].iloc[0]
        assert at_1["recall"] == pytest.approx(1 / 2)  # A hits, B misses
        assert at_1["mrr"] == pytest.approx(1 / 2)
        assert at_1["n_eval_users"] == 2

        at_2 = result[result["k"] == 2].iloc[0]
        assert at_2["recall"] == pytest.approx(1.0)  # both found by rank 2
        assert at_2["mrr"] == pytest.approx((1.0 + 0.5) / 2)
        assert at_2["ndcg"] == pytest.approx((1.0 + D2) / 2)

    def test_columns_and_row_count(self):
        result = evaluate_rankings(np.array([[0, 1]]), [{0}], ks=[1, 2])
        assert list(result["k"]) == [1, 2]
        assert set(result.columns) == {
            "k",
            "recall",
            "ndcg",
            "mrr",
            "hit_rate",
            "n_eval_users",
        }

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="relevant sets"):
            evaluate_rankings(np.array([[0, 1], [1, 0]]), [{0}], ks=[1])

    def test_k_wider_than_supplied_rankings_raises(self):
        # Asking for NDCG@50 from a 10-wide ranking would silently truncate and
        # report an optimistic number under a misleading label.
        with pytest.raises(ValueError, match="only 2 ranked items"):
            evaluate_rankings(np.array([[0, 1]]), [{0}], ks=[5])

    def test_no_users_raises(self):
        with pytest.raises(ValueError, match="no evaluation users"):
            evaluate_rankings(np.empty((0, 2), dtype=int), [], ks=[1])
