"""The reranker: it reorders within groups, respects the candidate set, trains under
both objectives, and is reproducible."""

from __future__ import annotations

import numpy as np
import pytest

from reclab.ranking.dataset import FEATURES, RankerFrame
from reclab.ranking.ranker import Reranker, e2e_per_session, e2e_top_k, evaluate_e2e


def _synthetic_frame(n_sessions=200, n_cand=20, seed=0):
    """Frames where the target's retr_score is (noisily) higher, so a ranker can
    learn to lift it — enough to exercise fit/predict/eval, not a benchmark."""
    rng = np.random.default_rng(seed)
    Xs, ys, items, sess, groups, targets = [], [], [], [], [], []
    for s in range(n_sessions):
        pos = rng.integers(0, n_cand)
        retr = rng.normal(0, 1, n_cand)
        retr[pos] += 1.5  # the target scores a bit higher on average
        feat = np.column_stack([
            retr, np.arange(n_cand), rng.random(n_cand), rng.integers(0, 2, n_cand),
            np.full(n_cand, 3), np.full(n_cand, 2), rng.integers(0, 2, n_cand), rng.random(n_cand),
        ])
        label = np.zeros(n_cand, dtype=int); label[pos] = 1
        cand = np.arange(s * n_cand, (s + 1) * n_cand)  # unique item ids per session
        Xs.append(feat); ys.append(label); items.append(cand)
        sess.append(np.full(n_cand, s)); groups.append(n_cand); targets.append(cand[pos])
    frame = RankerFrame(
        X=np.vstack(Xs), y=np.concatenate(ys), groups=np.asarray(groups),
        candidate_items=np.concatenate(items), session_row=np.concatenate(sess),
        positive_rate=float(np.concatenate(ys).mean()),
    )
    return frame, np.asarray(targets)


class TestReranker:
    @pytest.mark.parametrize("objective", ["lambdarank", "binary"])
    def test_trains_and_predicts_finite_scores(self, objective):
        frame, _ = _synthetic_frame()
        model = Reranker(objective=objective, n_estimators=30, seed=0).fit(frame)
        scores = model.predict(frame.X)
        assert scores.shape == (len(frame.y),)
        assert np.isfinite(scores).all()

    def test_reordering_never_invents_or_drops_candidates(self):
        frame, _ = _synthetic_frame()
        model = Reranker(n_estimators=30, seed=0).fit(frame)
        top = e2e_top_k(model, frame, k=5)
        for sess, ranked in top.items():
            block = frame.candidate_items[frame.session_row == sess]
            assert set(ranked).issubset(set(block))     # only real candidates
            assert len(set(ranked)) == len(ranked)        # no duplicates
            assert len(ranked) == 5

    def test_learns_to_lift_the_target_above_retrieval_only(self):
        # On data where the target's retrieval score is higher on average, the ranker
        # should at least match ranking by retr_score — a sanity floor, not a benchmark.
        frame, targets = _synthetic_frame(seed=1)
        model = Reranker(n_estimators=60, seed=0).fit(frame)
        res = evaluate_e2e(model, frame, targets, ks=(5,))
        two = res[(res.model == "two_stage") & (res.metric == "ndcg")].value.iloc[0]
        retr = res[(res.model == "retrieval_only") & (res.metric == "ndcg")].value.iloc[0]
        assert two >= retr - 0.02  # in-sample: the ranker recovers the retrieval signal

    def test_per_session_mean_matches_evaluate_e2e(self):
        # The paired-CI path (e2e_per_session) must be consistent with the reported
        # mean (evaluate_e2e), or the interval would be around a different quantity.
        frame, targets = _synthetic_frame(seed=2)
        model = Reranker(n_estimators=40, seed=0).fit(frame)
        res = evaluate_e2e(model, frame, targets, ks=(20,), label="two_stage")
        reranked = model.predict(frame.X)
        per = e2e_per_session(reranked, frame, targets, "ndcg", 20)
        reported = res[(res.model == "two_stage") & (res.metric == "ndcg")].value.iloc[0]
        assert per.mean() == pytest.approx(reported)
        assert per.shape == (frame.n_sessions,)

    def test_reproducible_under_seed(self):
        frame, _ = _synthetic_frame()
        a = Reranker(n_estimators=30, seed=0).fit(frame).predict(frame.X)
        b = Reranker(n_estimators=30, seed=0).fit(frame).predict(frame.X)
        np.testing.assert_array_equal(a, b)

    def test_rejects_bad_objective(self):
        with pytest.raises(ValueError, match="objective"):
            Reranker(objective="pointwise")


class TestGrouping:
    def test_group_sizes_sum_to_rows(self):
        frame, _ = _synthetic_frame()
        assert frame.groups.sum() == len(frame.y)

    def test_feature_importance_covers_every_feature(self):
        frame, _ = _synthetic_frame()
        model = Reranker(n_estimators=30, seed=0).fit(frame)
        imp = model.feature_importance()
        assert set(imp.index) == set(FEATURES)
