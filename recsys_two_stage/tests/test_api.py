"""The /recommend endpoint: contract and the per-stage timing structure.

Timing *values* are never asserted — those are flaky and machine-dependent. What is
asserted is structure: the per-stage timings exist and sum to the reported total, and
the endpoint validates input rather than 500-ing.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from reclab.ranking.dataset import RankerFrame
from reclab.ranking.ranker import Reranker
from reclab.serving.ann import ANNIndex
from reclab.serving.pipeline import RecommenderService


def _synthetic_service(n_items=500, dim=16, seed=0, n_candidates=100) -> RecommenderService:
    rng = np.random.default_rng(seed)
    item_factors = rng.normal(size=(n_items, dim)).astype(np.float64)
    # A tiny reranker trained on a synthetic frame so predict() works.
    n_sess, n_cand = 100, 20
    Xs, ys, groups = [], [], []
    for _ in range(n_sess):
        pos = rng.integers(0, n_cand)
        retr = rng.normal(size=n_cand); retr[pos] += 1.5
        feat = np.column_stack([retr, np.arange(n_cand), rng.random(n_cand),
                                rng.integers(0, 2, n_cand), np.full(n_cand, 3),
                                np.full(n_cand, 2), rng.integers(0, 2, n_cand), rng.random(n_cand)])
        lab = np.zeros(n_cand, int); lab[pos] = 1
        Xs.append(feat); ys.append(lab); groups.append(n_cand)
    frame = RankerFrame(np.vstack(Xs), np.concatenate(ys), np.asarray(groups),
                        np.zeros(n_sess * n_cand, int), np.zeros(n_sess * n_cand, int), 0.05)
    reranker = Reranker(n_estimators=20, seed=0).fit(frame)

    ann = ANNIndex(item_factors, M=16, ef_construction=100, seed=0).build()
    return RecommenderService(
        item_ids=np.arange(n_items), item_factors=item_factors,
        YtY=item_factors.T @ item_factors, alpha=40.0, reg=1.0,
        raw_cat=rng.integers(0, 10, n_items), available=rng.integers(0, 2, n_items).astype(float),
        item_pop=rng.random(n_items), ann=ann, reranker=reranker.model_, n_candidates=n_candidates,
    )


@pytest.fixture
def client():
    import reclab.serving.app as app_module
    app_module._service = _synthetic_service()
    return TestClient(app_module.app)


class TestRecommendContract:
    def test_returns_ranked_items_with_scores(self, client):
        r = client.post("/recommend", json={"history": [1, 2, 3], "k": 10})
        assert r.status_code == 200
        body = r.json()
        assert len(body["items"]) == 10
        assert len(body["scores"]) == 10
        # Scores are returned best-first (non-increasing).
        assert all(a >= b for a, b in zip(body["scores"], body["scores"][1:]))

    def test_per_stage_timings_exist_and_sum_to_total(self, client):
        body = client.post("/recommend", json={"history": [1, 2, 3], "k": 5}).json()
        t = body["timings_ms"]
        for stage in ("user_embedding", "retrieval", "features", "ranking", "filter_topk", "total"):
            assert stage in t
        parts = sum(v for k, v in t.items() if k != "total")
        assert t["total"] == pytest.approx(parts, rel=1e-6)  # structure, not wall-clock

    def test_recommended_items_exclude_history(self, client):
        body = client.post("/recommend", json={"history": [1, 2, 3], "k": 20}).json()
        assert not (set(body["items"]) & {1, 2, 3})  # never recommend a seen item

    def test_empty_and_single_item_histories_are_valid(self, client):
        for hist in ([], [7]):
            r = client.post("/recommend", json={"history": hist, "k": 5})
            assert r.status_code == 200
            assert len(r.json()["items"]) == 5

    def test_unknown_items_in_history_are_dropped_not_fatal(self, client):
        r = client.post("/recommend", json={"history": [1, 999_999], "k": 5})
        assert r.status_code == 200  # the unknown id is ignored, no crash

    def test_response_reports_strategy_and_requested_k(self, client):
        body = client.post("/recommend", json={"history": [1, 2, 3], "k": 7}).json()
        assert body["strategy"] == "two_stage"     # warm path
        assert body["requested_k"] == 7

    def test_duplicate_history_ids_do_not_change_the_result(self, client):
        # P2-3: repeated/retried ids are deduplicated, so the fold-in (and thus the
        # recommendation) is invariant to how many times a client sends the same item.
        once = client.post("/recommend", json={"history": [1, 2, 3], "k": 10}).json()
        many = client.post("/recommend", json={"history": [1, 1, 2, 3, 3, 3, 2], "k": 10}).json()
        assert once["items"] == many["items"]

    def test_empty_history_uses_a_deterministic_popularity_fallback(self, client):
        # P1-4: a cold (empty) history is served from a deterministic popularity list,
        # flagged as such, not the ANN's arbitrary tied set.
        a = client.post("/recommend", json={"history": [], "k": 8}).json()
        b = client.post("/recommend", json={"history": [], "k": 8}).json()
        assert a["strategy"] == "popularity_fallback"
        assert a["items"] == b["items"]            # deterministic across calls
        assert len(a["items"]) == 8


class TestCapacity:
    def test_tops_up_to_k_when_retrieval_is_short(self):
        # P2-4: n_candidates is small, so ranking alone cannot supply k unseen items; the
        # response is topped up from the popularity fallback and still returns exactly k,
        # with no seen item, scores non-increasing, and the shortfall flagged via strategy.
        svc = _synthetic_service(n_items=20, n_candidates=5, seed=1)
        rec = svc.recommend([0, 1, 2], k=8)
        assert len(rec.items) == 8
        assert rec.requested_k == 8
        assert rec.strategy == "two_stage+fallback"
        assert not (set(rec.items) & {0, 1, 2})                      # seen excluded
        assert all(a >= b for a, b in zip(rec.scores, rec.scores[1:]))  # non-increasing


class TestValidation:
    def test_bad_k_returns_422_not_500(self, client):
        assert client.post("/recommend", json={"history": [1], "k": 0}).status_code == 422
        assert client.post("/recommend", json={"history": [1], "k": 9999}).status_code == 422

    def test_missing_history_returns_422(self, client):
        assert client.post("/recommend", json={"k": 5}).status_code == 422

    def test_health_reports_catalogue_size(self, client):
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["n_items"] == 500
