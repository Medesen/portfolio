"""RecommenderService — the two-stage pipeline behind the endpoint, and its timing.

One `recommend` call runs the whole system a production request would:

    fold-in user vector → ANN retrieval → feature assembly → rerank → filter-seen → top-k

each step timed separately. The interesting result is not the absolute latency (a
single-threaded laptop measurement in Docker) but the *per-stage share*: feature
assembly and ranking dominate cheap retrieval, which is exactly why production splits
retrieval (cheap, wide) from ranking (expensive, narrow), and why the two-stage
architecture exists at all.

The service takes a history of item ids — not a user id — so it serves anonymous and
unseen visitors, which the history-based ALS fold-in makes possible and which is worth
more to exercise than a user-id lookup.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from reclab.evaluation.metrics import order_by_score
from reclab.ranking.dataset import compute_features, raw_categories
from reclab.serving.ann import ANNIndex


def build_service(split, features, retriever, reranker, n_candidates: int = 500,
                  ann_M: int = 16, ann_ef_construction: int = 200) -> "RecommenderService":
    """Assemble a service from a fitted ALS retriever, item features, and a reranker.

    Builds the HNSW index over the retriever's item embeddings; everything the
    endpoint needs is captured, so the service is self-contained once saved."""
    ann = ANNIndex(retriever.item_factors_, M=ann_M,
                   ef_construction=ann_ef_construction).build()
    return RecommenderService(
        item_ids=np.asarray(split.item_ids),
        item_factors=np.asarray(retriever.item_factors_),
        YtY=np.asarray(retriever._YtY),
        alpha=float(retriever.alpha),
        reg=float(retriever.regularization),
        raw_cat=raw_categories(features),
        available=np.asarray(features.available),
        item_pop=np.log1p(np.asarray(split.train.sum(axis=0)).ravel()),
        ann=ann,
        reranker=reranker.model_,
        n_candidates=n_candidates,
    )


@dataclass
class Recommendation:
    items: list[int]                 # raw item ids, best first
    scores: list[float]
    timings_ms: dict[str, float]     # per-stage, plus "total"
    n_candidates: int
    strategy: str = "two_stage"      # which path served this: two_stage / *+fallback / popularity_fallback
    requested_k: int = 0             # k asked for (so a short response is never silent)


@dataclass
class RecommenderService:
    """Holds fitted artifacts and serves recommendations with per-stage timing."""

    item_ids: np.ndarray             # index -> raw item id
    item_factors: np.ndarray         # (n_items, f) ALS item embeddings
    YtY: np.ndarray                  # (f, f) for the fold-in
    alpha: float
    reg: float
    raw_cat: np.ndarray              # per-item raw categoryid (-1 unknown)
    available: np.ndarray
    item_pop: np.ndarray             # log1p train popularity
    ann: ANNIndex
    reranker: object                 # lightgbm Booster
    n_candidates: int = 500
    _id_to_index: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self._id_to_index:
            self._id_to_index = {int(r): i for i, r in enumerate(self.item_ids)}
        # Deterministic availability-aware popularity order, precomputed once, for the
        # cold-user fallback and the top-up-to-k path. Availability is a large additive
        # bonus (so any available item outranks any unavailable one) rather than a hard
        # sort key, which keeps the fallback scores monotonically non-increasing; ties
        # break by ascending item index via order_by_score.
        avail = np.asarray(self.available, dtype=np.float64)
        bonus = float(self.item_pop.max(initial=0.0)) + 1.0
        self._fallback_score = self.item_pop + bonus * avail
        self._fallback_order = order_by_score(
            self._fallback_score, np.arange(len(self.item_ids))
        )

    @property
    def factors(self) -> int:
        return self.item_factors.shape[1]

    def _popularity_fallback(self, exclude: set[int], n: int) -> list[int]:
        """Top-``n`` items from the precomputed availability-aware popularity order, with
        ``exclude`` (seen/already-picked) removed. Deterministic and independent of the
        user vector — the sensible thing to serve when the fold-in has no signal."""
        picked = []
        for idx in self._fallback_order:
            i = int(idx)
            if i in exclude:
                continue
            picked.append(i)
            if len(picked) == n:
                break
        return picked

    def _fold_in(self, item_indices: np.ndarray) -> np.ndarray:
        """ALS fold-in: a latent user vector from the session's item history.

        The normal case is a non-empty warm history. The empty/all-unknown case is
        short-circuited *before* this by ``recommend`` into a deterministic popularity
        fallback (an empty history folds in to the zero vector, which would otherwise let
        the ANN return an implementation-defined tied set); the zero-vector guard below is
        kept only as a defensive floor.
        """
        if len(item_indices) == 0:
            return np.zeros(self.factors)
        Ys = self.item_factors[item_indices]
        A = self.YtY + self.alpha * (Ys.T @ Ys) + self.reg * np.eye(self.factors)
        b = (1.0 + self.alpha) * Ys.sum(axis=0)
        return np.linalg.solve(A, b)

    def recommend(self, history_item_ids: list[int], k: int = 10,
                  ef_search: int = 64) -> Recommendation:
        if k < 1:
            raise ValueError("k must be >= 1")
        timings: dict[str, float] = {}

        def clock(name, fn):
            t = time.perf_counter()
            out = fn()
            timings[name] = (time.perf_counter() - t) * 1000.0
            return out

        # Known items only, deduplicated while preserving first-touch order. A repeated or
        # retried id must not change the result: the offline interaction matrices are
        # binary/first-touch, so letting a duplicate count twice in the fold-in would be
        # undocumented train/serve skew (P2-3).
        seen_ids = list(dict.fromkeys(
            i for i in history_item_ids if i in self._id_to_index))
        hist = np.array([self._id_to_index[i] for i in seen_ids], dtype=np.int64)

        # Cold user: no known history, so the fold-in carries no signal. Serve a
        # deterministic popularity fallback instead of the ANN's implementation-defined
        # tied set, and say so via `strategy` (P1-4).
        if len(hist) == 0:
            for stage in ("user_embedding", "retrieval", "features", "ranking"):
                timings[stage] = 0.0
            picked = clock("filter_topk", lambda: self._popularity_fallback(set(), k))
            scores = [float(self._fallback_score[c]) for c in picked]
            return self._respond(picked, scores, timings, k, "popularity_fallback")

        user_vec = clock("user_embedding", lambda: self._fold_in(hist))
        # Over-fetch so that, after removing seen items, at least n_candidates *unseen*
        # candidates remain to rank. Offline evaluation masks seen items *before* forming
        # the top-N; querying exactly n_candidates and filtering afterwards would quietly
        # shrink the budget for warm sessions whose own history ranks high (P1-3 parity).
        n_query = min(self.n_candidates + len(hist), len(self.item_ids))
        retrieved = clock("retrieval", lambda: self.ann.query(
            user_vec[None, :], k=n_query, ef_search=ef_search)[0])
        seen = set(hist.tolist())
        cand = np.array([c for c in retrieved if c not in seen][: self.n_candidates],
                        dtype=np.int64)

        # Candidate retrieval scores = the retriever's dot product (ANN's own metric); the
        # rank feature uses the project tie policy so train and serve agree on ties.
        cand_scores = self.item_factors[cand] @ user_vec
        ranks = order_by_score(cand_scores, cand).argsort().astype(np.float64)
        feats = clock("features", lambda: compute_features(
            cand, cand_scores, ranks, hist, self.item_pop, self.raw_cat, self.available))
        rank_scores = clock("ranking", lambda: self.reranker.predict(feats))

        def finalize():
            # Seen items were already removed before ranking (parity above), so this just
            # takes the top-k by the ranker's score, tie-broken by item id.
            order = order_by_score(rank_scores, cand)
            picked = [int(cand[i]) for i in order[:k]]
            scores = [float(rank_scores[i]) for i in order[:k]]
            # If retrieval could not supply k unseen candidates (tiny catalogue or a very
            # long history), top up deterministically from the popularity fallback so the
            # "return k items" contract holds, placing extras just below the ranked scores
            # (P2-4). requested_k in the response makes any shortfall explicit.
            if len(picked) < k:
                extra = self._popularity_fallback(seen | set(picked), k - len(picked))
                floor = scores[-1] if scores else 0.0
                scores += [floor - 1.0 - j for j in range(len(extra))]
                picked += extra
                return picked, scores, True
            return picked, scores, False
        picked, scores, topped = clock("filter_topk", finalize)
        return self._respond(picked, scores, timings, k,
                             "two_stage+fallback" if topped else "two_stage")

    def _respond(self, picked, scores, timings, k, strategy) -> Recommendation:
        # Round per-stage first, then derive the total from the rounded parts, so the
        # reported numbers always add up exactly (round-then-sum, not sum-then-round — the
        # latter lets the displayed total disagree with its parts in the last digit).
        timings_ms = {kk: round(v, 3) for kk, v in timings.items()}
        timings_ms["total"] = round(sum(timings_ms.values()), 3)
        return Recommendation(
            # Scores are the *ranker's* scores — the ones that set the order, so they are
            # non-increasing — not the retrieval dot product used only to select.
            items=[int(self.item_ids[c]) for c in picked],
            scores=scores,
            timings_ms=timings_ms,
            n_candidates=int(self.n_candidates),
            strategy=strategy,
            requested_k=int(k),
        )

    # -- persistence -------------------------------------------------------- #
    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez(directory / "artifacts.npz",
                 item_ids=self.item_ids, item_factors=self.item_factors, YtY=self.YtY,
                 raw_cat=self.raw_cat, available=self.available, item_pop=self.item_pop)
        (directory / "config.json").write_text(json.dumps({
            "alpha": self.alpha, "reg": self.reg, "n_candidates": self.n_candidates,
            "ann_M": self.ann.M, "ann_ef_construction": self.ann.ef_construction,
            "dim": int(self.item_factors.shape[1]),
        }))
        self.ann.index.save_index(str(directory / "ann.bin"))
        self.reranker.save_model(str(directory / "reranker.txt"))

    @classmethod
    def load(cls, directory: Path) -> "RecommenderService":
        import hnswlib
        import lightgbm as lgb

        directory = Path(directory)
        a = np.load(directory / "artifacts.npz")
        cfg = json.loads((directory / "config.json").read_text())

        ann = ANNIndex(a["item_factors"], M=cfg["ann_M"],
                       ef_construction=cfg["ann_ef_construction"])
        index = hnswlib.Index(space="ip", dim=cfg["dim"])
        index.load_index(str(directory / "ann.bin"), max_elements=len(a["item_ids"]))
        ann.index = index

        reranker = lgb.Booster(model_file=str(directory / "reranker.txt"))
        return cls(
            item_ids=a["item_ids"], item_factors=a["item_factors"], YtY=a["YtY"],
            alpha=cfg["alpha"], reg=cfg["reg"], raw_cat=a["raw_cat"],
            available=a["available"], item_pop=a["item_pop"], ann=ann,
            reranker=reranker, n_candidates=cfg["n_candidates"],
        )
