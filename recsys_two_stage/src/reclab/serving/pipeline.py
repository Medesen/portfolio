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

from reclab.ranking.dataset import compute_features, raw_categories
from reclab.serving.ann import ANNIndex


def build_service(split, features, retriever, reranker, n_candidates: int = 500,
                  ann_M: int = 16, ann_ef_construction: int = 200) -> "RecommenderService":
    """Assemble a service from a fitted ALS retriever, item features, and a reranker.

    Builds the HNSW index over the retriever's item embeddings; everything the
    endpoint needs is captured, so the service is self-contained once saved."""
    import numpy as np

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

    @property
    def factors(self) -> int:
        return self.item_factors.shape[1]

    def _fold_in(self, item_indices: np.ndarray) -> np.ndarray:
        """ALS fold-in: a latent user vector from the session's item history."""
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

        # Known items only; unknown/cold ids are simply dropped from the history.
        hist = np.array([self._id_to_index[i] for i in history_item_ids
                         if i in self._id_to_index], dtype=np.int64)

        user_vec = clock("user_embedding", lambda: self._fold_in(hist))
        cand = clock("retrieval", lambda: self.ann.query(
            user_vec[None, :], k=self.n_candidates, ef_search=ef_search)[0])
        # Candidate retrieval scores = the retriever's dot product (ANN's own metric).
        cand_scores = self.item_factors[cand] @ user_vec
        ranks = np.argsort(-cand_scores).argsort().astype(np.float64)

        feats = clock("features", lambda: compute_features(
            cand, cand_scores, ranks, hist, self.item_pop, self.raw_cat, self.available))
        rank_scores = clock("ranking", lambda: self.reranker.predict(feats))

        def finalize():
            seen = set(hist.tolist())
            order = np.argsort(-rank_scores)  # by the ranker's score, best first
            picked, picked_scores = [], []
            for idx in order:
                c = int(cand[idx])
                if c not in seen:
                    picked.append(c)
                    picked_scores.append(float(rank_scores[idx]))
                if len(picked) == k:
                    break
            return picked, picked_scores
        picked, picked_scores = clock("filter_topk", finalize)

        timings["total"] = sum(timings.values())
        return Recommendation(
            # Scores are the *ranker's* scores — the ones that set the order, so they
            # are non-increasing — not the retrieval dot product used only to select.
            items=[int(self.item_ids[c]) for c in picked],
            scores=picked_scores,
            timings_ms={kk: round(v, 3) for kk, v in timings.items()},
            n_candidates=int(self.n_candidates),
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
