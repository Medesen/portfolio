"""Ranker training data — the nested temporal design, and the leakage it prevents.

A reranker is trained on *candidates a retriever produced*, labelled by what the user
did next. The subtle failure: if the retriever that generated the training candidates
was fit on the very interactions used as labels, its scores are in-sample during
training and out-of-sample at serving time, so the ranker learns to trust a signal
that will be weaker in production. Nothing crashes; the model is quietly miscalibrated.

The fix is a nested temporal split:

    |<-------- A: retrieval fit -------->|<-- B: ranker labels -->|<-- C: test -->|
    start                               T1                       T2             end

- **A** `[start, T1)` — fit the retriever that generates *training* candidates.
- **B** `[T1, T2)` — for each B session, retrieve from the A-retriever, label a
  candidate 1 iff it is the session's held-out next item. Train the ranker on these.
- **C** `[T2, end)` — the Stage 1/2 test window, untouched. For evaluation the
  retriever is refit on `[start, T2)` and the trained ranker applied on top — mirroring
  production, where retrieval retrains often and the ranker rarely.

Both windows are produced by the *same* ``temporal_split`` used everywhere else, so A
precedes B precedes C by construction. Every ranker feature is computed from data in
the retrieval-fit window (strictly before the labels), which is what makes the
feature-leakage test pass: corrupt everything at or after the label time and no
feature moves.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from reclab.evaluation.full_catalogue import history_chunk
from reclab.features.item_features import ItemFeatures
from reclab.splitting.protocols import SessionSplit, temporal_split

# Feature columns, in a fixed order. Deliberately thin: RetailRocket has no prices,
# no text, and hashed properties, so the ranker has little to work with beyond the
# retrieval signal itself — a property of the data, stated not discovered (see README).
FEATURES = [
    "retr_score",      # the retriever's score for this candidate
    "retr_rank",       # its rank in the candidate list (0 = top)
    "item_pop",        # log1p train-window interaction count
    "item_available",  # availability flag as-of the retrieval-fit cutoff
    "sess_len",        # number of items in the session's history
    "sess_n_cats",     # distinct categories in that history
    "cat_match",       # 1 if the candidate's category appears in the session history
    "cat_share",       # share of the session's history sharing the candidate's category
]


@dataclass
class RankerFrame:
    """A ranker dataset: features X, binary labels y, and per-session group sizes."""

    X: np.ndarray            # (n_rows, n_features)
    y: np.ndarray            # (n_rows,) 0/1
    groups: np.ndarray       # group sizes; sum == n_rows, one group per session
    candidate_items: np.ndarray  # (n_rows,) item index of each candidate (for e2e eval)
    session_row: np.ndarray  # (n_rows,) which evaluation session each row belongs to
    positive_rate: float

    @property
    def n_sessions(self) -> int:
        return len(self.groups)


def nested_split(session_items: pd.DataFrame, test_days: int = 28):
    """Return (ranker_train_split, eval_split).

    ``ranker_train_split`` has train = period A and test = period B (its held-out
    "next item" is the ranker label). ``eval_split`` is the Stage 1/2 split: train =
    ``[start, T2)``, test = period C.
    """
    cutoff_t2 = session_items["ts"].max().normalize() - pd.Timedelta(days=test_days)
    before_t2 = session_items[session_items["ts"] < cutoff_t2]
    # Period A/B split: reuse temporal_split on the pre-T2 data, so B is a test window
    # of the same length carved just before T2.
    ranker_train = temporal_split(before_t2, test_days=test_days)
    eval_split = temporal_split(session_items, test_days=test_days)
    return ranker_train, eval_split


def retrieve_with_scores(retriever, split: SessionSplit, n_candidates: int,
                         chunk_size: int = 1024):
    """Top-``n_candidates`` items and their scores per test session, seen items masked."""
    from reclab.evaluation.metrics import top_k_from_scores

    n = split.n_test_sessions
    items = np.empty((n, n_candidates), dtype=np.int64)
    scores = np.empty((n, n_candidates), dtype=np.float64)
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        history = history_chunk(split, start, stop)
        s = retriever.score(history)
        seen = history.matrix.toarray() > 0
        ranked = top_k_from_scores(s, n_candidates, mask=seen)
        items[start:stop] = ranked
        rows = np.arange(stop - start)[:, None]
        scores[start:stop] = s[rows, ranked]
    return items, scores


def raw_categories(features: ItemFeatures) -> np.ndarray:
    """Raw categoryid per item (vocab-agnostic), so category signals mean the same
    thing across the two windows' different vocabularies."""
    inv = {idx: raw for raw, idx in features.category_vocab.items()}
    return np.array([inv.get(int(c), -1) for c in features.category_ids], dtype=np.int64)


def compute_features(
    cand: np.ndarray, cand_scores: np.ndarray, cand_ranks: np.ndarray,
    history_items: np.ndarray, item_pop: np.ndarray, raw_cat: np.ndarray,
    available: np.ndarray,
) -> np.ndarray:
    """The feature matrix for one session's candidates.

    Extracted so training (``build_ranker_frame``) and serving (``RecommenderService``)
    compute features the *same* way — train/serve skew from two copies of this logic is
    a classic and silent ranker bug.
    """
    hist_cats = raw_cat[history_items]
    hist_cats_valid = hist_cats[hist_cats >= 0]
    sess_len = len(history_items)
    sess_n_cats = len(np.unique(hist_cats_valid))
    cand_cats = raw_cat[cand]
    cat_match = np.isin(cand_cats, hist_cats_valid).astype(np.float64)
    if len(hist_cats_valid):
        cat_share = np.array([np.mean(hist_cats_valid == c) if c >= 0 else 0.0
                              for c in cand_cats])
    else:
        cat_share = np.zeros(len(cand))
    return np.column_stack([
        cand_scores,
        cand_ranks.astype(np.float64),
        item_pop[cand],
        available[cand].astype(np.float64),
        np.full(len(cand), sess_len, dtype=np.float64),
        np.full(len(cand), sess_n_cats, dtype=np.float64),
        cat_match,
        cat_share,
    ])


def build_ranker_frame(
    retriever, split: SessionSplit, features: ItemFeatures, n_candidates: int,
    negatives_per_session: int | None = None, seed: int = 0,
) -> RankerFrame:
    """Retrieve candidates for every test session and assemble the ranker dataset.

    ``negatives_per_session`` downsamples negatives for *training* (ranking is
    invariant to it; calibration is not, which we do not need). Leave ``None`` for
    evaluation, where the full candidate set must be ranked.
    """
    items, scores = retrieve_with_scores(retriever, split, n_candidates)
    targets = split.test_target
    n_sessions = split.n_test_sessions

    item_pop = np.log1p(np.asarray(split.train.sum(axis=0)).ravel())
    raw_cat = raw_categories(features)
    available = features.available
    rng = np.random.default_rng(seed)

    rows_X, rows_y, rows_item, rows_sess, group_sizes = [], [], [], [], []
    for s in range(n_sessions):
        cand = items[s]
        label = (cand == targets[s]).astype(np.int64)

        if negatives_per_session is not None:
            pos_idx = np.flatnonzero(label)
            neg_idx = np.flatnonzero(label == 0)
            if len(neg_idx) > negatives_per_session:
                neg_idx = rng.choice(neg_idx, negatives_per_session, replace=False)
            keep = np.sort(np.concatenate([pos_idx, neg_idx]))
            cand, label = cand[keep], label[keep]
            cand_scores, cand_ranks = scores[s][keep], keep.astype(np.float64)
        else:
            cand_scores, cand_ranks = scores[s], np.arange(len(cand), dtype=np.float64)

        # Session-history signals (from the retrieval-fit window only).
        hist = split.test_prefix[s].indices
        feat = compute_features(cand, cand_scores, cand_ranks, hist,
                                item_pop, raw_cat, available)
        rows_X.append(feat)
        rows_y.append(label)
        rows_item.append(cand)
        rows_sess.append(np.full(len(cand), s, dtype=np.int64))
        group_sizes.append(len(cand))

    X = np.vstack(rows_X)
    y = np.concatenate(rows_y)
    return RankerFrame(
        X=X, y=y, groups=np.asarray(group_sizes),
        candidate_items=np.concatenate(rows_item),
        session_row=np.concatenate(rows_sess),
        positive_rate=float(y.mean()),
    )
