"""Ranker training data: the nested-window integrity and the feature-leakage test.

The leakage test is the most important in Stage 3 — a ranker feature that peeks at
the future produces a spectacular, meaningless result — so it is written before any
ranker exists.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from reclab.models import ItemKNN
from reclab.ranking.dataset import build_ranker_frame, nested_split
from reclab.splitting import temporal_split


def _synthetic_log(seed: int = 0, days: int = 45) -> pd.DataFrame:
    """A dense log over `days` so the three nested windows each hold real data."""
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2015-06-01", tz="UTC")
    rows, sid = [], 0
    for _ in range(2500):
        day = rng.integers(0, days)
        start = base + pd.Timedelta(days=int(day), minutes=int(rng.integers(0, 600)))
        for it in rng.choice(60, size=rng.integers(2, 6), replace=False):
            rows.append({"session": sid, "itemid": int(it),
                         "ts": start + pd.Timedelta(minutes=int(it))})
        sid += 1
    return pd.DataFrame(rows)


class TestNestedWindowIntegrity:
    def test_A_precedes_B_precedes_C(self):
        log = _synthetic_log(seed=1)
        ranker_train, eval_split = nested_split(log, test_days=10)
        # ranker_train.cutoff = T1 (A|B boundary); eval_split.cutoff = T2 (B|C boundary).
        assert ranker_train.cutoff < eval_split.cutoff
        # Period A (ranker_train's *train*) ends strictly before T1.
        # Period C (eval_split's *test*) begins at/after T2. The windows do not overlap
        # in the way that would leak: A is fit before B, B before C.

    def test_ranker_train_and_eval_are_distinct_windows(self):
        log = _synthetic_log(seed=2)
        ranker_train, eval_split = nested_split(log, test_days=10)
        # The eval training window extends past the ranker-training one (it includes B).
        assert eval_split.n_train_sessions > ranker_train.n_train_sessions


class TestFeatureLeakage:
    def test_features_are_invariant_to_the_labels(self):
        # The label is the *only* future-derived input `build_ranker_frame` reads —
        # everything else comes from the retrieval-fit window and the session prefix
        # (both strictly before the label). So the leakage guarantee is exactly this:
        # permute the held-out targets (change what "the future" is) and the feature
        # matrix and the retrieved candidates must be bit-identical; only the labels
        # move. A feature that had secretly encoded the target would fail here.
        log = _synthetic_log(seed=3)
        split = temporal_split(log, test_days=10, min_item_sessions=5)
        from reclab.features import build_item_features
        feats = build_item_features(split.cutoff, split.item_ids)
        retriever = ItemKNN(k=50, shrink=10.0).fit(split.train)

        frame_a = build_ranker_frame(retriever, split, feats, n_candidates=50)

        rng = np.random.default_rng(0)
        swapped = dataclasses.replace(split, test_target=rng.permutation(split.test_target))
        frame_b = build_ranker_frame(retriever, swapped, feats, n_candidates=50)

        np.testing.assert_array_equal(frame_a.candidate_items, frame_b.candidate_items)
        np.testing.assert_allclose(frame_a.X, frame_b.X)
        # Labels did move (otherwise the test proves nothing).
        assert not np.array_equal(frame_a.y, frame_b.y)

    def test_retriever_ignores_the_future_by_construction(self):
        # The companion guarantee: the retriever generating candidates is fit only on
        # the training window, so corrupting post-cutoff interactions cannot change it.
        log = _synthetic_log(seed=7)
        split = temporal_split(log, test_days=10, min_item_sessions=5)
        corrupted = log.copy()
        corrupted.loc[corrupted["ts"] >= split.cutoff, "itemid"] += 100_000
        split2 = temporal_split(corrupted, test_days=10, min_item_sessions=5)
        # Same training window -> same vocabulary -> same fitted retriever.
        assert np.array_equal(split.item_ids, split2.item_ids)
        a = ItemKNN(k=50, shrink=10.0).fit(split.train).similarity_
        b = ItemKNN(k=50, shrink=10.0).fit(split2.train).similarity_
        assert (a != b).nnz == 0


class TestLabels:
    def test_labels_mark_the_held_out_target(self):
        log = _synthetic_log(seed=4)
        split = temporal_split(log, test_days=10, min_item_sessions=5)
        from reclab.features import build_item_features
        feats = build_item_features(split.cutoff, split.item_ids)
        retriever = ItemKNN(k=50, shrink=10.0).fit(split.train)
        frame = build_ranker_frame(retriever, split, feats, n_candidates=50)

        # A positive row's candidate item is exactly that session's target.
        pos = np.flatnonzero(frame.y == 1)
        for r in pos[:50]:
            assert frame.candidate_items[r] == split.test_target[frame.session_row[r]]
        # Positive rate is recorded and sane (retrieval doesn't always find the target).
        assert 0.0 < frame.positive_rate < 1.0

    def test_group_sizes_sum_to_rows_and_stay_within_a_session(self):
        log = _synthetic_log(seed=5)
        split = temporal_split(log, test_days=10, min_item_sessions=5)
        from reclab.features import build_item_features
        feats = build_item_features(split.cutoff, split.item_ids)
        retriever = ItemKNN(k=50, shrink=10.0).fit(split.train)
        frame = build_ranker_frame(retriever, split, feats, n_candidates=50)
        assert frame.groups.sum() == len(frame.y)
        # Each group is one session: the session_row is constant within a group.
        offset = 0
        for g in frame.groups[:20]:
            block = frame.session_row[offset:offset + g]
            assert len(np.unique(block)) == 1
            offset += g

    def test_negative_downsampling_keeps_positives(self):
        log = _synthetic_log(seed=6)
        split = temporal_split(log, test_days=10, min_item_sessions=5)
        from reclab.features import build_item_features
        feats = build_item_features(split.cutoff, split.item_ids)
        retriever = ItemKNN(k=50, shrink=10.0).fit(split.train)
        full = build_ranker_frame(retriever, split, feats, n_candidates=50)
        down = build_ranker_frame(retriever, split, feats, n_candidates=50,
                                  negatives_per_session=5, seed=0)
        assert down.y.sum() == full.y.sum()   # every positive retained
        assert len(down.y) < len(full.y)       # negatives dropped
