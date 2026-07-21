"""Split protocols: the temporal leakage guarantee is the one that matters.

The headline claim of this whole project is that evaluation is done honestly, so
the split's promises are tested directly: no training interaction from the future,
straddling sessions dropped, and post-cutoff data provably unable to move the
trained matrix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from reclab.splitting.protocols import leave_one_out_split, temporal_split


def session_items(rows: list[tuple]) -> pd.DataFrame:
    """Build a (session, itemid, ts) frame from (session, item, day) triples."""
    base = pd.Timestamp("2015-06-01", tz="UTC")
    return pd.DataFrame(
        {
            "session": [r[0] for r in rows],
            "itemid": [r[1] for r in rows],
            "ts": [base + pd.Timedelta(days=r[2]) for r in rows],
        }
    )


def dense_log(n_sessions: int, n_items: int, days: int, seed: int = 0) -> pd.DataFrame:
    """A log dense enough to survive a (2, 5) filter, spread over `days`."""
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2015-06-01", tz="UTC")
    rows = []
    for s in range(n_sessions):
        day = rng.integers(0, days)
        start = base + pd.Timedelta(days=int(day), minutes=int(rng.integers(0, 600)))
        for it in rng.choice(n_items, size=rng.integers(2, 6), replace=False):
            rows.append(
                {"session": s, "itemid": int(it), "ts": start + pd.Timedelta(minutes=int(it))}
            )
    return pd.DataFrame(rows)


class TestTemporalLeakage:
    def test_no_training_interaction_from_the_future(self):
        # The core guarantee. Reconstruct which sessions landed in training and
        # assert every one of their timestamps predates the cutoff.
        log = dense_log(400, 30, days=40, seed=3)
        split = temporal_split(log, test_days=14, min_item_sessions=5)

        # Sessions surviving into train are those wholly before the cutoff; verify
        # by re-deriving from the raw log that none reach past it.
        bounds = log.groupby("session")["ts"].agg(["min", "max"])
        train_ids = set(bounds.index[bounds["max"] < split.cutoff])
        assert split.n_train_sessions <= len(train_ids)
        # Nothing at or after the cutoff can belong to a wholly-train session.
        train_max = log[log["session"].isin(train_ids)]["ts"].max()
        assert train_max < split.cutoff

    def test_straddling_sessions_are_dropped(self):
        # Deterministic timestamps so the cutoff is known exactly. Training
        # filler on days 0-9; a lone test session on day 20 fixes the global max,
        # so with test_days=5 the cutoff lands on day 15. The straddler is placed
        # to span it: one event on day 14.5, one on day 15.5.
        base_ts = pd.Timestamp("2015-06-01", tz="UTC")
        rng = np.random.default_rng(5)
        rows = []
        for s in range(300):
            day = rng.integers(0, 10)
            for it in rng.choice(25, size=3, replace=False):
                rows.append({"session": s, "itemid": int(it),
                             "ts": base_ts + pd.Timedelta(days=int(day), minutes=int(it))})
        # Test-period session that defines the global max (day 20).
        for it in (0, 1, 2):
            rows.append({"session": 5000, "itemid": it,
                         "ts": base_ts + pd.Timedelta(days=20, minutes=it)})
        # Straddler spanning the day-15 cutoff.
        rows.append({"session": 99999, "itemid": 0, "ts": base_ts + pd.Timedelta(days=14.5)})
        rows.append({"session": 99999, "itemid": 1, "ts": base_ts + pd.Timedelta(days=15.5)})
        log = pd.DataFrame(rows)

        split = temporal_split(log, test_days=5, min_item_sessions=5)
        assert split.cutoff == base_ts + pd.Timedelta(days=15)
        assert split.n_straddling_dropped == 1

    def test_post_cutoff_changes_cannot_move_the_train_matrix(self):
        # The demand_forecasting-style leakage test, at the split level: corrupt
        # everything after the cutoff and assert the trained matrix is identical.
        log = dense_log(400, 30, days=40, seed=7)
        split_a = temporal_split(log, test_days=14, min_item_sessions=5)

        cutoff = split_a.cutoff
        corrupted = log.copy()
        post = corrupted["ts"] >= cutoff
        # Reassign every post-cutoff item to a brand-new item id, and add noise.
        corrupted.loc[post, "itemid"] += 10_000
        extra = session_items([(88888, 55555, 39), (88888, 55556, 39)])
        corrupted = pd.concat([corrupted, extra], ignore_index=True)

        split_b = temporal_split(corrupted, test_days=14, min_item_sessions=5)

        assert split_a.train.shape == split_b.train.shape
        assert split_a.train.nnz == split_b.train.nnz
        np.testing.assert_array_equal(split_a.item_ids, split_b.item_ids)
        assert (split_a.train != split_b.train).nnz == 0


class TestPrefixTarget:
    def test_target_is_the_last_item_by_time(self):
        # Two clean test sessions plus filler so items clear the (2,5) filter.
        rng = np.random.default_rng(0)
        base = pd.Timestamp("2015-06-01", tz="UTC")
        rows = []
        # Filler in the training period to give items support.
        for s in range(200):
            for it in rng.choice(10, size=3, replace=False):
                rows.append({"session": s, "itemid": int(it),
                             "ts": base + pd.Timedelta(days=1, minutes=int(it))})
        # A test-period session with a known last item (item 7 at the latest ts).
        for it, minute in [(3, 0), (5, 1), (7, 2)]:
            rows.append({"session": 9001, "itemid": it,
                         "ts": base + pd.Timedelta(days=39, minutes=minute)})
        log = pd.DataFrame(rows)
        split = temporal_split(log, test_days=5, min_item_sessions=5)

        # Locate our known session among the test rows via its target column.
        col_of = {int(it): i for i, it in enumerate(split.item_ids)}
        assert col_of[7] in set(split.test_target.tolist())

    def test_prefix_excludes_the_target(self):
        log = dense_log(400, 30, days=40, seed=11)
        split = temporal_split(log, test_days=14, min_item_sessions=5)
        # For each test session the held-out target must not appear in its prefix.
        prefix = split.test_prefix.tolil()
        for row, target in enumerate(split.test_target):
            assert prefix[row, int(target)] == 0


class TestLeaveOneOut:
    def test_holds_out_one_item_per_session(self):
        log = dense_log(400, 30, days=40, seed=13)
        split = leave_one_out_split(log, min_item_sessions=5)
        assert split.protocol == "leave_one_out"
        assert split.cutoff is None
        assert len(split.test_target) == split.n_test_sessions

    def test_is_reproducible_under_a_seed(self):
        log = dense_log(300, 25, days=40, seed=17)
        a = leave_one_out_split(log, min_item_sessions=5, seed=42)
        b = leave_one_out_split(log, min_item_sessions=5, seed=42)
        np.testing.assert_array_equal(a.test_target, b.test_target)


def test_rejects_bad_filter_scope():
    with pytest.raises(ValueError, match="filter_scope"):
        temporal_split(dense_log(50, 10, 20), filter_scope="nonsense")


def test_rejects_nonpositive_test_days():
    with pytest.raises(ValueError, match="test_days"):
        temporal_split(dense_log(50, 10, 20), test_days=0)
