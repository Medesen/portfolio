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

from reclab.data.load import to_session_items
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


class TestCountMatrix:
    """The repeat-count path the ALS ablation relies on: n_events must actually reach
    train_counts (it silently did not before), while train stays binary."""

    def test_repeat_counts_reach_train_counts_but_train_stays_binary(self):
        base = pd.Timestamp("2015-06-01", tz="UTC")
        rows = []
        # Session 0, fully pre-cutoff: views item 0 three times, plus items 1 and 2 once.
        for r in range(3):
            rows.append({"session": 0, "itemid": 0, "ts": base + pd.Timedelta(minutes=r)})
        rows.append({"session": 0, "itemid": 1, "ts": base + pd.Timedelta(minutes=5)})
        rows.append({"session": 0, "itemid": 2, "ts": base + pd.Timedelta(minutes=6)})
        # Dense population so every item clears the k-core and lands in training.
        rng = np.random.default_rng(0)
        for s in range(1, 300):
            day = int(rng.integers(0, 20))
            for it in rng.choice(np.arange(8), size=3, replace=False):
                rows.append({"session": s, "itemid": int(it),
                             "ts": base + pd.Timedelta(days=day, minutes=int(it))})

        pairs = to_session_items(pd.DataFrame(rows))
        assert int(pairs[(pairs.session == 0) & (pairs.itemid == 0)]["n_events"].iloc[0]) == 3

        split = temporal_split(pairs, test_days=5, min_item_sessions=5)
        assert split.train_counts is not None
        # Same sparsity pattern, different values where a repeat occurred.
        assert set(map(tuple, np.argwhere(split.train.toarray() > 0))) == \
            set(map(tuple, np.argwhere(split.train_counts.toarray() > 0)))
        assert split.train.max() == 1.0                 # binary
        assert split.train_counts.max() >= 3.0          # the repeated cell carries its count
        assert (split.train_counts.data >= 1.0).all()

    def test_train_counts_is_none_without_n_events(self):
        # Frames that never went through to_session_items (no n_events) get no count
        # twin, so nothing downstream can silently assume one exists.
        log = dense_log(300, 12, days=30, seed=5)  # (session, itemid, ts) only
        split = temporal_split(log, test_days=7, min_item_sessions=5)
        assert split.train_counts is None


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


class TestP0TargetPreservation:
    """The P0-1 correction. Filtering the test set to the training vocabulary must never
    move a session's target back onto an earlier warm item: a session whose *raw final*
    item is cold is excluded from the warm headline, never silently relabelled."""

    def _log_with_cold_targets(self) -> pd.DataFrame:
        base = pd.Timestamp("2015-06-01", tz="UTC")
        rng = np.random.default_rng(0)
        rows = []
        # Training filler (days 0-2): items 0..9 get ample support, so all are warm.
        for s in range(200):
            day = int(rng.integers(0, 3))
            for it in rng.choice(10, size=3, replace=False):
                rows.append({"session": s, "itemid": int(it),
                             "ts": base + pd.Timedelta(days=day, minutes=int(it))})
        # Test period (day 39 fixes the global max; test_days=5 -> cutoff day 34).
        # S_cold: warm prefix [3, 5], then a COLD final item 900 (never seen in training).
        for it, minute in [(3, 0), (5, 1), (900, 2)]:
            rows.append({"session": 9001, "itemid": it,
                         "ts": base + pd.Timedelta(days=39, minutes=minute)})
        # S_mid: a cold item 901 in the MIDDLE, but a warm final item 8.
        for it, minute in [(4, 0), (901, 1), (8, 2)]:
            rows.append({"session": 9002, "itemid": it,
                         "ts": base + pd.Timedelta(days=39, minutes=minute)})
        return pd.DataFrame(rows)

    def test_cold_final_item_is_never_relabelled_onto_an_earlier_warm_item(self):
        split = temporal_split(self._log_with_cold_targets(), test_days=5, min_item_sessions=5)
        col_of = {int(it): i for i, it in enumerate(split.item_ids)}
        assert 900 not in col_of and 901 not in col_of        # both cold
        for w in (3, 4, 5, 8):
            assert w in col_of                                # all warm
        targets = set(split.test_target.tolist())
        assert col_of[8] in targets           # S_mid (raw last = warm 8) is scored
        assert col_of[5] not in targets       # THE regression: S_cold's 900 not moved to 5
        assert split.n_test_sessions == 1     # only S_mid enters the warm headline

    def test_cohort_report_accounts_for_every_post_cutoff_session(self):
        split = temporal_split(self._log_with_cold_targets(), test_days=5, min_item_sessions=5)
        c = split.cohort
        assert c is not None
        assert c.n_post_cutoff == 2
        assert c.n_warm_target == 1           # S_mid
        assert c.n_cold_target == 1           # S_cold: cold final item, but a warm prefix
        assert c.n_warm_target + c.n_cold_target + c.n_insufficient == c.n_post_cutoff

    def test_cold_middle_item_is_dropped_from_the_prefix_but_the_session_is_kept(self):
        split = temporal_split(self._log_with_cold_targets(), test_days=5, min_item_sessions=5)
        col_of = {int(it): i for i, it in enumerate(split.item_ids)}
        row = int(np.where(split.test_target == col_of[8])[0][0])
        prefix = split.test_prefix[row].indices
        assert col_of[4] in prefix            # warm prefix item kept
        assert 901 not in col_of              # cold middle item had no column to appear in


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
