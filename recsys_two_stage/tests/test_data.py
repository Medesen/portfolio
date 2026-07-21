"""Loading, sessionization and k-core filtering on synthetic fixtures.

Synthetic rather than the bundled file: these tests assert behaviour on inputs
whose answers are known by construction, and they must run without the 32 MB
data file present.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from reclab.data.filtering import k_core_filter
from reclab.data.load import sessionize, to_session_items


def events(rows: list[tuple]) -> pd.DataFrame:
    """Build a loaded-events frame from (visitor, item, minute) triples."""
    base = pd.Timestamp("2015-06-01", tz="UTC")
    return pd.DataFrame(
        {
            "visitorid": [r[0] for r in rows],
            "itemid": [r[1] for r in rows],
            "event": ["view"] * len(rows),
            "ts": [base + pd.Timedelta(minutes=r[2]) for r in rows],
        }
    )


class TestSessionize:
    def test_gap_splits_sessions(self):
        # Same visitor, two clusters 60 minutes apart -> two sessions.
        df = events([(1, 10, 0), (1, 11, 5), (1, 12, 65), (1, 13, 70)])
        out = sessionize(df, gap_minutes=30)
        assert out["session"].nunique() == 2
        first = out[out["itemid"].isin([10, 11])]["session"].unique()
        second = out[out["itemid"].isin([12, 13])]["session"].unique()
        assert len(first) == 1 and len(second) == 1 and first[0] != second[0]

    def test_different_visitors_never_share_a_session(self):
        # Identical timestamps, different visitors: sessions must not merge.
        df = events([(1, 10, 0), (2, 20, 0), (1, 11, 1), (2, 21, 1)])
        out = sessionize(df, gap_minutes=30)
        per_visitor = out.groupby("session")["visitorid"].nunique()
        assert (per_visitor == 1).all()

    def test_within_gap_stays_one_session(self):
        df = events([(1, 10, 0), (1, 11, 10), (1, 12, 20)])
        out = sessionize(df, gap_minutes=30)
        assert out["session"].nunique() == 1

    def test_rejects_nonpositive_gap(self):
        with pytest.raises(ValueError, match="gap_minutes"):
            sessionize(events([(1, 10, 0)]), gap_minutes=0)


class TestToSessionItems:
    def test_collapses_repeats_and_counts_them(self):
        # Item 10 viewed 3 times in one session -> one row, n_events == 3.
        df = sessionize(events([(1, 10, 0), (1, 10, 1), (1, 10, 2), (1, 11, 3)]))
        out = to_session_items(df)
        assert len(out) == 2
        row10 = out[out["itemid"] == 10].iloc[0]
        assert row10["n_events"] == 3

    def test_keeps_first_touch_order(self):
        df = sessionize(events([(1, 11, 0), (1, 10, 1), (1, 12, 2)]))
        out = to_session_items(df)
        assert list(out["itemid"]) == [11, 10, 12]


def pairs(rows: list[tuple]) -> pd.DataFrame:
    """(session, item) pairs frame for filter tests."""
    return pd.DataFrame({"session": [r[0] for r in rows], "itemid": [r[1] for r in rows]})


class TestKCoreFilter:
    def test_reaches_a_genuine_fixed_point(self):
        # A 2-item session and a rare item that a single pass would miss the
        # knock-on effects of.
        rng = np.random.default_rng(0)
        rows = []
        for s in range(200):
            for it in rng.choice(30, size=rng.integers(2, 6), replace=False):
                rows.append((s, int(it)))
        # A few rare items and thin sessions to actually exercise the iteration.
        rows += [(900, 500), (901, 500), (902, 501)]
        df = pairs(rows).drop_duplicates()

        filtered, report = k_core_filter(df, min_session_items=2, min_item_sessions=5)
        assert report.converged

        # A second application changes nothing — the definition of a fixed point.
        again, _ = k_core_filter(filtered, 2, 5)
        assert len(again) == len(filtered)
        assert report.iterations >= 1

    def test_thresholds_actually_hold_after_filtering(self):
        rng = np.random.default_rng(1)
        rows = []
        for s in range(300):
            for it in rng.choice(40, size=rng.integers(1, 7), replace=False):
                rows.append((s, int(it)))
        filtered, _ = k_core_filter(pairs(rows).drop_duplicates(), 2, 5)
        assert (filtered["session"].value_counts() >= 2).all()
        assert (filtered["itemid"].value_counts() >= 5).all()

    def test_report_traces_every_iteration(self):
        filtered, report = k_core_filter(
            pairs([(0, 1), (0, 2), (1, 1), (1, 2), (2, 1)]).drop_duplicates(), 2, 2
        )
        frame = report.to_frame()
        assert "sessions" in frame.columns and "items" in frame.columns
        assert len(frame) == report.iterations + 1  # includes the initial state

    def test_rejects_bad_thresholds(self):
        with pytest.raises(ValueError, match="thresholds"):
            k_core_filter(pairs([(0, 1)]), 0, 5)

    def test_missing_column_raises(self):
        with pytest.raises(ValueError, match="missing required column"):
            k_core_filter(pd.DataFrame({"session": [0]}), 1, 1)
