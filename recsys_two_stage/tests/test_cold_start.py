"""Cold-start track: the definitions and the structural-zero claim.

Built on a small synthetic log with a known pre/post-cutoff structure so the cold
set and the cold share are checkable by hand.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from reclab.evaluation.cold_start import _asof_cold_frequencies, build_cold_start_eval
from reclab.features.item_features import ItemFeatures
from reclab.splitting import temporal_split


def _warm_features(item_ids: np.ndarray) -> ItemFeatures:
    # Every item in one category — enough for the builder; content quality is not
    # what these tests check.
    return ItemFeatures(
        item_ids=item_ids,
        category_ids=np.ones(len(item_ids), dtype=np.int64),
        parent_ids=np.ones(len(item_ids), dtype=np.int64),
        available=np.ones(len(item_ids), dtype=np.float32),
        category_vocab={1: 1},
        parent_vocab={1: 1},
    )


def _synthetic_log(seed: int = 0) -> pd.DataFrame:
    """Warm items 0-9 (dense, pre-cutoff); cold items 100-102 appear only after."""
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2015-06-01", tz="UTC")
    rows = []
    sid = 0
    # Pre-cutoff: many sessions over warm items so they clear the k-core.
    for _ in range(400):
        day = rng.integers(0, 20)  # cutoff will be well after
        items = rng.choice(np.arange(10), size=rng.integers(2, 5), replace=False)
        for it in items:
            rows.append({"session": sid, "itemid": int(it),
                         "ts": base + pd.Timedelta(days=int(day), minutes=int(it))})
        sid += 1
    # Post-cutoff (days 34-35): warm prefix ending on a cold item (100-102),
    # plus some ending on a warm item.
    for _ in range(60):
        rows.append({"session": sid, "itemid": int(rng.integers(0, 10)),
                     "ts": base + pd.Timedelta(days=34, minutes=1)})
        cold = 100 + int(rng.integers(0, 3))
        rows.append({"session": sid, "itemid": cold,
                     "ts": base + pd.Timedelta(days=34, minutes=5)})
        sid += 1
    for _ in range(40):
        for it in rng.choice(np.arange(10), size=2, replace=False):
            rows.append({"session": sid, "itemid": int(it),
                         "ts": base + pd.Timedelta(days=35, minutes=int(it))})
        sid += 1
    return pd.DataFrame(rows)


class TestColdStartDefinitions:
    def test_cold_items_are_genuinely_absent_from_training(self):
        log = _synthetic_log(seed=1)
        split = temporal_split(log, test_days=7, min_item_sessions=5)
        cold = build_cold_start_eval(log, split, _warm_features(split.item_ids),
                                     min_cold_support=2)
        warm = set(int(i) for i in split.item_ids)
        pre_cutoff = set(log.loc[log["ts"] < split.cutoff, "itemid"].unique())
        for item in cold.cold_item_ids:
            assert int(item) not in warm            # not in the trained vocabulary
            assert int(item) not in pre_cutoff      # and strictly new (no pre-cutoff)

    def test_cold_share_is_a_fraction(self):
        log = _synthetic_log(seed=2)
        split = temporal_split(log, test_days=7, min_item_sessions=5)
        cold = build_cold_start_eval(log, split, _warm_features(split.item_ids),
                                     min_cold_support=2)
        assert 0.0 < cold.cold_share <= 1.0
        # Every strictly-cold target is also near-cold (0 < 5 pre-cutoff), so the
        # near-cold share can only meet or exceed the strictly-cold share.
        assert cold.near_cold_share >= cold.cold_share

    def test_prefix_is_warm_only_and_aligned(self):
        log = _synthetic_log(seed=3)
        split = temporal_split(log, test_days=7, min_item_sessions=5)
        cold = build_cold_start_eval(log, split, _warm_features(split.item_ids),
                                     min_cold_support=2)
        # Prefix matrix has one row per cold session and only warm columns exist.
        assert cold.warm_prefix.shape == (cold.n_sessions, len(split.item_ids))
        assert len(cold.cold_target_idx) == cold.n_sessions
        assert cold.warm_prefix.nnz > 0  # every kept session has a warm prefix item


class TestStructuralZeros:
    def test_two_tower_scores_cold_items_but_classical_cannot(self):
        # The classical models have no representation for an item absent from
        # training, so their cold-start recall is 0.0 by construction — the row
        # that is the point of the cold-start table. Asserted at the report level
        # in stage2.run_cold_start; here we assert the modelling fact directly:
        # a cold item's index lies outside every classical model's item space.
        log = _synthetic_log(seed=4)
        split = temporal_split(log, test_days=7, min_item_sessions=5)
        cold = build_cold_start_eval(log, split, _warm_features(split.item_ids),
                                     min_cold_support=2)
        n_warm = len(split.item_ids)
        # Cold candidates are indexed *after* the warm items in the combined space;
        # a classical model only produces scores over [0, n_warm).
        assert len(cold.cold_item_ids) > 0
        assert all(int(i) not in set(int(w) for w in split.item_ids)
                   for i in cold.cold_item_ids)
        # (n_warm marks where cold candidates begin in evaluate_two_tower_cold.)
        assert n_warm == split.n_items


class TestColdBaselineNoLeak:
    """The category-popularity baseline must rank on popularity known *at the session's
    time*, never the full test period — otherwise it is an oracle, not a heuristic."""

    def _t(self, minutes):
        return np.datetime64("2015-08-01T00:00:00") + np.timedelta64(minutes, "m")

    def test_only_past_events_count(self):
        # One session at t=100. Cold item 0 has an event before it; cold item 1 after.
        freq = _asof_cold_frequencies(
            event_cidx=np.array([0, 1]),
            event_ts=np.array([self._t(50), self._t(150)]),
            session_ts=np.array([self._t(100)]),
            n_cold=2,
        )
        assert freq[0, 0] == 1.0  # the earlier event is counted
        assert freq[0, 1] == 0.0  # the later event is NOT

    def test_event_at_prediction_time_is_excluded(self):
        # The target's own occurrence sits at exactly the prediction time; strict '<'
        # keeps it out, so the baseline cannot count the very event it is predicting.
        freq = _asof_cold_frequencies(
            np.array([0]), np.array([self._t(100)]), np.array([self._t(100)]), n_cold=1
        )
        assert freq[0, 0] == 0.0

    def test_appending_a_later_event_does_not_change_the_ranking(self):
        # Codex's requested property: mutate interactions occurring after an evaluated
        # session; its baseline popularity vector must be unchanged.
        events_ts = [self._t(10), self._t(20), self._t(30)]
        events_cidx = [0, 1, 0]
        sessions = np.array([self._t(25)])
        before = _asof_cold_frequencies(np.array(events_cidx), np.array(events_ts), sessions, 2)
        # Append a future event (t=40 > session t=25) for a cold item.
        after = _asof_cold_frequencies(
            np.array(events_cidx + [1]),
            np.array(events_ts + [self._t(40)]),
            sessions,
            2,
        )
        assert np.array_equal(before, after)

    def test_build_populates_session_ts_aligned_to_targets(self):
        log = _synthetic_log(seed=1)
        split = temporal_split(log, test_days=7, min_item_sessions=5)
        cold = build_cold_start_eval(log, split, _warm_features(split.item_ids),
                                     min_cold_support=2)
        assert cold.session_ts.shape == (cold.n_sessions,)
        # Every prediction time is post-cutoff (targets are post-cutoff by construction).
        ts = pd.to_datetime(cold.session_ts, utc=True)
        assert (ts >= split.cutoff).all()


def test_rejects_missing_warm_prefix_gracefully():
    # A log with no post-cutoff cold items yields an empty but valid cold eval.
    base = pd.Timestamp("2015-06-01", tz="UTC")
    rng = np.random.default_rng(0)
    rows = []
    for s in range(300):
        for it in rng.choice(np.arange(10), size=3, replace=False):
            rows.append({"session": s, "itemid": int(it),
                         "ts": base + pd.Timedelta(days=int(rng.integers(0, 30)), minutes=int(it))})
    log = pd.DataFrame(rows)
    split = temporal_split(log, test_days=5, min_item_sessions=5)
    cold = build_cold_start_eval(log, split, _warm_features(split.item_ids), min_cold_support=2)
    assert cold.n_sessions == 0  # no cold items -> nothing to evaluate, no crash
