"""Train/test split protocols.

Two protocols, deliberately: the one a deployed system faces, and the one the
literature mostly uses.

``temporal_split`` trains on every session that started before a cutoff and
tests on sessions after it. This is what production knows — the past.

``leave_one_out_split`` holds out one item from every session regardless of when
it happened, which is the field's default. It trains on interactions that
occurred *after* the items it is asked to predict. It exists here to be measured
against the temporal split, not because it is defensible.

Filter scope is the other honest-protocol lever. Fitting the k-core filter on the
whole dataset lets test-period activity decide which items existed during
training. That is a real leak, and it is what most published preprocessing does.
``filter_scope="train"`` (the default) fits on the training window alone;
``"global"`` reproduces the common practice so the gap can be reported.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

from reclab.data.filtering import FilterReport, k_core_filter


@dataclass(frozen=True)
class SessionSplit:
    """A fitted split, ready for models and evaluation.

    ``train`` is the binary session x item matrix models learn from.
    ``test_prefix`` holds each test session's items except the last;
    ``test_target`` holds the index of that withheld last item.
    """

    train: sp.csr_matrix
    test_prefix: sp.csr_matrix
    test_target: np.ndarray
    item_ids: np.ndarray
    cutoff: pd.Timestamp | None
    protocol: str
    filter_scope: str
    filter_report: FilterReport
    n_straddling_dropped: int = 0
    # First-touch-ordered item-index sequences, aligned row-for-row to the
    # matrices above. The binary matrices discard order; sequential models
    # (SASRec) need it, so it is carried alongside rather than reconstructed —
    # a bag cannot be un-shuffled. Classical models ignore these.
    train_sequences: list[np.ndarray] | None = None
    test_prefix_sequences: list[np.ndarray] | None = None

    @property
    def n_items(self) -> int:
        return self.train.shape[1]

    @property
    def n_train_sessions(self) -> int:
        return self.train.shape[0]

    @property
    def n_test_sessions(self) -> int:
        return self.test_prefix.shape[0]

    @property
    def relevant(self) -> list[set[int]]:
        """Held-out item per test session, in the form the metrics expect."""
        return [{int(t)} for t in self.test_target]

    def __str__(self) -> str:
        cutoff = self.cutoff.date() if self.cutoff is not None else "n/a"
        return (
            f"{self.protocol} split (filter_scope={self.filter_scope}, cutoff={cutoff}): "
            f"{self.n_train_sessions:,} train sessions x {self.n_items:,} items, "
            f"{self.train.nnz:,} interactions; {self.n_test_sessions:,} test sessions"
        )


def _build_matrix(
    pairs: pd.DataFrame, item_to_col: dict[int, int], session_order: np.ndarray
) -> sp.csr_matrix:
    """Binary session x item matrix over the given session ordering."""
    session_to_row = {s: i for i, s in enumerate(session_order)}
    rows = pairs["session"].map(session_to_row).to_numpy()
    cols = pairs["itemid"].map(item_to_col).to_numpy()
    data = np.ones(len(pairs), dtype=np.float32)
    return sp.csr_matrix(
        (data, (rows, cols)), shape=(len(session_order), len(item_to_col))
    )


def _build_sequences(
    pairs: pd.DataFrame, item_to_col: dict[int, int], session_order: np.ndarray
) -> list[np.ndarray]:
    """First-touch-ordered item-index sequence per session, aligned to session_order.

    The order carried here is what sequential models consume and what the binary
    matrix throws away. Empty sequences (a session whose items were all filtered)
    come back as empty arrays, never dropped, so row alignment with the matrix
    holds exactly.
    """
    ordered = pairs.sort_values(["session", "ts"], kind="mergesort")
    grouped = {
        session: group["itemid"].map(item_to_col).to_numpy(dtype=np.int64)
        for session, group in ordered.groupby("session", sort=False)
    }
    empty = np.empty(0, dtype=np.int64)
    return [grouped.get(session, empty) for session in session_order]


def temporal_split(
    session_items: pd.DataFrame,
    test_days: int = 28,
    min_session_items: int = 2,
    min_item_sessions: int = 10,
    filter_scope: str = "train",
) -> SessionSplit:
    """Global temporal split at ``test_days`` before the end of the log.

    ``session_items`` holds distinct (session, itemid, ts) rows in first-touch
    order within each session.
    """
    if filter_scope not in ("train", "global"):
        raise ValueError(f"filter_scope must be 'train' or 'global', got {filter_scope!r}")
    if test_days <= 0:
        raise ValueError(f"test_days must be positive, got {test_days}")

    bounds = session_items.groupby("session")["ts"].agg(["min", "max"])
    cutoff = session_items["ts"].max().normalize() - pd.Timedelta(days=test_days)

    # A session belongs wholly to train (it *ends* before the cutoff) or wholly
    # to test (it *starts* at or after the cutoff). Sessions straddling the
    # cutoff are dropped, not truncated: assigning their post-cutoff items to
    # training leaks the future, and truncating them evaluates on a session the
    # model half-saw. A session spans at most a few hours against a 28-day test
    # window, so only a handful straddle — dropping them costs nothing and keeps
    # the guarantee exact (a test asserts no training item has ts >= cutoff).
    train_ids = set(bounds.index[bounds["max"] < cutoff])
    test_ids = set(bounds.index[bounds["min"] >= cutoff])
    n_straddling = len(bounds) - len(train_ids) - len(test_ids)

    train_pairs_raw = session_items[session_items["session"].isin(train_ids)]
    test_pairs_raw = session_items[session_items["session"].isin(test_ids)]

    # The filter decides the item vocabulary. Fitting it on the training window
    # alone is the whole point of filter_scope="train": under "global", an item
    # survives because of activity that happens after the cutoff, which the
    # model would not have known about.
    fit_on = session_items if filter_scope == "global" else train_pairs_raw
    filtered, report = k_core_filter(fit_on, min_session_items, min_item_sessions)

    kept_items = np.sort(filtered["itemid"].unique())
    item_to_col = {int(item): i for i, item in enumerate(kept_items)}

    train_pairs = train_pairs_raw[train_pairs_raw["itemid"].isin(item_to_col)]
    train_counts = train_pairs["session"].value_counts()
    keep_train = train_counts.index[train_counts >= min_session_items]
    train_pairs = train_pairs[train_pairs["session"].isin(set(keep_train))]
    train_order = np.sort(train_pairs["session"].unique())

    # Test sessions are restricted to the training vocabulary — an item with no
    # training history is unscoreable by every model here, so evaluating on it
    # would measure nothing. Those items are the cold-start question, taken up
    # in Stage 2 on a separate evaluation track.
    test_pairs = test_pairs_raw[test_pairs_raw["itemid"].isin(item_to_col)]
    test_counts = test_pairs["session"].value_counts()
    keep_test = test_counts.index[test_counts >= 2]  # need a prefix and a target
    test_pairs = test_pairs[test_pairs["session"].isin(set(keep_test))]

    train = _build_matrix(train_pairs, item_to_col, train_order)
    train_sequences = _build_sequences(train_pairs, item_to_col, train_order)
    prefix, target, prefix_sequences = _prefix_target(test_pairs, item_to_col)

    return SessionSplit(
        train=train,
        test_prefix=prefix,
        test_target=target,
        item_ids=kept_items,
        cutoff=cutoff,
        protocol="temporal",
        filter_scope=filter_scope,
        filter_report=report,
        n_straddling_dropped=n_straddling,
        train_sequences=train_sequences,
        test_prefix_sequences=prefix_sequences,
    )


def _prefix_target(
    test_pairs: pd.DataFrame, item_to_col: dict[int, int]
) -> tuple[sp.csr_matrix, np.ndarray, list[np.ndarray]]:
    """Split each test session into (all items but the last, the last item).

    Order is first-touch time within the session. Predicting the final item from
    its predecessors is the standard session-based protocol, which keeps these
    numbers comparable to the session-based literature.
    """
    ordered = test_pairs.sort_values(["session", "ts"], kind="mergesort")
    is_last = ~ordered["session"].duplicated(keep="last")

    targets_df = ordered[is_last]
    prefix_df = ordered[~is_last]

    session_order = targets_df["session"].to_numpy()
    prefix = _build_matrix(prefix_df, item_to_col, session_order)
    prefix_sequences = _build_sequences(prefix_df, item_to_col, session_order)
    target = targets_df["itemid"].map(item_to_col).to_numpy()
    return prefix, target, prefix_sequences


def leave_one_out_split(
    session_items: pd.DataFrame,
    min_session_items: int = 2,
    min_item_sessions: int = 10,
    seed: int = 0,
) -> SessionSplit:
    """The field's default protocol, present so it can be measured against.

    Every session contributes one randomly held-out item; the model trains on
    everything else — including sessions that happened *after* the items it is
    being asked to predict. No cutoff, no time ordering.
    """
    filtered, report = k_core_filter(session_items, min_session_items, min_item_sessions)
    kept_items = np.sort(filtered["itemid"].unique())
    item_to_col = {int(item): i for i, item in enumerate(kept_items)}

    rng = np.random.default_rng(seed)
    shuffled = filtered.sample(frac=1.0, random_state=seed)
    held_out = shuffled.groupby("session", sort=False).head(1)
    held_idx = set(held_out.index)
    train_pairs = filtered[~filtered.index.isin(held_idx)]

    # Sessions that lose their only remaining item can't inform training.
    counts = train_pairs["session"].value_counts()
    keep = counts.index[counts >= 1]
    train_pairs = train_pairs[train_pairs["session"].isin(set(keep))]
    held_out = held_out[held_out["session"].isin(set(keep))]

    session_order = np.sort(train_pairs["session"].unique())
    train = _build_matrix(train_pairs, item_to_col, session_order)
    train_sequences = _build_sequences(train_pairs, item_to_col, session_order)

    eval_sessions = held_out["session"].to_numpy()
    eval_pairs = train_pairs[train_pairs["session"].isin(set(eval_sessions))]
    prefix = _build_matrix(eval_pairs, item_to_col, eval_sessions)
    prefix_sequences = _build_sequences(eval_pairs, item_to_col, eval_sessions)
    target = held_out["itemid"].map(item_to_col).to_numpy()
    _ = rng  # seeding is via sample(random_state=seed); kept for signature clarity

    return SessionSplit(
        train=train,
        test_prefix=prefix,
        test_target=target,
        item_ids=kept_items,
        cutoff=None,
        protocol="leave_one_out",
        filter_scope="global",
        filter_report=report,
        train_sequences=train_sequences,
        test_prefix_sequences=prefix_sequences,
    )
