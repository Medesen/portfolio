"""Loading, validation and sessionization of the RetailRocket event log.

Why sessions rather than users
------------------------------
RetailRocket ``visitorid``s are cookies, not accounts. 79.6% of visitors view
exactly one item ever, and the mean visitor has 1.25 sessions — so a protocol
built on "users who return" evaluates on roughly a thousand people out of 1.4
million. The session is the unit that actually carries signal: someone who
arrives once, views four products and never comes back has still told us
something about how those four products relate.

Session boundaries are drawn at a 30-minute inactivity gap, the convention in
the session-based recommendation literature (Ludewig & Jannach 2018).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

EVENT_TYPES = ("view", "addtocart", "transaction")
_EVENTS_REL = Path("data") / "raw" / "events.csv.gz"


def default_events_path() -> Path:
    """Locate the bundled event log.

    Resolution order: the ``RECLAB_DATA`` env var, then the working directory
    (the Docker WORKDIR and the usual `reclab`-from-repo-root case), then a path
    relative to this module (editable/dev installs). ``__file__`` alone is not
    enough: in a regular (non-editable) install it points into site-packages,
    nowhere near the bundled data — which is exactly how the Docker image installs
    the package.
    """
    if env := os.environ.get("RECLAB_DATA"):
        return Path(env)
    candidates = [
        Path.cwd() / _EVENTS_REL,
        Path(__file__).resolve().parents[3] / _EVENTS_REL,  # data->reclab->src->root
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]  # report the CWD-relative path in the error

# The published collection window. Timestamps outside it mean the file is not
# the dataset we think it is.
EXPECTED_START = pd.Timestamp("2015-05-01", tz="UTC")
EXPECTED_END = pd.Timestamp("2015-09-20", tz="UTC")


@dataclass(frozen=True)
class LoadReport:
    """What the loader did, for DATA_NOTES and the EDA command."""

    rows_read: int
    rows_after_event_filter: int
    exact_duplicates_dropped: int
    n_visitors: int
    n_items: int
    first_event: pd.Timestamp
    last_event: pd.Timestamp

    def __str__(self) -> str:
        return (
            f"loaded {self.rows_read:,} rows -> {self.rows_after_event_filter:,} after "
            f"event-type filter, dropped {self.exact_duplicates_dropped:,} exact "
            f"duplicates; {self.n_visitors:,} visitors, {self.n_items:,} items, "
            f"{self.first_event.date()} to {self.last_event.date()}"
        )


def load_events(
    path: Path | None = None,
    event_types: tuple[str, ...] = EVENT_TYPES,
) -> tuple[pd.DataFrame, LoadReport]:
    """Read the event log, validate it, and return it with a load report.

    Returns columns ``visitorid``, ``itemid``, ``event``, ``ts`` (UTC datetime).

    All three event types are implicit positives by default. This is close to a
    free choice on this data: views alone give 2,132,127 distinct visitor-item
    pairs and all three give 2,145,179 — a 0.6% difference. Views are 99.4% of
    the signal, so the event-type question that looks important turns out not to
    be, and the reasoning is recorded in DATA_NOTES rather than left implicit.
    """
    path = Path(path) if path is not None else default_events_path()
    if not path.exists():
        raise FileNotFoundError(
            f"event log not found at {path}. The bundled file is "
            "data/raw/events.csv.gz; run from the project root or set RECLAB_DATA. "
            "See data/raw/README.md for provenance."
        )

    unknown = set(event_types) - set(EVENT_TYPES)
    if unknown:
        raise ValueError(f"unknown event types {sorted(unknown)}; valid: {EVENT_TYPES}")

    df = pd.read_csv(path)
    rows_read = len(df)

    required = {"timestamp", "visitorid", "event", "itemid"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"event log missing required columns: {sorted(missing)}")

    if df["visitorid"].isna().any() or df["itemid"].isna().any():
        raise ValueError("event log contains null visitorid or itemid")

    observed = set(df["event"].unique())
    if not observed <= set(EVENT_TYPES):
        raise ValueError(f"unexpected event values: {sorted(observed - set(EVENT_TYPES))}")

    df["ts"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    if df["ts"].min() < EXPECTED_START or df["ts"].max() > EXPECTED_END:
        raise ValueError(
            f"timestamps span {df['ts'].min()} to {df['ts'].max()}, outside the "
            f"published collection window {EXPECTED_START.date()}..{EXPECTED_END.date()}"
        )

    df = df[df["event"].isin(event_types)]
    after_filter = len(df)

    before_dedup = len(df)
    df = df.drop_duplicates(subset=["timestamp", "visitorid", "itemid", "event"])
    duplicates = before_dedup - len(df)

    df = df[["visitorid", "itemid", "event", "ts"]].reset_index(drop=True)
    report = LoadReport(
        rows_read=rows_read,
        rows_after_event_filter=after_filter,
        exact_duplicates_dropped=duplicates,
        n_visitors=df["visitorid"].nunique(),
        n_items=df["itemid"].nunique(),
        first_event=df["ts"].min(),
        last_event=df["ts"].max(),
    )
    return df, report


def sessionize(df: pd.DataFrame, gap_minutes: int = 30) -> pd.DataFrame:
    """Add a ``session`` column, splitting each visitor's events on inactivity.

    A new session starts at a visitor's first event or after a gap longer than
    ``gap_minutes``. Session ids are integers, assigned in (visitor, time) order.
    """
    if gap_minutes <= 0:
        raise ValueError(f"gap_minutes must be positive, got {gap_minutes}")

    df = df.sort_values(["visitorid", "ts"], kind="mergesort").reset_index(drop=True)
    gap = df.groupby("visitorid")["ts"].diff()
    starts_session = gap.isna() | (gap > pd.Timedelta(minutes=gap_minutes))
    df["session"] = starts_session.cumsum().astype("int64")
    return df


def to_session_items(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse to distinct (session, item) pairs, keeping first-touch order.

    Implicit-feedback models consume a binary session x item matrix, so repeat
    views of the same item within a session collapse. The repeat *count* does
    carry signal — P(purchase) rises from 0.4% at one view to 12.4% at six —
    which is what ALS's confidence weighting exists to exploit, so the count is
    retained in the ``n_events`` column rather than discarded.
    """
    grouped = (
        df.groupby(["session", "itemid"], sort=False)
        .agg(ts=("ts", "min"), n_events=("ts", "size"))
        .reset_index()
    )
    return grouped.sort_values(["session", "ts"], kind="mergesort").reset_index(drop=True)
