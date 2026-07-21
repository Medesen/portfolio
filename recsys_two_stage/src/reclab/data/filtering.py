"""Iterative k-core filtering of the session x item interaction log.

Dropping sparse rows changes which columns are sparse, and vice versa — so the
filter has to be applied repeatedly until it stops changing anything. A single
pass leaves items below threshold, which is a common quiet bug: the reported
"5-core" dataset is not actually 5-core.

The scope of the filter is a design decision with teeth, and it lives in the
caller rather than here: fitting it on the whole dataset lets test-period
activity decide which items existed during training, which is a subtle leak that
most published work carries. See ``reclab.splitting.protocols``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class FilterReport:
    """Per-iteration trace, so the filter's behaviour is inspectable."""

    min_session_items: int
    min_item_sessions: int
    iterations: int = 0
    converged: bool = False
    history: list[dict] = field(default_factory=list)

    def record(self, sessions: int, items: int, pairs: int) -> None:
        self.history.append({"sessions": sessions, "items": items, "pairs": pairs})

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.history).rename_axis("iteration").reset_index()

    def __str__(self) -> str:
        if not self.history:
            return "filter: no iterations recorded"
        first, last = self.history[0], self.history[-1]
        status = "converged" if self.converged else "HIT ITERATION CAP"
        return (
            f"k-core(session>={self.min_session_items}, item>={self.min_item_sessions}): "
            f"{first['sessions']:,} sessions / {first['items']:,} items / "
            f"{first['pairs']:,} pairs -> {last['sessions']:,} / {last['items']:,} / "
            f"{last['pairs']:,} in {self.iterations} iterations ({status})"
        )


def k_core_filter(
    pairs: pd.DataFrame,
    min_session_items: int,
    min_item_sessions: int,
    max_iterations: int = 50,
) -> tuple[pd.DataFrame, FilterReport]:
    """Iteratively drop thin sessions and thin items until a fixed point.

    ``pairs`` holds distinct (session, item) rows. A session below
    ``min_session_items`` carries no usable co-occurrence; an item below
    ``min_item_sessions`` has too little support to estimate similarity for.

    Returns the filtered frame and a report tracing every iteration.
    """
    if min_session_items < 1 or min_item_sessions < 1:
        raise ValueError("thresholds must be >= 1")
    for col in ("session", "itemid"):
        if col not in pairs.columns:
            raise ValueError(f"pairs frame missing required column {col!r}")

    report = FilterReport(min_session_items, min_item_sessions)
    current = pairs
    report.record(current["session"].nunique(), current["itemid"].nunique(), len(current))

    for iteration in range(1, max_iterations + 1):
        before = len(current)

        session_counts = current["session"].value_counts()
        keep_sessions = session_counts.index[session_counts >= min_session_items]
        current = current[current["session"].isin(keep_sessions)]

        item_counts = current["itemid"].value_counts()
        keep_items = item_counts.index[item_counts >= min_item_sessions]
        current = current[current["itemid"].isin(keep_items)]

        report.iterations = iteration
        report.record(
            current["session"].nunique(), current["itemid"].nunique(), len(current)
        )

        if len(current) == before:
            report.converged = True
            break

    return current.reset_index(drop=True), report
