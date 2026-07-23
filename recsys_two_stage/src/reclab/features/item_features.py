"""Item content features, snapshotted as-of the training cutoff.

The leakage hazard of Stage 2
-----------------------------
`item_properties.csv.gz` is a **timestamped change log**, not a snapshot: an item's
category and availability change over time. Taking an item's *current* category would
read a value that may have been set after the training cutoff — future information
leaking into the item tower. So for every item and every property we take the most
recent value with `timestamp <= cutoff`, and nothing later. A test asserts no feature
value derives from a post-cutoff row.

Items with no pre-cutoff value for a property get an explicit **"unknown"** index
(0), not a silent default: "we did not know this item's category at training time" is
itself information, and the model should be able to use it. This is also what makes
the content path honest for cold items — a brand-new item genuinely has no known
category, and the unknown index says exactly that.

The feature set is deliberately thin — category, parent category, availability —
because the distilled properties are all this dataset offers (the raw property values
are anonymised hashes with no readable text). The category vocabulary is fit on the
*training* items only; a category never seen in training maps to unknown, which is the
correct behaviour for the cold-start evaluation that reuses this vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from reclab.data.load import default_events_path

UNKNOWN = 0  # reserved index for "not known as-of cutoff" in every vocabulary


def _data_dir() -> Path:
    return default_events_path().parent


@dataclass
class ItemFeatures:
    """Content features aligned to an item index, plus the fitted vocabularies.

    ``category_ids`` / ``parent_ids`` are (n_items,) integer arrays indexing the
    category / parent-category embedding tables (0 = unknown). ``available`` is
    (n_items,) float in {0, 1}. The vocabularies are returned so the *same* mapping
    can be reused to build features for cold items without refitting.
    """

    item_ids: np.ndarray
    category_ids: np.ndarray
    parent_ids: np.ndarray
    available: np.ndarray
    category_vocab: dict[int, int]  # raw categoryid -> embedding index
    parent_vocab: dict[int, int]  # raw parentid   -> embedding index

    @property
    def n_categories(self) -> int:
        return len(self.category_vocab) + 1  # +1 for the unknown index

    @property
    def n_parents(self) -> int:
        return len(self.parent_vocab) + 1

    def coverage(self) -> dict[str, float]:
        """Fraction of items with a known value for each feature (as-of cutoff)."""
        n = len(self.item_ids)
        return {
            "n_items": n,
            "category_known": float((self.category_ids != UNKNOWN).mean()),
            "parent_known": float((self.parent_ids != UNKNOWN).mean()),
            "available_known": float((self.available >= 0).mean()),
            "n_categories": self.n_categories - 1,
            "n_parents": self.n_parents - 1,
        }


def _as_of_snapshot(props: pd.DataFrame, cutoff: pd.Timestamp, prop: str) -> pd.Series:
    """Most recent value of ``prop`` per item with timestamp <= cutoff."""
    sub = props[(props["property"] == prop) & (props["ts"] <= cutoff)]
    # Sort by time, then keep the last (latest) row per item.
    sub = sub.sort_values("ts", kind="mergesort")
    return sub.groupby("itemid")["value"].last()


def build_item_features(
    cutoff: pd.Timestamp,
    item_index: np.ndarray,
    category_vocab: dict[int, int] | None = None,
    parent_vocab: dict[int, int] | None = None,
    properties_path: Path | None = None,
    tree_path: Path | None = None,
) -> ItemFeatures:
    """Build as-of-cutoff content features for the items in ``item_index``.

    Pass ``category_vocab`` / ``parent_vocab`` to reuse a vocabulary fitted on the
    training items (the cold-start path); leave them ``None`` to fit fresh from
    ``item_index`` (the warm path).
    """
    data_dir = _data_dir()
    properties_path = properties_path or data_dir / "item_properties.csv.gz"
    tree_path = tree_path or data_dir / "category_tree.csv.gz"

    props = pd.read_csv(properties_path)
    props["ts"] = pd.to_datetime(props["timestamp"], unit="ms", utc=True)
    tree = pd.read_csv(tree_path)
    cat_to_parent = dict(zip(tree["categoryid"], tree["parentid"]))

    cat_snapshot = _as_of_snapshot(props, cutoff, "categoryid").astype("Int64")
    avail_snapshot = _as_of_snapshot(props, cutoff, "available").astype("Int64")

    item_index = np.asarray(item_index)
    raw_category = np.array(
        [cat_snapshot.get(item, pd.NA) for item in item_index], dtype=object
    )
    raw_parent = np.array(
        [cat_to_parent.get(int(c)) if pd.notna(c) else None for c in raw_category],
        dtype=object,
    )

    # Fit vocabularies from the observed (non-missing) values if none supplied.
    fitting = category_vocab is None
    if fitting:
        cats = sorted({int(c) for c in raw_category if pd.notna(c)})
        parents = sorted({int(p) for p in raw_parent if p is not None and pd.notna(p)})
        category_vocab = {c: i + 1 for i, c in enumerate(cats)}  # 0 reserved
        parent_vocab = {p: i + 1 for i, p in enumerate(parents)}

    def encode(value, vocab) -> int:
        if value is None or pd.isna(value):
            return UNKNOWN
        return vocab.get(int(value), UNKNOWN)  # unseen category -> unknown

    category_ids = np.array([encode(c, category_vocab) for c in raw_category], dtype=np.int64)
    parent_ids = np.array([encode(p, parent_vocab) for p in raw_parent], dtype=np.int64)
    # Availability: most recent known value, default 0 (treated as "not available /
    # unknown"). Kept as a plain binary rather than a 3-way, since 0 is the safe
    # assumption for an item we have no availability record for.
    available = np.array(
        [float(avail_snapshot.get(item, 0)) for item in item_index], dtype=np.float32
    )

    return ItemFeatures(
        item_ids=item_index,
        category_ids=category_ids,
        parent_ids=parent_ids,
        available=available,
        category_vocab=category_vocab,
        parent_vocab=parent_vocab,
    )
