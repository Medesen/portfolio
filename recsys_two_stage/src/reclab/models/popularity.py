"""Recommend the most-viewed items to everybody.

The floor. This model ignores the session entirely, so anything that cannot beat
it is not doing personalisation — it is doing arithmetic with extra steps.

It is included because it is routinely competitive, not as a formality. Popular
items are popular for a reason, and on short sessions with thin co-occurrence
the gap between "what everyone looks at" and "what this session suggests" can be
small. Reporting that gap honestly is the point.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from reclab.models.base import check_fitted


class Popularity:
    name = "popularity"

    def __init__(self) -> None:
        self.counts_: np.ndarray | None = None

    def fit(self, X: sp.csr_matrix) -> "Popularity":
        # Number of distinct sessions containing each item.
        self.counts_ = np.asarray(X.sum(axis=0)).ravel().astype(np.float64)
        return self

    def score(self, histories: sp.csr_matrix) -> np.ndarray:
        check_fitted(self, "counts_")
        return np.tile(self.counts_, (histories.shape[0], 1))
