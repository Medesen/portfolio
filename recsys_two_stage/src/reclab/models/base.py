"""The interface every recommender in this project implements.

Models are fitted on a binary session x item matrix and then score *unseen*
sessions from their item history. That second part matters: test sessions are
not rows of the training matrix, so a model cannot rely on having a learned
parameter per session. This rules out per-user embeddings by construction —
which is the right constraint on a dataset where sessions never repeat, and it
is why the Stage 2 two-tower model encodes history rather than identity.

Scoring returns a dense (n_sessions, n_items) block. Callers chunk over sessions
to bound memory; models do not chunk internally.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import scipy.sparse as sp


@runtime_checkable
class Recommender(Protocol):
    """Fit on a session x item matrix, score sessions from their history."""

    name: str

    def fit(self, X: sp.csr_matrix) -> "Recommender":
        """Learn from the binary training matrix. Returns self."""
        ...

    def score(self, histories: sp.csr_matrix) -> np.ndarray:
        """Score every item for each session, given that session's prefix items.

        ``histories`` is (n_sessions, n_items) binary. Returns a dense float
        array of the same shape. Items already in a session's history are *not*
        excluded here — masking happens once, centrally, in the evaluation loop.
        """
        ...


def check_fitted(model: object, attribute: str) -> None:
    """Raise a clear error if ``score`` is called before ``fit``."""
    if getattr(model, attribute, None) is None:
        raise RuntimeError(
            f"{type(model).__name__} has not been fitted — call fit() before score()"
        )
