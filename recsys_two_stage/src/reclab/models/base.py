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

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import scipy.sparse as sp


@dataclass
class HistoryBatch:
    """A chunk of session histories, carrying both representations models need.

    ``matrix`` is the (n, n_items) binary bag every classical model consumes.
    ``sequences`` is the first-touch-ordered item-index list a sequential model
    (SASRec) needs — a bag cannot be un-shuffled, so order is carried, not
    reconstructed. Classical models ignore ``sequences``; ``sequences`` is
    ``None`` when a caller (e.g. a unit test) passes only a matrix.

    The evaluation loop builds one of these per chunk and hands it to every
    model, so the harness itself stays model-agnostic — the Stage 1 abstraction
    generalised to carry order, rather than special-cased per model.
    """

    matrix: sp.csr_matrix
    sequences: list[np.ndarray] | None = None

    def __len__(self) -> int:
        return self.matrix.shape[0]


def as_matrix(history: "HistoryBatch | sp.csr_matrix") -> sp.csr_matrix:
    """Extract the binary matrix from a HistoryBatch, or pass a csr through.

    Lets classical models accept either a HistoryBatch (from the harness) or a
    bare csr matrix (from unit tests) without caring which."""
    return history.matrix if isinstance(history, HistoryBatch) else history


@runtime_checkable
class Recommender(Protocol):
    """Fit on a session x item matrix, score sessions from their history."""

    name: str

    def fit(self, X: sp.csr_matrix) -> "Recommender":
        """Learn from the binary training matrix. Returns self."""
        ...

    def score(self, history: "HistoryBatch | sp.csr_matrix") -> np.ndarray:
        """Score every item for each session, given that session's prefix items.

        ``history`` carries the (n_sessions, n_items) binary bag and, for models
        that need it, the ordered sequences. Returns a dense float array of shape
        (n_sessions, n_items). Items already in a session's history are *not*
        excluded here — masking happens once, centrally, in the evaluation loop.
        """
        ...


def check_fitted(model: object, attribute: str) -> None:
    """Raise a clear error if ``score`` is called before ``fit``."""
    if getattr(model, attribute, None) is None:
        raise RuntimeError(
            f"{type(model).__name__} has not been fitted — call fit() before score()"
        )
