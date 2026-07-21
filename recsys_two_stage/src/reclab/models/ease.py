"""EASE — Embarrassingly Shallow Autoencoder (Steck, WWW 2019).

A linear item-item model with a closed-form solution and exactly one
hyperparameter. About fifteen lines of linear algebra, routinely competitive
with far more elaborate neural models:

    G = XᵀX + λI
    P = G⁻¹
    B = I − P · diag(1 / diag(P)),  with the diagonal of B forced to zero
    scores = X · B

The zero diagonal is what stops the model learning the identity function — the
constraint that makes the closed form non-trivial.

The memory wall
---------------
G is items x items and *dense*. Cost grows with the square of the catalogue:

    10,000 items   ->   0.8 GB      comfortable
    20,000 items   ->   3.2 GB      workable
    40,000 items   ->  12.8 GB      marginal
   235,000 items   -> 441.8 GB      impossible

and the inversion needs a second array of the same size. This is not an
implementation detail to route around. It is the reason a model that can win the
benchmark is not the model anyone deploys, and it is measured rather than
asserted — see the catalogue-scaling sweep. The guard below fails loudly with
the arithmetic rather than letting the process die to the OOM killer.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from reclab.models.base import check_fitted


def estimate_memory_gb(n_items: int) -> float:
    """Bytes for one dense float64 item x item matrix, in GB."""
    return n_items**2 * 8 / 1e9


class EASE:
    name = "ease"

    def __init__(self, reg: float = 250.0, max_gb: float = 6.0) -> None:
        if reg <= 0:
            raise ValueError(f"reg must be positive, got {reg}")
        self.reg = reg
        self.max_gb = max_gb
        self.B_: np.ndarray | None = None

    def fit(self, X: sp.csr_matrix) -> "EASE":
        n_items = X.shape[1]
        needed = estimate_memory_gb(n_items)
        if needed > self.max_gb:
            raise MemoryError(
                f"EASE needs a dense {n_items:,} x {n_items:,} matrix "
                f"({needed:.1f} GB, and ~{2 * needed:.1f} GB peak during inversion), "
                f"above the {self.max_gb:.1f} GB limit. This is EASE's defining "
                f"constraint, not a bug: its memory grows with the square of the "
                f"catalogue. Raise max_gb, or shrink the catalogue with a stricter "
                f"item filter."
            )

        X = X.astype(np.float64)
        gram = np.asarray((X.T @ X).todense())
        # Ridge term on the diagonal. Without it G is singular whenever an item
        # co-occurs with nothing, which on sparse session data is common.
        gram[np.diag_indices(n_items)] += self.reg

        precision = np.linalg.inv(gram)
        del gram

        # B_ij = -P_ij / P_jj, then zero the diagonal.
        B = precision / (-np.diag(precision))[None, :]
        B[np.diag_indices(n_items)] = 0.0
        self.B_ = B
        return self

    def score(self, histories: sp.csr_matrix) -> np.ndarray:
        check_fitted(self, "B_")
        return np.asarray(histories @ self.B_, dtype=np.float64)
