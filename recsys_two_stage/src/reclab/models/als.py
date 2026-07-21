"""Implicit-feedback ALS (Hu, Koren & Volinsky 2008), via the `implicit` library.

Why the library learns the factors but we score by hand
-------------------------------------------------------
`implicit` provides a fast, standard, Cython ALS — no reason to reimplement the
optimisation. But it learns one latent vector per *training* session, and our
test sessions are new rows that never recur, so those vectors are useless to us.
What transfers is the item-factor matrix Y.

To score a new session we fold it in: solve for the session's latent vector from
its item history, holding Y fixed. This is exactly the per-user solve inside ALS
(Hu et al. eq. 4), run once for an unseen session:

    A = YᵀY + α·Y_Sᵀ Y_S + λI
    b = (1 + α)·Σ_{i∈S} y_i
    u = A⁻¹ b

with confidence c_i = 1 + α on the |S| interacted items and 1 elsewhere. Doing
the fold-in ourselves keeps ALS behind the same "score unseen sessions from their
history" contract as every other model here, and matches the no-per-user-identity
constraint the dataset forces.

Repeat-view counts carry real signal on this data — P(purchase) climbs from 0.4%
at one view to 12.4% at six — which is exactly what the confidence weighting α is
for. The training matrix passes those counts through (see ``use_counts``).
"""

from __future__ import annotations

import os

import numpy as np
import scipy.sparse as sp

from reclab.models.base import as_matrix, check_fitted

# implicit spawns a thread pool and prints a progress bar; keep both quiet and
# deterministic. Threads default to 1 so results are reproducible; callers can
# raise it for the full run where bit-reproducibility is not required.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


class ALS:
    name = "als"

    def __init__(
        self,
        factors: int = 64,
        regularization: float = 10.0,
        iterations: int = 15,
        alpha: float = 40.0,
        use_counts: bool = True,
        num_threads: int = 1,
        seed: int = 0,
    ) -> None:
        if factors < 1:
            raise ValueError(f"factors must be >= 1, got {factors}")
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha
        self.use_counts = use_counts
        self.num_threads = num_threads
        self.seed = seed
        self.item_factors_: np.ndarray | None = None
        self._YtY: np.ndarray | None = None

    def fit(self, X: sp.csr_matrix) -> "ALS":
        # Local import so the rest of the package (and its tests) load without
        # implicit present; only ALS needs it.
        from implicit.als import AlternatingLeastSquares

        confidence = X.astype(np.float32)
        if not self.use_counts:
            confidence = (confidence > 0).astype(np.float32)

        model = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            alpha=self.alpha,
            num_threads=self.num_threads,
            random_state=self.seed,
            use_gpu=False,
        )
        # implicit expects a (users x items) confidence matrix; our sessions play
        # the role of users. It applies confidence = 1 + alpha * value internally.
        model.fit(confidence, show_progress=False)

        self.item_factors_ = np.asarray(model.item_factors, dtype=np.float64)
        self._YtY = self.item_factors_.T @ self.item_factors_
        return self

    def score(self, history) -> np.ndarray:
        check_fitted(self, "item_factors_")
        Y = self.item_factors_
        reg_eye = self.regularization * np.eye(self.factors)
        histories = as_matrix(history).tocsr()

        session_factors = np.empty((histories.shape[0], self.factors), dtype=np.float64)
        for row in range(histories.shape[0]):
            items = histories.indices[histories.indptr[row] : histories.indptr[row + 1]]
            if len(items) == 0:
                session_factors[row] = 0.0
                continue
            Ys = Y[items]  # (|S|, f)
            A = self._YtY + self.alpha * (Ys.T @ Ys) + reg_eye
            b = (1.0 + self.alpha) * Ys.sum(axis=0)
            session_factors[row] = np.linalg.solve(A, b)

        return session_factors @ Y.T
