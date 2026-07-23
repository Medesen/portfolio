"""Item-item nearest neighbours: "sessions containing this also contained that".

This is the baseline that beat 11 of 18 published neural recommenders in
Dacrema, Cremonesi & Jannach (2019) once it was tuned properly. Their point was
not that the neural models were bad — it was that the baselines they were
compared against had not been tuned. So this one is tuned, on a validation
window, like everything else here.

Similarity is cosine over the binary session x item matrix with a shrinkage
term:

    sim(i, j) = c_ij / (sqrt(c_ii * c_jj) + shrink)

``shrink`` damps pairs whose co-occurrence rests on very few sessions. Without
it, two obscure items seen together twice score a perfect 1.0 and outrank
genuinely related popular pairs — the classic failure mode of raw cosine on
sparse count data.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from reclab.models.base import as_matrix, check_fitted


def _top_k_per_row(matrix: sp.csr_matrix, k: int) -> sp.csr_matrix:
    """Keep only the k largest entries in each row, dropping the rest."""
    matrix = matrix.tocsr()
    rows, cols, vals = [], [], []
    for row in range(matrix.shape[0]):
        start, end = matrix.indptr[row], matrix.indptr[row + 1]
        if start == end:
            continue
        row_cols = matrix.indices[start:end]
        row_vals = matrix.data[start:end]
        if len(row_vals) > k:
            keep = np.argpartition(-row_vals, k - 1)[:k]
            row_cols, row_vals = row_cols[keep], row_vals[keep]
        rows.append(np.full(len(row_cols), row, dtype=np.int32))
        cols.append(row_cols)
        vals.append(row_vals)
    if not rows:
        return sp.csr_matrix(matrix.shape, dtype=np.float64)
    return sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=matrix.shape,
    )


class ItemKNN:
    name = "itemknn"

    def __init__(self, k: int = 100, shrink: float = 100.0) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if shrink < 0:
            raise ValueError(f"shrink must be >= 0, got {shrink}")
        self.k = k
        self.shrink = shrink
        self.similarity_: sp.csr_matrix | None = None

    def fit(self, X: sp.csr_matrix) -> "ItemKNN":
        X = X.tocsc().astype(np.float64)
        # Co-occurrence counts. Stays sparse: only item pairs actually seen
        # together get an entry, which on this data is well under 1% of pairs.
        cooccurrence = (X.T @ X).tocoo()
        support = np.asarray(cooccurrence.tocsr().diagonal()).ravel()

        denominator = (
            np.sqrt(support[cooccurrence.row] * support[cooccurrence.col]) + self.shrink
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            values = np.where(denominator > 0, cooccurrence.data / denominator, 0.0)

        similarity = sp.csr_matrix(
            (values, (cooccurrence.row, cooccurrence.col)), shape=cooccurrence.shape
        )
        # An item is trivially its own nearest neighbour; leaving the diagonal
        # in would make every model score the session's own items highest.
        similarity.setdiag(0.0)
        similarity.eliminate_zeros()

        self.similarity_ = _top_k_per_row(similarity, self.k)
        return self

    def score(self, history) -> np.ndarray:
        check_fitted(self, "similarity_")
        histories = as_matrix(history)
        return np.asarray((histories @ self.similarity_).todense(), dtype=np.float64)
