"""Catalogue-scaling sweep — EASE's memory wall, measured rather than asserted.

The project's closing argument is that the benchmark-winning model (EASE) does not
scale, while the benchmark-losing one (the two-tower / ALS embedding family) does.
Stage 1 stated EASE's memory arithmetically (n_items² × 8 bytes → 442 GB at the full
235k-item catalogue). This sweep makes the *time* side of that wall empirical: fit
EASE and ALS on the real training matrix restricted to the top-K most popular items,
for growing K, and watch EASE's fit time climb super-linearly (its dense solve is
O(K³)) while ALS stays roughly linear.

Honesty about what this can and cannot show: the actual filtered catalogue is 13,754
items, where EASE fits comfortably (~1.5 GB, under a minute). So EASE never *fails*
here — the sweep measures the super-linear growth on real subsets, and the leap to
"442 GB, impossible" at the full catalogue remains arithmetic, stated as arithmetic.
The measured curve plus the stated arithmetic is the honest whole.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

from reclab.models import ALS, EASE, estimate_memory_gb


def catalogue_scaling_sweep(
    split, sizes=(2000, 5000, 8000, 11000, 13754), als_params: dict | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Fit EASE and ALS on top-K-popularity subsets of the training matrix, for each K.

    Returns fit times and the analytical EASE dense-matrix footprint per K.
    """
    als_params = als_params or dict(factors=128, regularization=1.0, alpha=40.0,
                                    iterations=15, num_threads=4, seed=seed)
    train = split.train.tocsc()
    popularity = np.asarray(train.sum(axis=0)).ravel()
    order = np.argsort(-popularity)  # most popular items first

    rows = []
    for k in sizes:
        k = min(k, split.n_items)
        cols = np.sort(order[:k])
        sub = train[:, cols].tocsr()
        # Drop sessions left with fewer than two items after the restriction.
        keep = np.asarray(sub.sum(axis=1)).ravel() >= 2
        sub = sub[keep]

        t = time.perf_counter()
        # Raise the guard so EASE actually runs at each K (the guard exists to prevent
        # an accidental OOM, not to stop this deliberate measurement).
        EASE(reg=500.0, max_gb=64.0).fit(sub)
        ease_s = time.perf_counter() - t

        t = time.perf_counter()
        ALS(**als_params).fit(sub)
        als_s = time.perf_counter() - t

        rows.append({
            "n_items": k,
            "ease_fit_s": ease_s,
            "als_fit_s": als_s,
            "ease_matrix_gb": estimate_memory_gb(k),
            "ease_vs_als_ratio": ease_s / als_s if als_s > 0 else float("nan"),
        })
    return pd.DataFrame(rows)


def extrapolate_ease_memory(catalogue_sizes=(20_000, 40_000, 100_000, 235_061)) -> pd.DataFrame:
    """The arithmetic half: EASE's dense item×item footprint at catalogue sizes this
    dataset cannot reach. Stated as arithmetic, not measured as a claim."""
    return pd.DataFrame([
        {"n_items": n, "ease_matrix_gb": estimate_memory_gb(n),
         "ease_peak_gb_approx": 2 * estimate_memory_gb(n),  # matrix + its inverse
         "feasible_on_32gb_ram": 2 * estimate_memory_gb(n) < 32}
        for n in catalogue_sizes
    ])
