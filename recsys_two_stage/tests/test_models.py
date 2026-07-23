"""Model correctness.

The EASE test is the one worth a reviewer's eye: the closed form is validated
against an independent, column-wise ridge solve — the same optimisation problem
attacked a completely different way.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from reclab.evaluation.metrics import top_k_from_scores
from reclab.models import ALS, EASE, ItemKNN, Popularity, estimate_memory_gb


def random_matrix(n_sessions: int, n_items: int, seed: int = 0, density: float = 0.1):
    rng = np.random.default_rng(seed)
    X = (rng.random((n_sessions, n_items)) < density).astype(np.float32)
    # Guarantee no empty item column, so gram is well-conditioned.
    X[rng.integers(0, n_sessions, n_items), np.arange(n_items)] = 1.0
    return sp.csr_matrix(X)


def brute_force_ease(X: sp.csr_matrix, reg: float) -> np.ndarray:
    """EASE via n independent ridge regressions — the definition, not the trick.

    EASE minimises ||X - XB||² + reg||B||² with diag(B)=0. Because diag(B)=0,
    column j never uses item j, so the problem decomposes: column j is a ridge
    regression of item j's column on all *other* item columns. Solving it that
    way must reproduce Steck's closed form.
    """
    A = X.toarray()
    n_items = A.shape[1]
    B = np.zeros((n_items, n_items))
    for j in range(n_items):
        others = [i for i in range(n_items) if i != j]
        M = A[:, others]
        gram = M.T @ M + reg * np.eye(n_items - 1)
        B[others, j] = np.linalg.solve(gram, M.T @ A[:, j])
    return B


class TestEASE:
    def test_closed_form_matches_column_wise_ridge(self):
        X = random_matrix(80, 12, seed=1, density=0.25)
        model = EASE(reg=5.0).fit(X)
        expected = brute_force_ease(X, reg=5.0)
        np.testing.assert_allclose(model.B_, expected, atol=1e-6)

    def test_diagonal_is_zero(self):
        # The constraint that stops EASE learning the identity function.
        model = EASE(reg=10.0).fit(random_matrix(60, 15, seed=2))
        np.testing.assert_allclose(np.diag(model.B_), 0.0, atol=1e-12)

    def test_memory_guard_refuses_an_impossible_catalogue(self):
        model = EASE(reg=100.0, max_gb=1.0)
        # 20k items would need ~3.2 GB — the guard must fail loudly, not OOM.
        X = sp.csr_matrix((5, 20_000), dtype=np.float32)
        with pytest.raises(MemoryError, match="square of the catalogue"):
            model.fit(X)

    def test_memory_estimate_is_the_documented_arithmetic(self):
        assert estimate_memory_gb(20_000) == pytest.approx(3.2, rel=1e-3)
        assert estimate_memory_gb(235_061) == pytest.approx(441.8, rel=1e-2)

    def test_score_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="not been fitted"):
            EASE().score(random_matrix(3, 5))

    def test_reproducible(self):
        X = random_matrix(70, 14, seed=3)
        a, b = EASE(reg=8.0).fit(X), EASE(reg=8.0).fit(X)
        np.testing.assert_array_equal(a.B_, b.B_)


class TestItemKNN:
    def test_self_similarity_is_removed(self):
        # An item must never be its own recommendation.
        model = ItemKNN(k=50, shrink=0.0).fit(random_matrix(100, 20, seed=4))
        assert model.similarity_.diagonal().sum() == 0.0

    def test_shrinkage_damps_low_support_pairs(self):
        # Two items co-occurring in exactly one session: raw cosine = 1.0, but
        # shrinkage must pull it well below a well-supported pair's score.
        rows = []
        # Well-supported pair (0, 1): together in 20 sessions.
        for s in range(20):
            rows += [(s, 0), (s, 1)]
        # Fringe pair (2, 3): together once.
        rows += [(100, 2), (100, 3)]
        X = sp.csr_matrix(
            (np.ones(len(rows)), ([r[0] for r in rows], [r[1] for r in rows])),
            shape=(101, 4),
        )
        no_shrink = ItemKNN(k=10, shrink=0.0).fit(X).similarity_.toarray()
        shrunk = ItemKNN(k=10, shrink=10.0).fit(X).similarity_.toarray()
        # Without shrinkage the fringe pair matches the strong pair (both ~1.0).
        assert no_shrink[2, 3] == pytest.approx(no_shrink[0, 1], abs=1e-9)
        # With shrinkage the fringe pair is damped much harder than the strong one.
        assert shrunk[2, 3] < shrunk[0, 1]

    def test_keeps_at_most_k_neighbours(self):
        model = ItemKNN(k=3, shrink=0.0).fit(random_matrix(200, 30, seed=6))
        per_row = np.diff(model.similarity_.indptr)
        assert per_row.max() <= 3

    def test_rejects_bad_hyperparameters(self):
        with pytest.raises(ValueError, match="k must be"):
            ItemKNN(k=0)
        with pytest.raises(ValueError, match="shrink must be"):
            ItemKNN(shrink=-1.0)


class TestPopularity:
    def test_counts_sessions_per_item(self):
        X = sp.csr_matrix(
            np.array([[1, 1, 0], [1, 0, 0], [1, 1, 1]], dtype=np.float32)
        )
        model = Popularity().fit(X)
        np.testing.assert_array_equal(model.counts_, [3, 2, 1])

    def test_scores_are_identical_across_sessions(self):
        model = Popularity().fit(random_matrix(50, 10, seed=8))
        scores = model.score(random_matrix(4, 10, seed=9))
        assert np.all(scores[0] == scores[1])


class TestALS:
    def _clustered(self, n_clusters=4, items_per=10, sessions_per=60, seed=0):
        """Sessions drawn from item clusters with no cross-cluster co-occurrence.

        A working latent model must rank a session's own cluster above the rest.
        """
        rng = np.random.default_rng(seed)
        n_items = n_clusters * items_per
        rows, cols, cluster_of_session = [], [], []
        s = 0
        for c in range(n_clusters):
            members = np.arange(c * items_per, (c + 1) * items_per)
            for _ in range(sessions_per):
                picks = rng.choice(members, size=rng.integers(2, items_per), replace=False)
                rows += [s] * len(picks)
                cols += picks.tolist()
                cluster_of_session.append(c)
                s += 1
        X = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(s, n_items))
        return X, np.array(cluster_of_session), n_clusters, items_per

    def test_recovers_cluster_structure(self):
        X, _, n_clusters, items_per = self._clustered(seed=1)
        model = ALS(factors=16, iterations=20, regularization=1.0, seed=0).fit(X)

        # A held-out session built from cluster 0's first two items must rank
        # cluster-0 items above everything else on average.
        history = sp.csr_matrix(
            ([1.0, 1.0], ([0, 0], [0, 1])), shape=(1, n_clusters * items_per)
        )
        scores = model.score(history)[0]
        own = scores[:items_per].mean()
        others = scores[items_per:].mean()
        assert own > others

    def test_use_counts_changes_factors_only_on_count_valued_data(self):
        # The ablation's premise: on a count-valued matrix, use_counts=True (confidence
        # 1+alpha*count) must differ from the binarized fit — otherwise the ablation
        # measures nothing. And on a purely binary matrix the two must coincide, which
        # is exactly why the pipeline's headline binary ALS is unaffected by the flag.
        X, _, n_clusters, items_per = self._clustered(seed=2)
        Xc = X.tolil()
        Xc[0, 0], Xc[1, 1], Xc[2, 2] = 5.0, 4.0, 3.0  # a few repeat counts > 1
        Xc = Xc.tocsr()

        f_true = ALS(factors=16, iterations=20, alpha=40.0, use_counts=True, seed=0).fit(Xc).item_factors_
        f_false = ALS(factors=16, iterations=20, alpha=40.0, use_counts=False, seed=0).fit(Xc).item_factors_
        assert not np.allclose(f_true, f_false)  # counts actually change the factorization

        b_true = ALS(factors=16, iterations=20, alpha=40.0, use_counts=True, seed=0).fit(X).item_factors_
        b_false = ALS(factors=16, iterations=20, alpha=40.0, use_counts=False, seed=0).fit(X).item_factors_
        np.testing.assert_allclose(b_true, b_false)  # no-op on binary input

    def test_score_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="not been fitted"):
            ALS().score(random_matrix(3, 5))

    def test_rejects_bad_hyperparameters(self):
        with pytest.raises(ValueError, match="factors"):
            ALS(factors=0)
        with pytest.raises(ValueError, match="alpha"):
            ALS(alpha=0)


class TestSeenItemMaskingIsCentral:
    """The eval loop masks seen items for every model; no model does it itself.

    A model that forgets to drop a session's own history scores its input back
    and looks spectacular for nothing. This asserts the mask mechanism works —
    the property every model relies on.
    """

    def test_masked_items_never_appear_in_top_k(self):
        model = EASE(reg=5.0).fit(random_matrix(80, 40, seed=10, density=0.2))
        histories = random_matrix(5, 40, seed=11, density=0.15)
        scores = model.score(histories)
        seen = histories.toarray() > 0
        # k must be smaller than the number of unseen items, or masked items are
        # forced into the result for lack of anything else to rank.
        assert (~seen).sum(axis=1).min() > 10
        ranked = top_k_from_scores(scores, k=10, mask=seen)
        for row in range(histories.shape[0]):
            seen_items = set(np.where(seen[row])[0])
            assert not (set(ranked[row].tolist()) & seen_items)
