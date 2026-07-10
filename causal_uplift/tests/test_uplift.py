import numpy as np
from scipy.stats import spearmanr

from upliftlab.data import make_synthetic_rct
from upliftlab.uplift import (
    LEARNERS,
    incremental_curve,
    qini_coefficient,
    qini_curve,
    uplift_by_group,
)

FAST = dict(n_estimators=120, learning_rate=0.1, num_leaves=15,
            min_child_samples=40, random_state=0, n_jobs=1, verbosity=-1)


def _split():
    s = make_synthetic_rct(n=12000, seed=7, kind="binary", ate=0.5, hetero=1.5)
    d = s.data
    X = d[["x1", "x2", "x3"]].to_numpy()
    n = len(d)
    rng = np.random.default_rng(0)
    mask = rng.random(n) < 0.7
    return d, X, mask


def test_qini_curve_endpoints():
    d, X, _ = _split()
    x, q = qini_curve(d["x1"].to_numpy(), d["y"].to_numpy(), d["t"].to_numpy())
    assert x[0] == 0.0 and q[0] == 0.0
    assert x[-1] == 1.0
    assert np.isfinite(q).all()


def test_xlearner_recovers_heterogeneity_and_beats_random():
    d, X, mask = _split()
    t, y = d["t"].to_numpy(), d["y"].to_numpy()
    model = LEARNERS["x-learner"](kind="binary", params=FAST).fit(X[mask], t[mask], y[mask])
    pred = model.predict_uplift(X[~mask])
    cate_true = d["cate_true"].to_numpy()[~mask]
    # ranking correlates with the true individual effect
    rho, _ = spearmanr(pred, cate_true)
    assert rho > 0.3
    # Qini beats random targeting on held-out data
    qc = qini_coefficient(pred, y[~mask], t[~mask])
    assert qc["qini"] > 0
    # mean predicted uplift is in the neighbourhood of the true ATE (recovery)
    assert abs(pred.mean() - cate_true.mean()) < 0.1


def test_uplift_by_group_monotone_ish():
    d, X, mask = _split()
    t, y = d["t"].to_numpy(), d["y"].to_numpy()
    model = LEARNERS["t-learner"](kind="binary", params=FAST).fit(X[mask], t[mask], y[mask])
    pred = model.predict_uplift(X[~mask])
    deciles = uplift_by_group(pred, y[~mask], t[~mask], n_groups=5)
    # top group's observed uplift exceeds the bottom group's
    assert deciles.iloc[0]["obs_uplift"] > deciles.iloc[-1]["obs_uplift"]


def test_incremental_curve_shapes():
    d, X, mask = _split()
    t, y = d["t"].to_numpy(), d["y"].to_numpy()
    curve = incremental_curve(d["x1"].to_numpy(), y, t, ks=[0.1, 0.5, 1.0])
    assert list(curve["k"]) == [0.1, 0.5, 1.0]
    assert (curve["n_targeted"] > 0).all()
