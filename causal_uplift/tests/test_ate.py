import numpy as np

from upliftlab.data import make_synthetic_rct
from upliftlab.experiment import estimate_all, estimate_ate
from upliftlab.experiment.ate import diff_in_means, minimum_detectable_effect


def test_ate_recovers_known_effect_with_covering_ci():
    d = make_synthetic_rct(n=20000, seed=3, kind="continuous", ate=1.0, hetero=0.0).data
    est = estimate_ate(d, outcome="y", arm=1, arm_col="t", control=0)
    assert abs(est.effect - 1.0) < 0.1                 # point estimate near truth
    assert est.ci_low < 1.0 < est.ci_high              # CI covers the truth
    assert est.p_value < 1e-6                           # a real effect is detected


def test_multiplicity_adjustment_is_conservative():
    d = make_synthetic_rct(n=8000, seed=4, kind="continuous", ate=0.2).data
    table = estimate_all(d, outcomes=["y"], treatments=[1], control=0, arm_col="t")
    # adjusted p-values are never smaller than raw ones
    assert (table["p_holm"] >= table["p_value"] - 1e-12).all()
    assert (table["p_bh"] >= table["p_value"] - 1e-12).all()
    assert {"mde_80", "power"} <= set(table.columns)


def test_diff_in_means_matches_manual():
    y = np.array([1.0, 3.0, 5.0, 7.0])
    t = np.array([1, 1, 0, 0])
    effect, se = diff_in_means(y, t)
    assert effect == (2.0 - 6.0)


def test_mde_shrinks_with_n():
    small = minimum_detectable_effect(1000, 1000, sd=1.0)
    big = minimum_detectable_effect(100000, 100000, sd=1.0)
    assert big < small
