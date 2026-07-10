from upliftlab.data import make_synthetic_rct
from upliftlab.experiment import standardized_mean_differences


def _smd(confound):
    d = make_synthetic_rct(n=15000, seed=2, confound=confound).data
    return standardized_mean_differences(
        d, numeric=["x1", "x2", "x3"], categorical=[],
        arm_col="t", control=0, treatments=[1],
    )


def test_balance_passes_on_randomized():
    table = _smd(confound=0.0)
    # a correctly randomized trial: every covariate well inside |SMD| < 0.1
    assert table["abs_max"].max() < 0.1


def test_balance_flags_confounding():
    table = _smd(confound=1.5)
    # treatment assignment tilted on x1 -> x1 must be flagged as imbalanced
    assert abs(table.loc["x1", "1"]) > 0.1
    assert table["abs_max"].max() > 0.1
