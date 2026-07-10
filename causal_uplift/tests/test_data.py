import pandas as pd
import pytest

from upliftlab.data import (
    ARMS,
    NUMERIC_COVARIATES,
    OUTCOMES,
    design_matrix,
    load,
    make_synthetic_rct,
    two_arm,
)


def test_load_and_validate():
    df = load()
    assert len(df) == 64000
    assert set(df["segment"].unique()) == set(ARMS)
    assert set(OUTCOMES) <= set(df.columns)
    # zip_code typo is normalised
    assert "Surburban" not in set(df["zip_code"].unique())
    assert "Suburban" in set(df["zip_code"].unique())


def test_validate_rejects_broken_binary():
    df = load().copy()
    df.loc[0, "visit"] = 2
    with pytest.raises(ValueError):
        # re-run validation via the loader's internal check
        from upliftlab.data.load import _validate
        _validate(df)


def test_two_arm_and_design_matrix():
    df = load()
    sub = two_arm(df, "Womens E-Mail")
    assert set(sub["segment"].unique()) == {"No E-Mail", "Womens E-Mail"}
    assert set(sub["t"].unique()) == {0, 1}
    X, names = design_matrix(sub)
    assert len(X) == len(sub)
    assert all(c in names for c in NUMERIC_COVARIATES)  # numeric passthrough
    assert any(c.startswith("zip_code_") for c in names)  # one-hot expansion


def test_synthetic_is_randomized_and_recovers_truth():
    s = make_synthetic_rct(n=20000, seed=1, kind="continuous", ate=1.0, hetero=0.0)
    d = s.data
    # randomized: treatment share ~ 0.5
    assert 0.48 < d["t"].mean() < 0.52
    # empirical difference in means ~ true ATE
    emp = d.loc[d.t == 1, "y"].mean() - d.loc[d.t == 0, "y"].mean()
    assert emp == pytest.approx(s.ate, abs=0.1)
