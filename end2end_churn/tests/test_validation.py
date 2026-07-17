"""Unit tests for the API schema-alignment helper."""

import numpy as np
import pandas as pd
import pytest

from src.api.validation import align_schema


@pytest.mark.unit
def test_align_schema_does_not_mutate_input():
    """The caller's DataFrame must be unchanged by any alignment branch."""
    df = pd.DataFrame({"a": [1, 2], "extra": [3, 4]})
    original = df.copy(deep=True)

    with pytest.warns(UserWarning):
        aligned, info = align_schema(df, ["a", "b"])

    pd.testing.assert_frame_equal(df, original)
    assert list(aligned.columns) == ["a", "b"]
    assert info["missing_columns"] == ["b"]
    assert info["extra_columns"] == ["extra"]
    assert aligned["b"].isna().all()


@pytest.mark.unit
def test_align_schema_no_changes_passthrough():
    """Already-aligned input keeps its values and reports no changes."""
    df = pd.DataFrame({"a": [1.0], "b": [2.0]})
    aligned, info = align_schema(df, ["a", "b"])

    assert info == {"missing_columns": [], "extra_columns": [], "reordered": False}
    np.testing.assert_array_equal(aligned["a"].to_numpy(), df["a"].to_numpy())


@pytest.mark.unit
def test_align_schema_reorders_to_expected_order():
    df = pd.DataFrame({"b": [2], "a": [1]})
    aligned, info = align_schema(df, ["a", "b"])

    assert list(aligned.columns) == ["a", "b"]
    assert info["reordered"] is True
