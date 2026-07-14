"""Load and validate the Hillstrom e-mail experiment into a tidy analysis frame.

The raw CSV is already one row per customer. This module fails fast if the file
is not what the rest of the code assumes (arm labels, binary outcomes,
non-negative spend), normalises the one spelling quirk (``Surburban``), and
exposes the schema constants and a design-matrix builder used everywhere
downstream.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

_RAW_REL = Path("data") / "raw" / "hillstrom_email.csv"

CONTROL = "No E-Mail"
TREATMENTS = ["Mens E-Mail", "Womens E-Mail"]
ARMS = [CONTROL, *TREATMENTS]

OUTCOMES = ["visit", "conversion", "spend"]
BINARY_OUTCOMES = ["visit", "conversion"]

# Pre-treatment covariates (measured before the campaign — safe to condition on).
NUMERIC_COVARIATES = ["recency", "history", "mens", "womens", "newbie"]
CATEGORICAL_COVARIATES = ["zip_code", "channel"]
# ``history_segment`` is a redundant bucketing of ``history`` and is dropped from
# the feature set (kept in the raw frame for reference only).


def default_raw_path() -> Path:
    """Locate the bundled CSV.

    Resolution order: the ``UPLIFTLAB_DATA`` env var, then the working directory
    (the Docker image's WORKDIR), then the repository layout relative to this
    file (editable/dev installs). ``__file__`` alone is not enough: in a regular
    (non-editable) install this module lives in site-packages, nowhere near the
    data.
    """
    if env := os.environ.get("UPLIFTLAB_DATA"):
        return Path(env)
    candidates = [
        Path.cwd() / _RAW_REL,
        Path(__file__).parents[3] / _RAW_REL,  # data -> upliftlab -> src -> root
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "Could not locate hillstrom_email.csv. Tried: "
        + ", ".join(str(c) for c in candidates)
        + ". Run from the project root or set UPLIFTLAB_DATA."
    )


def load(path: str | Path | None = None, validate: bool = True) -> pd.DataFrame:
    """Read the experiment CSV, validate it, and normalise the ``zip_code`` typo.

    Returns the raw columns unchanged except ``zip_code`` (``Surburban`` ->
    ``Suburban``). No treatment/outcome columns are altered.
    """
    df = pd.read_csv(path if path is not None else default_raw_path())
    df["zip_code"] = df["zip_code"].replace({"Surburban": "Suburban"})
    if validate:
        _validate(df)
    return df.reset_index(drop=True)


def two_arm(df: pd.DataFrame, treatment: str) -> pd.DataFrame:
    """Restrict to control + one treatment arm and add a 0/1 ``t`` column.

    Used by the uplift models, which estimate a single binary-treatment CATE.
    """
    if treatment not in TREATMENTS:
        raise ValueError(f"treatment must be one of {TREATMENTS}, got {treatment!r}")
    sub = df[df["segment"].isin([CONTROL, treatment])].copy()
    sub["t"] = (sub["segment"] == treatment).astype(int)
    return sub.reset_index(drop=True)


def design_matrix(
    df: pd.DataFrame,
    numeric: list[str] | None = None,
    categorical: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Build a numeric feature matrix (one-hot categoricals) for the learners.

    Returns ``(X, feature_names)``. One-hot columns are derived from the
    categories present in ``df`` (plain ``pd.get_dummies``), so encode the
    FULL dataset first and split by row index afterwards — encoding train and
    test separately can produce misaligned matrices if a level is absent from
    one side. Every caller in this package follows that encode-then-split
    order.
    """
    numeric = numeric if numeric is not None else NUMERIC_COVARIATES
    categorical = categorical if categorical is not None else CATEGORICAL_COVARIATES
    parts = [df[numeric].astype(float).reset_index(drop=True)]
    for col in categorical:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True, dtype=float)
        parts.append(dummies.reset_index(drop=True))
    X = pd.concat(parts, axis=1)
    return X, list(X.columns)


def _validate(df: pd.DataFrame) -> None:
    """Fail fast if the file is not the experiment the code assumes."""
    required = set(NUMERIC_COVARIATES) | set(CATEGORICAL_COVARIATES) | {"segment"} | set(OUTCOMES)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")

    arms = set(df["segment"].unique())
    if arms != set(ARMS):
        raise ValueError(f"Unexpected treatment arms: {sorted(arms)} (expected {ARMS})")

    for col in [*BINARY_OUTCOMES, "mens", "womens", "newbie"]:
        vals = set(pd.unique(df[col]))
        if not vals <= {0, 1}:
            raise ValueError(f"Column {col!r} is not binary: extra values {vals - {0, 1}}")

    if (df["spend"] < 0).any():
        raise ValueError("Negative spend in raw file")

    # A purchase (conversion) should never coincide with zero spend, and every
    # conversion should be a visit — cheap integrity checks on the outcome funnel.
    if ((df["conversion"] == 1) & (df["spend"] <= 0)).any():
        raise ValueError("Found conversions with non-positive spend")
    if ((df["conversion"] == 1) & (df["visit"] == 0)).any():
        raise ValueError("Found conversions without a visit")
