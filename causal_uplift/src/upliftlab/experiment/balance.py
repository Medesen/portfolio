"""Covariate balance via standardized mean differences (SMD).

The first thing to check in *any* experiment — before touching an outcome — is
whether randomization actually produced comparable groups. The standardized mean
difference expresses each covariate gap in pooled-standard-deviation units, so it
is comparable across variables on different scales. The conventional threshold is
|SMD| > 0.1 = a materially imbalanced covariate. On a correctly randomized trial
every SMD should sit well inside that band; a large one is a red flag for a
broken randomization or a data problem.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _expand(df: pd.DataFrame, numeric: list[str], categorical: list[str]) -> pd.DataFrame:
    """Numeric covariates as-is; categoricals one-hot expanded to 0/1 columns."""
    parts = [df[numeric].astype(float).reset_index(drop=True)]
    for col in categorical:
        parts.append(pd.get_dummies(df[col], prefix=col, dtype=float).reset_index(drop=True))
    return pd.concat(parts, axis=1)


def _smd(a: pd.Series, b: pd.Series) -> float:
    """SMD of covariate between treatment (a) and control (b) groups."""
    pooled_sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    if pooled_sd == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled_sd)


def standardized_mean_differences(
    df: pd.DataFrame,
    numeric: list[str],
    categorical: list[str] | None = None,
    arm_col: str = "segment",
    control="No E-Mail",
    treatments: list | None = None,
) -> pd.DataFrame:
    """Table of |SMD| for every covariate against control, one column per arm.

    Returns a DataFrame indexed by covariate (categoricals expanded to indicator
    levels) with a signed SMD column per treatment arm and an ``abs_max`` column.
    """
    categorical = categorical or []
    if treatments is None:
        treatments = [a for a in df[arm_col].unique() if a != control]

    expanded = _expand(df, numeric, categorical)
    expanded[arm_col] = df[arm_col].values

    ctrl = expanded[expanded[arm_col] == control]
    cols = {}
    for arm in treatments:
        t = expanded[expanded[arm_col] == arm]
        cols[str(arm)] = {c: _smd(t[c], ctrl[c]) for c in expanded.columns if c != arm_col}
    table = pd.DataFrame(cols)
    table["abs_max"] = table.abs().max(axis=1)
    return table
