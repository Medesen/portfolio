"""Calendar features for daily Italian retail sales.

The dataset ships no covariates beyond the promo flag, so holiday and
seasonality information is reconstructed here. The retailer is Italian
(Mancuso et al., 2021), hence the Italian national holiday calendar.
"""

from __future__ import annotations

import holidays
import pandas as pd


def italian_holidays(years: range) -> set[pd.Timestamp]:
    """Italian national holidays for the given years, as normalized Timestamps."""
    return {pd.Timestamp(d) for d in holidays.Italy(years=list(years)).keys()}


def calendar_features(dates: pd.Series) -> pd.DataFrame:
    """Deterministic calendar features for a series of dates.

    These are known arbitrarily far into the future, so they are safe to use
    at any forecast horizon (no leakage).
    """
    dates = pd.to_datetime(dates)
    years = range(dates.dt.year.min(), dates.dt.year.max() + 1)
    hols = italian_holidays(years)

    out = pd.DataFrame(index=dates.index)
    out["dayofweek"] = dates.dt.dayofweek.astype("int8")
    out["month"] = dates.dt.month.astype("int8")
    out["dayofmonth"] = dates.dt.day.astype("int8")
    out["weekofyear"] = dates.dt.isocalendar().week.astype("int16")
    out["is_weekend"] = (out["dayofweek"] >= 5).astype("int8")
    out["is_holiday"] = dates.isin(hols).astype("int8")
    # Shopping spikes around holidays show up the day before, not on the day.
    out["is_holiday_eve"] = (dates + pd.Timedelta(days=1)).isin(hols).astype("int8")
    return out
