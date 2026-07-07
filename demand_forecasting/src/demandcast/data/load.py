"""Load the hierarchical pasta sales dataset into a tidy long frame.

The raw CSV is wide: one row per day, one ``QTY_B{brand}_{item}`` and one
``PROMO_B{brand}_{item}`` column per SKU. Everything downstream (backtesting,
models, the promo-lift analysis) works on the long format produced here:
one row per (date, sku) with ``qty`` and ``promo`` columns.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

_RAW_REL = Path("data") / "raw" / "hierarchical_sales_data.csv"

QTY_PREFIX = "QTY_"
PROMO_PREFIX = "PROMO_"


def default_raw_path() -> Path:
    """Locate the bundled CSV.

    Resolution order: the ``DEMANDCAST_DATA`` env var, then the working
    directory (the Docker image's WORKDIR), then the repository layout
    relative to this file (editable/dev installs). ``__file__`` alone is not
    enough: in a regular (non-editable) install this module lives in
    site-packages, nowhere near the data.
    """
    if env := os.environ.get("DEMANDCAST_DATA"):
        return Path(env)
    candidates = [
        Path.cwd() / _RAW_REL,
        Path(__file__).parents[3] / _RAW_REL,  # data -> demandcast -> src -> root
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "Could not locate hierarchical_sales_data.csv. Tried: "
        + ", ".join(str(c) for c in candidates)
        + ". Run from the project root or set DEMANDCAST_DATA."
    )


def load_raw(path: str | Path | None = None) -> pd.DataFrame:
    """Read the wide CSV with ``DATE`` parsed as datetime."""
    wide = pd.read_csv(path if path is not None else default_raw_path(),
                       parse_dates=["DATE"])
    return wide.sort_values("DATE").reset_index(drop=True)


def to_long(wide: pd.DataFrame, validate: bool = True) -> pd.DataFrame:
    """Melt the wide frame to one row per (date, sku).

    Returns columns: ``date``, ``sku`` (e.g. ``B1_1``), ``brand`` (e.g. ``B1``),
    ``qty``, and ``promo`` (int8, 0/1). Dates form a *trading-day* grid: the 27
    calendar gaps are Italian public holidays (store closures) — see
    DATA_NOTES.md for why they are deliberately not zero-filled.
    """
    qty_cols = [c for c in wide.columns if c.startswith(QTY_PREFIX)]
    promo_cols = [c for c in wide.columns if c.startswith(PROMO_PREFIX)]

    if validate:
        _validate_wide(wide, qty_cols, promo_cols)

    qty = wide.melt(
        id_vars="DATE", value_vars=qty_cols, var_name="sku", value_name="qty"
    )
    qty["sku"] = qty["sku"].str.removeprefix(QTY_PREFIX)

    promo = wide.melt(
        id_vars="DATE", value_vars=promo_cols, var_name="sku", value_name="promo"
    )
    promo["sku"] = promo["sku"].str.removeprefix(PROMO_PREFIX)

    long = qty.merge(promo, on=["DATE", "sku"], how="left")
    long = long.rename(columns={"DATE": "date"})
    long["brand"] = long["sku"].str.split("_").str[0]
    long["promo"] = long["promo"].astype("int8")
    long = long.sort_values(["sku", "date"]).reset_index(drop=True)
    return long[["date", "sku", "brand", "qty", "promo"]]


def load_long(path: str | Path | None = None, validate: bool = True) -> pd.DataFrame:
    """Convenience: :func:`load_raw` then :func:`to_long`."""
    return to_long(load_raw(path), validate=validate)


def _validate_wide(
    wide: pd.DataFrame, qty_cols: list[str], promo_cols: list[str]
) -> None:
    """Fail fast if the raw file is not what the rest of the code assumes."""
    qty_skus = {c.removeprefix(QTY_PREFIX) for c in qty_cols}
    promo_skus = {c.removeprefix(PROMO_PREFIX) for c in promo_cols}
    if qty_skus != promo_skus:
        raise ValueError(
            "QTY/PROMO columns are not paired: "
            f"{sorted(qty_skus ^ promo_skus)[:5]} ..."
        )

    promo_values = set(pd.unique(wide[promo_cols].to_numpy().ravel()))
    if not promo_values <= {0, 1}:
        raise ValueError(f"PROMO columns contain non-binary values: {promo_values - {0, 1}}")

    if wide["DATE"].duplicated().any():
        raise ValueError("Duplicate dates in raw file")

    if (wide[qty_cols].to_numpy() < 0).any():
        raise ValueError("Negative quantities in raw file")
