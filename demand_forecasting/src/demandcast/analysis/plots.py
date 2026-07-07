"""Forecast visualization: actuals vs 28-day-ahead forecast with P10-P90 band.

One figure, one story: what the planner sees for a single SKU over one
held-out window — point forecast, calibrated uncertainty band, and the promo
days that drive the demand spikes.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

INK = "#1f2937"      # actuals: dark neutral ink
BLUE = "#2563eb"     # forecast + interval band: one hue, two intensities
AMBER = "#f59e0b"    # promo-day background spans (recessive, low alpha)
GRID = "#e5e7eb"


def pick_example_sku(preds_fold: pd.DataFrame, long: pd.DataFrame) -> str:
    """Deterministic default: highest-volume non-B2 SKU with >=3 promo days
    in the fold window (B2 is the perma-promo brand — shading every day
    tells no story)."""
    window_dates = preds_fold["date"].unique()
    window = long[long["date"].isin(window_dates)]
    promo_days = window.groupby("sku")["promo"].sum()
    stats = long.groupby("sku").agg(mean_qty=("qty", "mean"), brand=("brand", "first"))
    eligible = stats[(stats["brand"] != "B2") & (promo_days.reindex(stats.index).fillna(0) >= 3)]
    return eligible["mean_qty"].idxmax()


def plot_forecast(
    preds: pd.DataFrame,
    long: pd.DataFrame,
    sku: str | None = None,
    fold: int | None = None,
    out: Path = Path("outputs/forecast_example.png"),
) -> Path:
    fold = fold if fold is not None else int(preds["fold"].max())
    f = preds[preds["fold"] == fold].copy()
    f["date"] = pd.to_datetime(f["date"])
    sku = sku or pick_example_sku(f, long)

    p = f[f["sku"] == sku].sort_values("date")
    if p.empty:
        raise ValueError(f"no predictions for sku={sku!r} in fold {fold}")
    p = p.merge(long[long["sku"] == sku][["date", "promo"]], on="date", how="left")

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)

    for d in p.loc[p["promo"] == 1, "date"]:
        ax.axvspan(
            d - pd.Timedelta(hours=12), d + pd.Timedelta(hours=12),
            color=AMBER, alpha=0.12, lw=0,
        )

    has_band = {"y_q10", "y_q90"} <= set(p.columns)
    if has_band:
        ax.fill_between(
            p["date"], p["y_q10"], p["y_q90"],
            color=BLUE, alpha=0.18, lw=0, label="P10–P90 interval",
        )
    ax.plot(p["date"], p["y_true"], color=INK, lw=1.6, marker="o", ms=3.5, label="actual")
    ax.plot(p["date"], p["y_pred"], color=BLUE, lw=2, label="forecast")

    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor=AMBER, alpha=0.25, label="promo day"))
    ax.legend(handles=handles, frameon=False, loc="upper left", fontsize=9)

    ax.set_title(
        f"{sku}: 28-trading-day-ahead forecast (backtest fold {fold})",
        fontsize=11, color=INK, loc="left",
    )
    ax.set_ylabel("units sold", fontsize=9, color=INK)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color=GRID, lw=0.8)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=8.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out
