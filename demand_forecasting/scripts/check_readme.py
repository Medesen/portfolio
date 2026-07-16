#!/usr/bin/env python3
"""Check README/DATA_NOTES headline numbers against the generated outputs/.

Run after ``make reproduce``:

    make check-readme        (or: python3 scripts/check_readme.py)

Every headline claim that has a counterpart CSV in ``outputs/`` is parsed
out of the markdown and compared within rounding tolerance. Exits non-zero
listing every mismatch — so "reproducible" is *checked*, not merely
asserted. Stdlib only: runs on any Python 3.10+, no Docker or project
dependencies needed. (The overall PPML lift and the P10-P90 coverage are
printed by the CLI but not persisted to CSV, so they are not checked here.)
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"

failures: list[str] = []
passes = 0


def check(label: str, readme_value: float, actual: float, decimals: int) -> None:
    """Markdown states values rounded to `decimals`; allow half-ulp slack."""
    global passes
    tolerance = 0.5 * 10 ** -decimals + 1e-9
    if abs(readme_value - actual) <= tolerance:
        passes += 1
    else:
        failures.append(
            f"  MISMATCH {label}: doc says {readme_value}, outputs say {actual:.{decimals + 2}f}"
        )


def scores(name: str) -> dict[str, dict]:
    """Read a backtest scores CSV as {segment: row}."""
    path = OUTPUTS / f"backtest_{name}_scores.csv"
    if not path.exists():
        sys.exit(f"Missing {path} — run `make reproduce` first.")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    seg_col = list(rows[0].keys())[0]
    return {r[seg_col]: r for r in rows}


def row_cells(text: str, anchor: str) -> list[str]:
    """Cells after the label in the markdown table row starting with `anchor`."""
    m = re.search(rf"^\s*\|\s*(?:\*\*)?{re.escape(anchor)}(?:\*\*)?\s*\|(.+)$", text, re.M)
    if m is None:
        failures.append(f"  MISSING table row: {anchor}")
        return []
    return [c.strip().strip("*") for c in m.group(1).strip().strip("|").split("|")]


def check_score_row(text: str, anchor: str, tag: str, extra_low_mase: bool = False) -> None:
    cells = row_cells(text, anchor)
    if not cells:
        return
    s = scores(tag)
    check(f"{anchor}: MASE", float(cells[0]), float(s["overall"]["mase"]), 3)
    check(f"{anchor}: WAPE", float(cells[1]), float(s["overall"]["wape"]), 3)
    check(f"{anchor}: RMSE", float(cells[2]), float(s["overall"]["rmse"]), 2)
    if extra_low_mase:
        check(f"{anchor}: low-tercile MASE", float(cells[3]), float(s["low"]["mase"]), 2)


def main() -> None:
    readme = (ROOT / "README.md").read_text()
    notes = (ROOT / "DATA_NOTES.md").read_text()

    # --- README: all-118-SKU headline table ---------------------------------
    check_score_row(readme, "naive (last value)", "naive")
    check_score_row(readme, "seasonal-naive (m=7)", "seasonal_naive")
    check_score_row(readme, "global LightGBM (Tweedie)", "lgbm")

    # Volume-tercile prose: "(low 0.64 / mid 0.66 / high 0.66)"
    m = re.search(r"low ([\d.]+) / mid ([\d.]+) / high ([\d.]+)", readme)
    if m is None:
        failures.append("  MISSING tercile prose: 'low … / mid … / high …'")
    else:
        lgbm = scores("lgbm")
        for group, value in zip(["low", "mid", "high"], m.groups()):
            check(f"LightGBM MASE tercile ({group})", float(value), float(lgbm[group]["mase"]), 2)

    # --- README: 8-SKU subset comparison table ------------------------------
    check_score_row(readme, "SARIMAX (promo + holiday exog)", "sarimax_subset-sarimax")
    check_score_row(readme, "LightGBM (trained on the 8-SKU subset)", "lgbm_subset-sarimax")
    check_score_row(readme, "LightGBM (global training, evaluated on the subset)",
                    "lgbm_subset-sarimax_train-global")
    check_score_row(readme, "seasonal-naive", "seasonal_naive_subset-sarimax")

    # --- README: promotion lift by brand -------------------------------------
    path = OUTPUTS / "promo_lift_by_brand.csv"
    if not path.exists():
        sys.exit(f"Missing {path} — run `make reproduce` first.")
    with open(path, newline="") as f:
        brands = {r["brand"]: r for r in csv.DictReader(f)}
    for brand in ["B1", "B2", "B3", "B4"]:
        cells = row_cells(readme, brand)
        if not cells:
            continue
        row = brands[brand]
        ci = re.search(r"\[\+([\d.]+)%,\s*\+([\d.]+)%\]", cells[1])
        check(f"{brand} lift (%)", float(cells[0].replace("+", "").rstrip("%")),
              float(row["lift_pct"]), 0)
        check(f"{brand} ci_low (%)", float(ci.group(1)), float(row["ci_low_pct"]), 0)
        check(f"{brand} ci_high (%)", float(ci.group(2)), float(row["ci_high_pct"]), 0)
        check(f"{brand} n_skus", float(cells[2]), float(row["n_skus"]), 0)
        check(f"{brand} promo-day share", float(cells[3]), float(row["promo_share"]), 2)

    # --- DATA_NOTES §2: objective ablation table -----------------------------
    check_score_row(notes, "Tweedie (power 1.2)", "lgbm", extra_low_mase=True)
    check_score_row(notes, "Poisson", "lgbm-poisson", extra_low_mase=True)
    check_score_row(notes, "plain L2", "lgbm-l2", extra_low_mase=True)

    # --- Report --------------------------------------------------------------
    if failures:
        print(f"README/DATA_NOTES check: {passes} ok, {len(failures)} MISMATCHED:")
        print("\n".join(failures))
        sys.exit(1)
    print(f"README/DATA_NOTES check: all {passes} headline numbers match outputs/ ✓")


if __name__ == "__main__":
    main()
