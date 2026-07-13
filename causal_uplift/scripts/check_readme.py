#!/usr/bin/env python3
"""Check that README headline numbers match the generated outputs/ tables.

Run after ``make reproduce``:

    make check-readme        (or: python3 scripts/check_readme.py)

Every headline claim in the README that has a counterpart in ``outputs/``
is parsed out of the markdown and compared against the CSV value within
rounding tolerance. Exits non-zero listing every mismatch — so
"reproducible" is *checked*, not merely asserted. Stdlib only: runs on any
Python 3.10+, no Docker or project dependencies needed.
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
OUTPUTS = ROOT / "outputs"

failures: list[str] = []
passes = 0


def check(label: str, readme_value: float, actual: float, decimals: int) -> None:
    """README states values rounded to `decimals`; allow half-ulp slack."""
    global passes
    tolerance = 0.5 * 10 ** -decimals + 1e-9
    if abs(readme_value - actual) <= tolerance:
        passes += 1
    else:
        failures.append(
            f"  MISMATCH {label}: README says {readme_value}, outputs say {actual:.{decimals + 2}f}"
        )


def read_csv(name: str) -> list[dict]:
    path = OUTPUTS / name
    if not path.exists():
        sys.exit(f"Missing {path} — run `make reproduce` first.")
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    text = README.read_text()
    ate = {(r["arm"], r["outcome"]): r for r in read_csv("ate.csv")}
    targeting = {round(float(r["k"]), 2): r for r in read_csv("uplift_targeting.csv")}
    qini = {r["learner"]: r for r in read_csv("uplift_qini.csv")}

    # --- ATE table: effect and CI per arm x outcome -------------------------
    for arm_readme, arm_csv in [("Mens", "Mens E-Mail"), ("Womens", "Womens E-Mail")]:
        for outcome in ["visit", "conversion", "spend"]:
            row = ate[(arm_csv, outcome)]
            m = re.search(rf"^\|\s*{arm_readme}\s*\|\s*{outcome}\s*\|(.+)$", text, re.M)
            if not m:
                failures.append(f"  MISSING README row: {arm_readme} / {outcome}")
                continue
            cells = [c.strip().strip("*") for c in m.group(1).strip("|").split("|")]
            effect_cell, ci_cell = cells[2], cells[4]
            ci = re.search(r"\[([+-][\d.]+),\s*([+-][\d.]+)\]", ci_cell)
            if outcome == "spend":  # dollars, 3 decimals
                check(f"{arm_readme}/{outcome} effect ($)",
                      float(effect_cell.replace("+$", "")), float(row["effect"]), 3)
                check(f"{arm_readme}/{outcome} ci_low", float(ci.group(1)), float(row["ci_low"]), 3)
                check(f"{arm_readme}/{outcome} ci_high", float(ci.group(2)), float(row["ci_high"]), 3)
            else:  # percentage points, 2 decimals
                check(f"{arm_readme}/{outcome} effect (pp)",
                      float(effect_cell.replace("pp", "")), float(row["effect"]) * 100, 2)
                check(f"{arm_readme}/{outcome} ci_low", float(ci.group(1)), float(row["ci_low"]) * 100, 2)
                check(f"{arm_readme}/{outcome} ci_high", float(ci.group(2)), float(row["ci_high"]) * 100, 2)

    # --- MDE ----------------------------------------------------------------
    m = re.search(r"as small as \*\*([\d.]+)pp\*\*", text)
    check("MDE (visit, 80% power)", float(m.group(1)),
          float(ate[("Mens E-Mail", "visit")]["mde_80"]) * 100, 2)

    # --- Revenue lens: ratio of ATEs and top-10% value ----------------------
    w_visit = float(ate[("Womens E-Mail", "visit")]["effect"])
    w_spend = float(ate[("Womens E-Mail", "spend")]["effect"])
    ratio = w_spend / w_visit
    m = re.search(r"~\*\*\$([\d.]+) of incremental spend", text)
    check("incremental spend per incremental visit ($)", float(m.group(1)), ratio, 2)
    m = re.search(r"worth ~\*\*\$([\d,]+) per 1,000\s+mailed", text)
    top10 = float(targeting[0.1]["uplift_visits/1k"])
    check("top-10% value per 1,000 mailed ($)",
          float(m.group(1).replace(",", "")), top10 * ratio, 0)

    # --- Targeting table ----------------------------------------------------
    for pct, k in [(10, 0.1), (20, 0.2), (30, 0.3), (50, 0.5)]:
        row = targeting[k]
        m = re.search(rf"^\|\s*top {pct}%\s*\|(.+)$", text, re.M)
        cells = [c.strip().strip("*") for c in m.group(1).strip("|").split("|")]
        check(f"top {pct}% by uplift (visits/1k)", float(cells[0]),
              float(row["uplift_visits/1k"]), 1)
        check(f"top {pct}% by response model", float(cells[1]),
              float(row["response_visits/1k"]), 1)
        check(f"top {pct}% random", float(cells[2]), float(row["random_visits/1k"]), 1)
        check(f"top {pct}% capture share (%)", float(cells[3].rstrip("%")),
              float(row["uplift_capture_%"]), 0)

    # --- Qini normalized coefficients (S vs X prose) ------------------------
    m = re.search(r"normalized Qini ([\d.]+) vs ([\d.]+)", text)
    check("S-learner normalized Qini", float(m.group(1)), float(qini["S-learner"]["qini_norm"]), 2)
    check("X-learner normalized Qini", float(m.group(2)), float(qini["X-learner"]["qini_norm"]), 2)

    # --- Report -------------------------------------------------------------
    if failures:
        print(f"README check: {passes} ok, {len(failures)} MISMATCHED:")
        print("\n".join(failures))
        sys.exit(1)
    print(f"README check: all {passes} headline numbers match outputs/ ✓")


if __name__ == "__main__":
    main()
