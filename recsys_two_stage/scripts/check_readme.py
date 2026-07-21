#!/usr/bin/env python3
"""Check that README headline numbers match the generated outputs/ tables.

Run after ``make reproduce``:

    make check-readme        (or: python3 scripts/check_readme.py)

Every headline number in the README that has a counterpart in ``outputs/`` is
parsed out of the markdown and compared against the CSV value within rounding
tolerance. Exits non-zero listing every mismatch — so "reproducible" is *checked*,
not merely asserted. Stdlib only: runs on any Python 3.10+, no Docker or project
dependencies needed.
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text()
OUTPUTS = ROOT / "outputs"

failures: list[str] = []
passes = 0

# README label -> CSV model id
MODELS = {"ItemKNN": "itemknn", "EASE": "ease", "ALS": "als", "Popularity": "popularity"}


def check(label: str, readme_value: float, actual: float, decimals: int) -> None:
    global passes
    tol = 0.5 * 10**-decimals + 1e-9
    if abs(readme_value - actual) <= tol:
        passes += 1
    else:
        failures.append(
            f"  MISMATCH {label}: README says {readme_value}, "
            f"outputs say {actual:.{decimals + 2}f}"
        )


def read_csv(name: str) -> list[dict]:
    path = OUTPUTS / name
    if not path.exists():
        sys.exit(f"Missing {path} — run `make reproduce` first.")
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def num(s: str) -> float:
    return float(s.strip().replace("*", ""))


def main() -> None:
    full = {(r["model"], r["metric"], int(r["k"])): r for r in read_csv("metrics_full.csv")}
    beyond = {(r["model"], int(r["k"])): r for r in read_csv("metrics_beyond.csv")}
    protocols = {(r["protocol"], r["model"]): r for r in read_csv("protocol_comparison.csv")}

    # --- Full-catalogue table: NDCG@20 (+CI), HitRate@20, MRR@20 -------------
    # Rows like: | **ItemKNN** | **0.319** | [0.315, 0.324] | 0.562 | 0.248 |
    for label, model in MODELS.items():
        pattern = (
            rf"\|\s*\*{{0,2}}{label}\*{{0,2}}\s*\|\s*\*{{0,2}}([\d.]+)\*{{0,2}}\s*\|\s*"
            rf"\[([\d.]+),\s*([\d.]+)\]\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|"
        )
        m = re.search(pattern, README)
        if not m:
            failures.append(f"  NOT FOUND: full-catalogue row for {label}")
            continue
        ndcg, lo, hi, hr, mrr = (num(g) for g in m.groups())
        check(f"{model} NDCG@20", ndcg, num(full[(model, "ndcg", 20)]["value"]), 3)
        check(f"{model} NDCG@20 CI-low", lo, num(full[(model, "ndcg", 20)]["ci_low"]), 3)
        check(f"{model} NDCG@20 CI-high", hi, num(full[(model, "ndcg", 20)]["ci_high"]), 3)
        check(f"{model} HR@20", hr, num(full[(model, "hit_rate", 20)]["value"]), 3)
        check(f"{model} MRR@20", mrr, num(full[(model, "mrr", 20)]["value"]), 3)

    # --- Beyond-accuracy table: coverage% / gini / pop-percentile at k=20 ----
    # Rows like: | ItemKNN | 98.7% | 0.53 | 0.65 |
    for label, model in MODELS.items():
        pattern = (
            rf"\|\s*{label}\s*\|\s*([\d.]+)%\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|"
        )
        m = re.search(pattern, README)
        if not m:
            failures.append(f"  NOT FOUND: beyond-accuracy row for {label}")
            continue
        cov, gini, pop = (num(g) for g in m.groups())
        check(f"{model} coverage@20", cov, num(beyond[(model, 20)]["coverage"]) * 100, 1)
        check(f"{model} gini@20", gini, num(beyond[(model, 20)]["gini"]), 2)
        check(f"{model} pop@20", pop, num(beyond[(model, 20)]["mean_pop_percentile"]), 2)

    # --- Temporal vs LOO table ---------------------------------------------
    # Rows like: | ItemKNN | **0.319** | 0.271 |  (popularity omitted by design)
    for label, model in {k: v for k, v in MODELS.items() if v != "popularity"}.items():
        pattern = (
            rf"\|\s*{label}\s*\|\s*\*{{0,2}}([\d.]+)\*{{0,2}}\s*\|\s*"
            rf"\*{{0,2}}([\d.]+)\*{{0,2}}\s*\|"
        )
        # Restrict to the temporal-vs-LOO section to avoid matching other tables.
        section = README[README.index("A third protocol") :]
        m = re.search(pattern, section)
        if not m:
            failures.append(f"  NOT FOUND: temporal-vs-LOO row for {label}")
            continue
        temporal, loo = num(m.group(1)), num(m.group(2))
        check(f"{model} temporal NDCG@20", temporal,
              num(protocols[("temporal", model)]["ndcg@20"]), 3)
        check(f"{model} LOO NDCG@20", loo,
              num(protocols[("leave_one_out", model)]["ndcg@20"]), 3)

    _check_stage2(README)

    print(f"check-readme: {passes} numbers verified against outputs/")
    if failures:
        print(f"\n{len(failures)} MISMATCH(ES):")
        print("\n".join(failures))
        sys.exit(1)
    print("All README headline numbers match the generated outputs. ✓")


def _check_stage2(readme: str) -> None:
    """Verify the Stage 2 headline numbers if their outputs are present."""
    if not (OUTPUTS / "metrics_neural.csv").exists():
        return  # Stage 2 not run; Stage 1 checks stand alone

    neural = {(r["model"], r["metric"], int(r["k"])): r for r in read_csv("metrics_neural.csv")}
    ceiling = {(r["model"], int(r["n"])): r for r in read_csv("retrieval_ceiling.csv")}
    cold = {(r["model"], int(r["k"])): r for r in read_csv("metrics_cold.csv")}

    # Stage 2 headline table: | **Two-tower** | 0.260 | 0.510 |
    for label, model in [("Two-tower", "two_tower"),
                         ("SASRec (full-softmax)", "sasrec_full"),
                         ("SASRec (sampled-BCE)", "sasrec_sampled")]:
        row = re.search(rf"\|\s*\*{{0,2}}{re.escape(label)}\*{{0,2}}\s*\|\s*"
                        rf"([\d.]+)\s*\|\s*([\d.]+)\s*\|", readme)
        if row:
            check(f"{model} NDCG@20 (neural table)", num(row.group(1)),
                  num(neural[(model, "ndcg", 20)]["value"]), 3)
            check(f"{model} HR@20 (neural table)", num(row.group(2)),
                  num(neural[(model, "hit_rate", 20)]["value"]), 3)
        else:
            failures.append(f"  NOT FOUND: neural table row for {label}")

    # Retrieval-ceiling table's R@2000 column (last cell): | **Two-tower** | … | **0.929** |
    for label, model in [("ItemKNN", "itemknn"), ("EASE", "ease"), ("ALS", "als"),
                         ("Two-tower", "two_tower"), ("SASRec", "sasrec")]:
        row = re.search(rf"\|\s*\*{{0,2}}{re.escape(label)}\*{{0,2}}\s*\|"
                        rf"(?:\s*\*{{0,2}}[\d.]+\*{{0,2}}\s*\|){{4}}\s*\*{{0,2}}([\d.]+)\*{{0,2}}\s*\|",
                        readme)
        if row:
            check(f"{model} Recall@2000", num(row.group(1)),
                  num(ceiling[(model, 2000)]["recall"]), 3)

    # Cold-start table: | Two-tower (content) | 0.157 |
    for label, model in [("category-popularity heuristic", "category_popularity"),
                         ("Two-tower (content)", "two_tower")]:
        row = re.search(rf"\|\s*{re.escape(label)}\s*\|\s*\*{{0,2}}([\d.]+)\*{{0,2}}\s*\|", readme)
        if row:
            check(f"{model} cold Recall@20", num(row.group(1)),
                  num(cold[(model, 20)]["recall"]), 3)


if __name__ == "__main__":
    main()
