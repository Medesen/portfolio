#!/usr/bin/env python3
"""Check that README headline numbers match the generated outputs/ tables.

Run after ``make reproduce``:

    make check-readme        (or: python3 scripts/check_readme.py)

Every headline number in the project README — and the recsys numbers on the *portfolio
root* README — that has a counterpart in ``outputs/`` is parsed out of the markdown and
compared against the CSV value within rounding tolerance. Exits non-zero listing every
mismatch, so "reproducible" is *checked*, not merely asserted. Stdlib only: runs on any
Python 3.10+, no Docker or project dependencies needed.
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text()
ROOT_README_PATH = ROOT.parent / "README.md"
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
    # Accept a leading unicode minus (−) as well as ASCII, and strip bold markers/commas.
    return float(s.strip().replace("*", "").replace(",", "").replace("−", "-"))


def main() -> None:
    full = {(r["model"], r["metric"], int(r["k"])): r for r in read_csv("metrics_full.csv")}
    operational = {(r["model"], r["metric"], int(r["k"])): r
                   for r in read_csv("metrics_operational.csv")}
    beyond = {(r["model"], int(r["k"])): r for r in read_csv("metrics_beyond.csv")}
    protocols = {(r["protocol"], r["model"]): r for r in read_csv("protocol_comparison.csv")}

    # --- Full-catalogue table: NDCG@20 (+CI), HitRate@20, MRR@20, Operational NDCG@20 ---
    # Rows like: | **ItemKNN** | **0.335** | [0.330, 0.341] | 0.585 | 0.262 | 0.232 |
    for label, model in MODELS.items():
        pattern = (
            rf"\|\s*\*{{0,2}}{label}\*{{0,2}}\s*\|\s*\*{{0,2}}([\d.]+)\*{{0,2}}\s*\|\s*"
            rf"\[([\d.]+),\s*([\d.]+)\]\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|"
        )
        m = re.search(pattern, README)
        if not m:
            failures.append(f"  NOT FOUND: full-catalogue row for {label}")
            continue
        ndcg, lo, hi, hr, mrr, op = (num(g) for g in m.groups())
        check(f"{model} NDCG@20", ndcg, num(full[(model, "ndcg", 20)]["value"]), 3)
        check(f"{model} NDCG@20 CI-low", lo, num(full[(model, "ndcg", 20)]["ci_low"]), 3)
        check(f"{model} NDCG@20 CI-high", hi, num(full[(model, "ndcg", 20)]["ci_high"]), 3)
        check(f"{model} HR@20", hr, num(full[(model, "hit_rate", 20)]["value"]), 3)
        check(f"{model} MRR@20", mrr, num(full[(model, "mrr", 20)]["value"]), 3)
        check(f"{model} operational NDCG@20", op,
              num(operational[(model, "ndcg", 20)]["value"]), 3)

    # --- Cohort flow: the warm/cold/total counts that define the headline population ----
    cohort = read_csv("cohort_flow.csv")[0]
    for phrase, key in [(r"Warm target[^|]*\|\s*\*{0,2}([\d,]+)", "n_warm_target"),
                        (r"Cold target[^|]*\|\s*\*{0,2}([\d,]+)", "n_cold_target"),
                        (r"Post-cutoff total[^|]*\|\s*\*{0,2}([\d,]+)", "n_post_cutoff")]:
        m = re.search(phrase, README)
        if m:
            check(f"cohort {key}", num(m.group(1)), num(cohort[key]), 0)
        else:
            failures.append(f"  NOT FOUND: cohort-flow {key}")

    # --- Beyond-accuracy table: coverage% / gini / pop-percentile at k=20 ----
    for label, model in MODELS.items():
        pattern = rf"\|\s*{label}\s*\|\s*([\d.]+)%\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|"
        m = re.search(pattern, README)
        if not m:
            failures.append(f"  NOT FOUND: beyond-accuracy row for {label}")
            continue
        cov, gini, pop = (num(g) for g in m.groups())
        check(f"{model} coverage@20", cov, num(beyond[(model, 20)]["coverage"]) * 100, 1)
        check(f"{model} gini@20", gini, num(beyond[(model, 20)]["gini"]), 2)
        check(f"{model} pop@20", pop, num(beyond[(model, 20)]["mean_pop_percentile"]), 2)

    # --- Temporal vs LOO table ---------------------------------------------
    for label, model in {k: v for k, v in MODELS.items() if v != "popularity"}.items():
        pattern = (
            rf"\|\s*{label}\s*\|\s*\*{{0,2}}([\d.]+)\*{{0,2}}\s*\|\s*"
            rf"\*{{0,2}}([\d.]+)\*{{0,2}}\s*\|"
        )
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

    _check_sampled(README)
    _check_stage2(README)
    _check_ablations(README)
    _check_seed_sensitivity(README)
    _check_stage3(README)
    _check_paired(README)
    _check_root_readme(full, operational)

    print(f"check-readme: {passes} numbers verified against outputs/")
    if failures:
        print(f"\n{len(failures)} MISMATCH(ES):")
        print("\n".join(failures))
        sys.exit(1)
    print("All README headline numbers match the generated outputs. ✓")


def _check_sampled(readme: str) -> None:
    """The Stage-1 headline: ALS's sampled NDCG@20 towers over the others under the shortcut."""
    sampled = {(r["model"], r["protocol"], r["metric"], int(r["k"])): r
               for r in read_csv("metrics_sampled.csv")}
    m = re.search(r"sampled NDCG@20 \(([\d.]+) uniform / ([\d.]+) popularity\)", readme)
    if m:
        check("ALS sampled NDCG@20 (uniform)", num(m.group(1)),
              num(sampled[("als", "sampled_uniform", "ndcg", 20)]["value"]), 2)
        check("ALS sampled NDCG@20 (popularity)", num(m.group(2)),
              num(sampled[("als", "sampled_popularity", "ndcg", 20)]["value"]), 2)
    else:
        failures.append("  NOT FOUND: ALS sampled NDCG@20 phrase")


def _check_stage2(readme: str) -> None:
    """Verify the Stage 2 headline numbers if their outputs are present."""
    if not (OUTPUTS / "metrics_neural.csv").exists():
        return

    neural = {(r["model"], r["metric"], int(r["k"])): r for r in read_csv("metrics_neural.csv")}
    ceiling = {(r["model"], int(r["n"])): r for r in read_csv("retrieval_ceiling.csv")}
    cold = {(r["model"], int(r["k"])): r for r in read_csv("metrics_cold.csv")}
    blend = {r["blend"]: r for r in read_csv("retrieval_blend.csv")}

    # Stage 2 headline table: | **Two-tower** | 0.272 | 0.529 |
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

    # Retrieval-ceiling table — the WHOLE curve (R@50/100/500/1000/2000), not just the last
    # cell: the "ranking winners are the retrieval losers" argument rests on the shape.
    ceiling_cols = (50, 100, 500, 1000, 2000)
    section = readme[readme.index("| Retriever | R@50"):]
    for label, model in [("ItemKNN", "itemknn"), ("EASE", "ease"), ("ALS", "als"),
                         ("Two-tower", "two_tower"), ("SASRec", "sasrec")]:
        row = re.search(rf"\|\s*\*{{0,2}}{re.escape(label)}\*{{0,2}}\s*\|([^\n]+)", section)
        if not row:
            failures.append(f"  NOT FOUND: ceiling row for {label}")
            continue
        cells = re.findall(r"[\d.]+", row.group(1))
        for n, cell in zip(ceiling_cols, cells):
            check(f"{model} Recall@{n}", num(cell), num(ceiling[(model, n)]["recall"]), 3)

    # Blend headline: union of all five vs the best single retriever, at a 500 budget.
    m = re.search(r"union of all five reaches\s+([\d.]+)\s+at a\s+500-item budget each,\s+vs\s+"
                  r"([\d.]+)\s+for the best single retriever", readme)
    if m:
        check("blend all-five R@500", num(m.group(1)), num(blend["all"]["recall"]), 3)
        check("blend best-single R@500", num(m.group(2)), num(blend["als"]["recall"]), 3)
    else:
        failures.append("  NOT FOUND: retrieval-blend headline")

    # Cold-start table: | Two-tower (content) | 0.161 |
    for label, model in [("category-popularity heuristic", "category_popularity"),
                         ("Two-tower (content)", "two_tower")]:
        row = re.search(rf"\|\s*{re.escape(label)}\s*\|\s*\*{{0,2}}([\d.]+)\*{{0,2}}\s*\|", readme)
        if row:
            check(f"{model} cold Recall@20", num(row.group(1)),
                  num(cold[(model, 20)]["recall"]), 3)


def _check_ablations(readme: str) -> None:
    """The three Stage-2 ablation rows against ablations.csv (the table that had drifted)."""
    abl = {(r["family"], r["variant"]): r for r in read_csv("ablations.csv")}

    m = re.search(r"id\+content \*\*([\d.]+)\*\* > id-only ([\d.]+) > content-only ([\d.]+)", readme)
    if m:
        check("ablation id+content", num(m.group(1)),
              num(abl[("item_tower", "id_plus_content")]["ndcg@20"]), 3)
        check("ablation id-only", num(m.group(2)),
              num(abl[("item_tower", "id_only")]["ndcg@20"]), 3)
        check("ablation content-only", num(m.group(3)),
              num(abl[("item_tower", "content_only")]["ndcg@20"]), 3)
    else:
        failures.append("  NOT FOUND: item-tower ablation row")

    m = re.search(r"on \*\*([\d.]+)\*\* vs off ([\d.]+):", readme)
    if m:
        check("ablation logq on", num(m.group(1)),
              num(abl[("logq", "correction_on")]["ndcg@20"]), 3)
        check("ablation logq off", num(m.group(2)),
              num(abl[("logq", "correction_off")]["ndcg@20"]), 3)
    else:
        failures.append("  NOT FOUND: logQ ablation row")

    m = re.search(r"full-softmax \*\*([\d.]+)\*\* vs sampled-BCE ([\d.]+)", readme)
    if m:
        check("ablation sasrec full", num(m.group(1)),
              num(abl[("sasrec_loss", "full_softmax")]["ndcg@20"]), 3)
        check("ablation sasrec sampled", num(m.group(2)),
              num(abl[("sasrec_loss", "sampled_bce")]["ndcg@20"]), 3)
    else:
        failures.append("  NOT FOUND: SASRec-loss ablation row")


def _check_seed_sensitivity(readme: str) -> None:
    """The P1-5 seed-robustness table: mean and range of the three headline deltas."""
    deltas = {r["delta"]: r for r in read_csv("seed_sensitivity_deltas.csv")}
    rows = [
        (r"logQ on . off \(two-tower\)\s*\|\s*\*\*\+([\d.]+)\*\*\s*\|\s*\[\+([\d.]+),\s*\+([\d.]+)\]",
         "logq_on_minus_off"),
        (r"reranker . retriever \(Stage 3\)\s*\|\s*\*\*\+([\d.]+)\*\*\s*\|\s*\[\+([\d.]+),\s*\+([\d.]+)\]",
         "rerank_minus_retrieval"),
        (r"LambdaMART . pointwise \(Stage 3\)\s*\|\s*\*\*\+([\d.]+)\*\*\s*\|\s*\[\+([\d.]+),\s*\+([\d.]+)\]",
         "lambdarank_minus_binary"),
    ]
    for pattern, key in rows:
        m = re.search(pattern, readme)
        if m:
            check(f"seed {key} mean", num(m.group(1)), num(deltas[key]["mean"]), 3)
            check(f"seed {key} min", num(m.group(2)), num(deltas[key]["min"]), 3)
            check(f"seed {key} max", num(m.group(3)), num(deltas[key]["max"]), 3)
        else:
            failures.append(f"  NOT FOUND: seed-sensitivity row {key}")


def _check_stage3(readme: str) -> None:
    """Verify the reproducible Stage 3 numbers (end-to-end accuracy)."""
    if not (OUTPUTS / "metrics_e2e.csv").exists():
        return
    e2e = {(r["model"], r["metric"], int(r["k"])): r for r in read_csv("metrics_e2e.csv")}

    for label, model in [("Retrieval only (ALS order)", "retrieval_only"),
                         ("Two-stage (LambdaMART rerank)", "two_stage_lambdarank"),
                         ("Two-stage (pointwise rerank)", "two_stage_binary")]:
        row = re.search(rf"\|\s*\*{{0,2}}{re.escape(label)}\*{{0,2}}\s*\|\s*"
                        rf"\*{{0,2}}([\d.]+)\*{{0,2}}\s*\|\s*([\d.]+)\s*\|", readme)
        if row:
            check(f"{model} e2e NDCG@20", num(row.group(1)),
                  num(e2e[(model, "ndcg", 20)]["value"]), 3)
            check(f"{model} e2e HR@20", num(row.group(2)),
                  num(e2e[(model, "hit_rate", 20)]["value"]), 3)
        else:
            failures.append(f"  NOT FOUND: e2e row for {label}")


def _check_paired(readme: str) -> None:
    """Verify the paired-bootstrap intervals — including the ItemKNN-EASE result, which now
    *declines* to resolve a winner (the interval includes zero)."""
    if (OUTPUTS / "paired_ci_full.csv").exists():
        row = {r["comparison"]: r for r in read_csv("paired_ci_full.csv")}["itemknn_minus_ease"]
        m = re.search(r"does not resolve a winner:\s*\+?([\d.]+)\s*"
                      r"\[([−+-]?[\d.]+),\s*\+?([\d.]+)\]", readme)
        if m:
            check("paired ItemKNN-EASE diff", num(m.group(1)), num(row["diff"]), 4)
            check("paired ItemKNN-EASE CI-low", num(m.group(2)), num(row["ci_low"]), 4)
            check("paired ItemKNN-EASE CI-high", num(m.group(3)), num(row["ci_high"]), 4)
        else:
            failures.append("  NOT FOUND: paired ItemKNN-EASE interval")

    if (OUTPUTS / "paired_ci_e2e.csv").exists():
        e2e = {r["comparison"]: r for r in read_csv("paired_ci_e2e.csv")}
        for phrase, key in [
            (r"beats its own retriever[^+]*\+([\d.]+) \[\+([\d.]+),\s*\+([\d.]+)\]",
             "rerank_minus_retrieval"),
            (r"beats the pointwise objective[^+]*\+([\d.]+) \[\+([\d.]+),\s*\+([\d.]+)\]",
             "lambdarank_minus_pointwise"),
        ]:
            m = re.search(phrase, readme)
            if m:
                check(f"e2e paired {key} diff", num(m.group(1)), num(e2e[key]["diff"]), 3)
                check(f"e2e paired {key} CI-low", num(m.group(2)), num(e2e[key]["ci_low"]), 3)
                check(f"e2e paired {key} CI-high", num(m.group(3)), num(e2e[key]["ci_high"]), 3)
            else:
                failures.append(f"  NOT FOUND: e2e paired interval {key}")


def _check_root_readme(full: dict, operational: dict) -> None:
    """The portfolio *root* README's recsys headline must tell the same (corrected) story —
    the landing page a hiring manager sees first (P0-2)."""
    if not ROOT_README_PATH.exists():
        failures.append(f"  NOT FOUND: root README at {ROOT_README_PATH}")
        return
    root = ROOT_README_PATH.read_text()
    neural = {(r["model"], r["metric"], int(r["k"])): r for r in read_csv("metrics_neural.csv")}
    ceiling = {(r["model"], int(r["n"])): r for r in read_csv("retrieval_ceiling.csv")}
    e2e = {(r["model"], r["metric"], int(r["k"])): r for r in read_csv("metrics_e2e.csv")}

    checks = [
        (r"ItemKNN/EASE NDCG@20 ≈ ([\d.]+)", num(full[("itemknn", "ndcg", 20)]["value"]), 2,
         "root ItemKNN/EASE NDCG@20"),
        (r"ALS ([\d.]+) \(best of the four", num(full[("als", "ndcg", 20)]["value"]), 3,
         "root ALS NDCG@20"),
        (r"two-tower \*\*loses\*\* on ranking \(([\d.]+)\)",
         num(neural[("two_tower", "ndcg", 20)]["value"]), 3, "root two-tower NDCG@20"),
        (r"Recall@2000 ([\d.]+) vs ItemKNN's ([\d.]+)",
         num(ceiling[("two_tower", 2000)]["recall"]), 2, "root two-tower R@2000"),
        (r"beats the best single model\*\* \(([\d.]+) vs ([\d.]+) NDCG@20, paired \+([\d.]+)\)",
         num(e2e[("two_stage_lambdarank", "ndcg", 20)]["value"]), 3, "root two-stage NDCG@20"),
    ]
    for pattern, actual, dec, label in checks:
        m = re.search(pattern, root)
        if m:
            check(label, num(m.group(1)), actual, dec)
        else:
            failures.append(f"  NOT FOUND: {label} in root README")

    # The two-part claims: two-tower R@2000 vs ItemKNN's, and two-stage vs best-single.
    m = re.search(r"Recall@2000 ([\d.]+) vs ItemKNN's ([\d.]+)", root)
    if m:
        check("root ItemKNN R@2000", num(m.group(2)),
              num(ceiling[("itemknn", 2000)]["recall"]), 2)
    m = re.search(r"beats the best single model\*\* \(([\d.]+) vs ([\d.]+) NDCG@20, paired \+([\d.]+)\)",
                  root)
    if m:
        check("root best-single ItemKNN NDCG@20", num(m.group(2)),
              num(full[("itemknn", "ndcg", 20)]["value"]), 3)


if __name__ == "__main__":
    main()
