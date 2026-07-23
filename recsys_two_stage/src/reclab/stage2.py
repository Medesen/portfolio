"""Stage 2 CLI orchestration: neural models, ablations, cold-start, retrieval ceiling.

Kept out of ``main.py`` so the Stage 1 CLI stays legible. Every function writes CSVs
to ``out`` and returns what downstream steps reuse, so ``reclab all`` can fit each
model once and thread the fitted objects through the analyses rather than retraining.

Neural hyperparameters are lightly tuned by hand within the CPU budget rather than
grid-searched: SASRec at ~6 min/run (~48 s/epoch × 8 epochs) makes a large grid expensive
for a result whose qualitative shape (neural loses the full-catalogue benchmark) is already
clear. The values and that decision are documented in the README.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from reclab.evaluation.full_catalogue import evaluate

# max_len is small because RetailRocket sessions are short (~3 items); a longer window
# is wasted compute, exactly as the plan anticipated. SASRec epochs are kept modest
# (~48s/epoch on CPU) so the full Stage 2 pipeline fits a ~20-minute budget.
NEURAL_PARAMS = {
    "two_tower": dict(emb_dim=64, hidden=128, temperature=0.05, epochs=12,
                      batch_size=512, logq_correction=True,
                      item_tower_mode="id_plus_content", seed=0),
    "sasrec": dict(emb_dim=64, max_len=20, n_blocks=2, n_heads=2,
                   loss="full_softmax", epochs=8, batch_size=512, seed=0),
}


def build_features(split):
    from reclab.features import build_item_features
    return build_item_features(split.cutoff, split.item_ids)


def fit_two_tower(split, features, **overrides):
    from reclab.models import TwoTower
    params = {**NEURAL_PARAMS["two_tower"], **overrides}
    return TwoTower(n_items=split.n_items, item_features=features, **params).fit(split)


def fit_sasrec(split, **overrides):
    from reclab.models import SASRec
    params = {**NEURAL_PARAMS["sasrec"], **overrides}
    return SASRec(n_items=split.n_items, **params).fit(split)


# --------------------------------------------------------------------------- #
# feature coverage
# --------------------------------------------------------------------------- #
def run_features(split, out: Path, features=None) -> None:
    features = features or build_features(split)
    cov = features.coverage()
    print("\n=== Item feature coverage (training items, as-of cutoff) ===")
    for k, v in cov.items():
        print(f"  {k:20s}: {v:.4f}" if isinstance(v, float) else f"  {k:20s}: {v}")
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([cov]).to_csv(out / "item_feature_coverage.csv", index=False)
    print(f"\nwrote {out}/item_feature_coverage.csv")


# --------------------------------------------------------------------------- #
# neural models in the comparison table
# --------------------------------------------------------------------------- #
def fit_stage2_models(split, features, log=print) -> dict:
    """Fit every unique neural model Stage 2 needs, exactly once.

    The main two-tower (id+content, logQ on) doubles as the ``logq=on`` and
    ``item_tower=id_plus_content`` ablation rows, so nothing is trained twice — the
    difference between this finishing in ~15 minutes and ~40.
    """
    import time

    specs = {
        "two_tower": lambda: fit_two_tower(split, features),
        "two_tower_no_logq": lambda: fit_two_tower(split, features, logq_correction=False),
        "two_tower_id_only": lambda: fit_two_tower(split, features, item_tower_mode="id_only"),
        "two_tower_content_only": lambda: fit_two_tower(split, features,
                                                        item_tower_mode="content_only"),
        "sasrec_full": lambda: fit_sasrec(split, loss="full_softmax"),
        "sasrec_sampled": lambda: fit_sasrec(split, loss="sampled_bce"),
    }
    models = {}
    for label, fitter in specs.items():
        s = time.time()
        models[label] = fitter()
        log(f"  fitted {label} in {time.time() - s:.0f}s")
    return models


def run_neural(split, out: Path, features=None, models=None) -> pd.DataFrame:
    features = features or build_features(split)
    models = models or fit_stage2_models(split, features)
    reported = {"two_tower": models["two_tower"], "sasrec_full": models["sasrec_full"],
                "sasrec_sampled": models["sasrec_sampled"]}
    rows = []
    print("\n=== Neural models — full-catalogue evaluation (same protocol as Stage 1) ===")
    for label, model in reported.items():
        result = evaluate(model, split, ks=(10, 20))
        for k in (10, 20):
            for metric in ("hit_rate", "ndcg", "mrr"):
                mean, lo, hi = result.bootstrap_ci(metric, k)
                rows.append({"model": label, "metric": metric, "k": k,
                             "value": mean, "ci_low": lo, "ci_high": hi,
                             "n_sessions": result.n_sessions})
        m, lo, hi = result.bootstrap_ci("ndcg", 20)
        print(f"  {label:16s} NDCG@20={m:.4f} [{lo:.4f}, {hi:.4f}]  "
              f"HR@20={result.per_session[('hit_rate', 20)].mean():.4f}")
    table = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "metrics_neural.csv", index=False)
    print(f"\nwrote {out}/metrics_neural.csv")
    return table


# --------------------------------------------------------------------------- #
# ablations: logQ on/off, id vs content, full vs sampled loss
# --------------------------------------------------------------------------- #
def run_ablations(split, out: Path, features=None, models=None) -> pd.DataFrame:
    from reclab.evaluation.beyond_accuracy import evaluate_beyond_accuracy

    features = features or build_features(split)
    models = models or fit_stage2_models(split, features)
    item_pop = np.asarray(split.train.sum(axis=0)).ravel()
    rows = []
    print("\n=== Ablations ===")

    def record(family, variant, model):
        r = evaluate(model, split, ks=(20,))
        b = evaluate_beyond_accuracy(model, split, item_pop, ks=(20,)).iloc[0]
        ndcg = r.per_session[("ndcg", 20)].mean()
        rows.append({"family": family, "variant": variant, "ndcg@20": ndcg,
                     "hit_rate@20": r.per_session[("hit_rate", 20)].mean(),
                     "coverage@20": b["coverage"], "pop_percentile@20": b["mean_pop_percentile"]})
        print(f"  {family:12s} {variant:16s} NDCG@20={ndcg:.4f} "
              f"cov={b['coverage']:.3f} pop%={b['mean_pop_percentile']:.3f}")

    # The main two-tower is both the logQ-on and the id+content row — reused, not refit.
    record("logq", "correction_on", models["two_tower"])
    record("logq", "correction_off", models["two_tower_no_logq"])
    record("item_tower", "id_plus_content", models["two_tower"])
    record("item_tower", "id_only", models["two_tower_id_only"])
    record("item_tower", "content_only", models["two_tower_content_only"])
    record("sasrec_loss", "full_softmax", models["sasrec_full"])
    record("sasrec_loss", "sampled_bce", models["sasrec_sampled"])

    table = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "ablations.csv", index=False)
    print(f"\nwrote {out}/ablations.csv")
    return table


# --------------------------------------------------------------------------- #
# retrieval ceiling
# --------------------------------------------------------------------------- #
def run_ceiling(split, out: Path, classical: dict, neural: dict) -> None:
    from reclab.evaluation.retrieval_ceiling import blend_ceiling, retrieval_ceiling

    # Retrieval candidates only come from the embedding/co-occurrence retrievers,
    # not the trivial popularity baseline; use the strongest of each family.
    retrievers = {
        "itemknn": classical["itemknn"],
        "ease": classical["ease"],
        "als": classical["als"],
        "two_tower": neural["two_tower"],
        "sasrec": neural["sasrec_full"],
    }
    print("\n=== Retrieval-ceiling analysis ===")
    table, rankings = retrieval_ceiling(retrievers, split, ns=(50, 100, 200, 500, 1000, 2000))
    pivot = table.pivot(index="model", columns="n", values="recall")
    print(pivot.to_string(float_format=lambda x: f"{x:.4f}"))

    blend = blend_ceiling(rankings, split.test_target, budget=500)
    print("\n--- blended retrievers (union, budget 500 each) ---")
    print(blend.head(8).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "retrieval_ceiling.csv", index=False)
    blend.to_csv(out / "retrieval_blend.csv", index=False)
    _plot_ceiling(table, out / "retrieval_ceiling.png")
    print(f"\nwrote {out}/retrieval_ceiling.csv, retrieval_blend.csv, retrieval_ceiling.png")


def _plot_ceiling(table: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model in table["model"].unique():
        sub = table[table["model"] == model].sort_values("n")
        ax.plot(sub["n"], sub["recall"], marker="o", label=model)
    ax.set_xscale("log")
    ax.set_xlabel("candidate-set size N (log scale)")
    ax.set_ylabel("Recall@N  (target found within top N)")
    ax.set_title("Retrieval ceiling — how often the target is retrievable at all")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# cold start
# --------------------------------------------------------------------------- #
def run_cold_start(pairs, split, out: Path, two_tower=None, features=None) -> None:
    from reclab.evaluation.cold_start import (
        build_cold_start_eval,
        evaluate_category_popularity_cold,
        evaluate_two_tower_cold,
    )

    features = features or build_features(split)
    two_tower = two_tower or fit_two_tower(split, features)
    cold = build_cold_start_eval(pairs, split, features, min_cold_support=2)

    print("\n=== Cold-start evaluation (new items) ===")
    print(f"  cold-item share of evaluable test targets: {cold.cold_share:.1%}")
    print(f"  near-cold (<5 train interactions) share:   {cold.near_cold_share:.1%}")
    print(f"  evaluable cold sessions: {cold.n_sessions:,}  cold candidates: {len(cold.cold_item_ids):,}")

    tt = evaluate_two_tower_cold(two_tower, cold, ks=(10, 20, 50))
    catpop = evaluate_category_popularity_cold(cold, pairs, split, features, ks=(10, 20, 50))
    # Classical models score 0.0 on cold items by construction.
    structural = pd.DataFrame([
        {"model": name, "k": k, "recall": 0.0, "ndcg": 0.0, "n_sessions": cold.n_sessions}
        for name in ("itemknn", "ease", "als") for k in (10, 20, 50)
    ])
    table = pd.concat([tt, catpop, structural], ignore_index=True)
    print("\ncold-start recall/NDCG @20:")
    print(table[table["k"] == 20].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "metrics_cold.csv", index=False)
    pd.DataFrame([{"cold_share": cold.cold_share, "near_cold_share": cold.near_cold_share,
                   "n_cold_sessions": cold.n_sessions, "n_cold_items": len(cold.cold_item_ids)}]
                 ).to_csv(out / "cold_start_share.csv", index=False)
    print(f"\nwrote {out}/metrics_cold.csv, cold_start_share.csv")
