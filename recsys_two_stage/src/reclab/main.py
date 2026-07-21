"""CLI: ``reclab {eda,tune,evaluate,sampled,protocols,beyond,all}``.

Each subcommand writes CSV(s) to ``--out`` (default ``outputs/``) and prints a
human-readable summary. ``all`` runs the full pipeline and reproduces every number
in the README.

Note on the metric set: because the session-based protocol holds out a single next
item per test session, Recall@k collapses to HitRate@k (one relevant item), so the
honest metric set here is HitRate@k / NDCG@k / MRR@k. This is stated in the README
rather than quietly keeping a "recall" column that no longer distinguishes anything.

Tuning is deliberately separated from ``all``: a full grid search dominates runtime
for marginal insight on already-clear winners, so the tuned hyperparameters live in
``TUNED_PARAMS`` (produced by ``reclab tune``) and ``all`` uses them. Re-run ``tune``
to regenerate them; the grid points are written to ``outputs/tuning_<model>.csv`` so
the choice is auditable.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from reclab.data import load_events, sessionize, to_session_items
from reclab.data.filtering import k_core_filter
from reclab.evaluation.beyond_accuracy import evaluate_beyond_accuracy
from reclab.evaluation.full_catalogue import evaluate
from reclab.evaluation.sampled import evaluate_sampled, protocol_disagreement
from reclab.models import ALS, EASE, ItemKNN, Popularity
from reclab.splitting import leave_one_out_split, temporal_split

# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
TEST_DAYS = 28
VAL_DAYS = 28
MIN_SESSION_ITEMS = 2
MIN_ITEM_SESSIONS = 10
KS = (10, 20, 50)

# Hyperparameter grids searched by ``reclab tune`` on the validation window.
GRIDS = {
    "itemknn": {"k": [50, 100, 200, 300], "shrink": [0.0, 50.0, 100.0, 500.0]},
    "ease": {"reg": [100.0, 250.0, 500.0, 1000.0, 2000.0]},
    "als": {
        "factors": [64, 128],
        "regularization": [1.0, 10.0],
        "alpha": [20.0, 40.0],
    },
}
ALS_FIXED = {"iterations": 15, "num_threads": 4, "seed": 0}

# Winners from ``reclab tune``, selected on the validation window by NDCG@20.
# Reproduce with ``reclab tune``; grid points are in outputs/tuning_<model>.csv.
TUNED_PARAMS: dict[str, dict] = {
    "popularity": {},
    "itemknn": {"k": 300, "shrink": 100.0},
    "ease": {"reg": 500.0},
    "als": {"factors": 128, "regularization": 1.0, "alpha": 40.0, **ALS_FIXED},
}

MODEL_CLASSES = {
    "popularity": Popularity,
    "itemknn": ItemKNN,
    "ease": EASE,
    "als": ALS,
}


def build_model(name: str):
    return MODEL_CLASSES[name](**TUNED_PARAMS[name])


# --------------------------------------------------------------------------- #
# shared data loading
# --------------------------------------------------------------------------- #
def load_pairs(gap_minutes: int = 30) -> pd.DataFrame:
    events, report = load_events()
    print(f"  {report}")
    events = sessionize(events, gap_minutes=gap_minutes)
    pairs = to_session_items(events)
    print(f"  {pairs['session'].nunique():,} sessions, {len(pairs):,} session-item pairs")
    return pairs


def build_split(pairs: pd.DataFrame, filter_scope: str = "train"):
    split = temporal_split(
        pairs,
        test_days=TEST_DAYS,
        min_session_items=MIN_SESSION_ITEMS,
        min_item_sessions=MIN_ITEM_SESSIONS,
        filter_scope=filter_scope,
    )
    print(f"  {split}")
    if split.n_straddling_dropped:
        print(f"  dropped {split.n_straddling_dropped} sessions straddling the cutoff")
    return split


# --------------------------------------------------------------------------- #
# subcommands
# --------------------------------------------------------------------------- #
def run_eda(pairs: pd.DataFrame, out: Path) -> None:
    print("\n=== Dataset summary ===")
    summary = pd.DataFrame(
        [{
            "sessions": pairs["session"].nunique(),
            "items": pairs["itemid"].nunique(),
            "pairs": len(pairs),
            "mean_items_per_session": len(pairs) / pairs["session"].nunique(),
        }]
    )
    print(summary.to_string(index=False))

    print("\n=== Filter-threshold grid (session>=s, item>=i) ===")
    rows = []
    for s in (2, 3):
        for i in (5, 10, 20):
            filtered, rep = k_core_filter(pairs[["session", "itemid"]].drop_duplicates(), s, i)
            n_sess = filtered["session"].nunique() if len(filtered) else 0
            n_items = filtered["itemid"].nunique() if len(filtered) else 0
            rows.append({
                "min_session_items": s, "min_item_sessions": i,
                "sessions": n_sess, "items": n_items, "pairs": len(filtered),
                "density_%": 100 * len(filtered) / max(n_sess * n_items, 1),
                "ease_matrix_gb": n_items ** 2 * 8 / 1e9,
            })
    grid = pd.DataFrame(rows)
    print(grid.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "data_summary.csv", index=False)
    grid.to_csv(out / "filter_thresholds.csv", index=False)
    print(f"\nwrote {out}/data_summary.csv, {out}/filter_thresholds.csv")


def run_tune(pairs: pd.DataFrame, out: Path) -> None:
    from reclab.tuning import grid_search, validation_split

    val = validation_split(
        pairs, test_days=TEST_DAYS, val_days=VAL_DAYS,
        min_session_items=MIN_SESSION_ITEMS, min_item_sessions=MIN_ITEM_SESSIONS,
    )
    print(f"\n  validation split: {val}")
    print(f"  validation cutoff {val.cutoff.date()} (strictly before the test cutoff)")

    out.mkdir(parents=True, exist_ok=True)
    for name, grid in GRIDS.items():
        fixed = ALS_FIXED if name == "als" else {}
        result = grid_search(MODEL_CLASSES[name], grid, val, metric="ndcg", k=20, fixed=fixed)
        print(f"\n=== tune {name} ===\n{result}")
        print(result.table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        result.table.to_csv(out / f"tuning_{name}.csv", index=False)
    print(f"\nwrote {out}/tuning_<model>.csv — transfer winners into TUNED_PARAMS")


def run_evaluate(pairs: pd.DataFrame, out: Path) -> pd.DataFrame:
    split = build_split(pairs)
    rows = []
    print("\n=== Full-catalogue evaluation (honest protocol) ===")
    for name in MODEL_CLASSES:
        model = build_model(name).fit(split.train)
        result = evaluate(model, split, ks=KS)
        for k in KS:
            for metric in ("hit_rate", "ndcg", "mrr"):
                mean, lo, hi = result.bootstrap_ci(metric, k)
                rows.append({
                    "model": name, "metric": metric, "k": k,
                    "value": mean, "ci_low": lo, "ci_high": hi,
                    "n_sessions": result.n_sessions,
                })
        m, lo, hi = result.bootstrap_ci("ndcg", 20)
        print(f"  {name:11s} NDCG@20={m:.4f} [{lo:.4f}, {hi:.4f}]  "
              f"HR@20={result.per_session[('hit_rate', 20)].mean():.4f}")
        del model
    table = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "metrics_full.csv", index=False)
    print(f"\nwrote {out}/metrics_full.csv")
    return table


def run_sampled(pairs: pd.DataFrame, out: Path) -> None:
    split = build_split(pairs)
    item_pop = np.asarray(split.train.sum(axis=0)).ravel()
    full_rows, sampled_frames = [], []

    print("\n=== Sampled-negative evaluation (the shortcut) ===")
    for name in MODEL_CLASSES:
        model = build_model(name).fit(split.train)
        full = evaluate(model, split, ks=(10, 20))
        full_rows.append(full.summary())
        for sampler in ("uniform", "popularity"):
            sampled_frames.append(
                evaluate_sampled(model, split, item_pop, sampler=sampler, ks=(10, 20))
            )
        del model

    full_df = pd.concat(full_rows, ignore_index=True)
    sampled_df = pd.concat(sampled_frames, ignore_index=True)
    out.mkdir(parents=True, exist_ok=True)
    sampled_df.to_csv(out / "metrics_sampled.csv", index=False)

    from scipy.stats import spearmanr

    disagreements = []
    for sampler in ("uniform", "popularity"):
        sub = sampled_df[sampled_df["protocol"] == f"sampled_{sampler}"]
        table = protocol_disagreement(full_df, sub, metric="ndcg", k=20)
        table["sampler"] = sampler
        rho, _ = spearmanr(table["full_rank"], table["sampled_rank"])
        disagreements.append(table)
        print(f"\n--- full vs sampled ({sampler} negatives), NDCG@20 ---")
        print(table.drop(columns="sampler").to_string(index=False,
              float_format=lambda x: f"{x:.4f}"))
        print(f"Spearman(full_rank, sampled_rank) = {rho:.3f}"
              f"{'  <-- rankings disagree' if rho < 1.0 else ''}")

    pd.concat(disagreements, ignore_index=True).to_csv(
        out / "protocol_disagreement.csv", index=False
    )
    print(f"\nwrote {out}/metrics_sampled.csv, {out}/protocol_disagreement.csv")


def run_protocols(pairs: pd.DataFrame, out: Path) -> None:
    print("\n=== Temporal vs leave-one-out protocol ===")
    rows = []
    temporal = build_split(pairs)
    loo = leave_one_out_split(
        pairs, min_session_items=MIN_SESSION_ITEMS, min_item_sessions=MIN_ITEM_SESSIONS
    )
    print(f"  leave-one-out: {loo}")
    for label, split in [("temporal", temporal), ("leave_one_out", loo)]:
        for name in MODEL_CLASSES:
            model = build_model(name).fit(split.train)
            result = evaluate(model, split, ks=(10, 20))
            rows.append({
                "protocol": label, "model": name,
                "ndcg@20": result.per_session[("ndcg", 20)].mean(),
                "hit_rate@20": result.per_session[("hit_rate", 20)].mean(),
            })
            del model
    table = pd.DataFrame(rows)
    print("\n" + table.pivot(index="model", columns="protocol", values="ndcg@20")
          .to_string(float_format=lambda x: f"{x:.4f}"))
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "protocol_comparison.csv", index=False)
    print(f"\nwrote {out}/protocol_comparison.csv")


def run_beyond(pairs: pd.DataFrame, out: Path) -> None:
    split = build_split(pairs)
    item_pop = np.asarray(split.train.sum(axis=0)).ravel()
    frames = []
    print("\n=== Beyond-accuracy metrics ===")
    for name in MODEL_CLASSES:
        model = build_model(name).fit(split.train)
        frames.append(evaluate_beyond_accuracy(model, split, item_pop, ks=(10, 20)))
        del model
    table = pd.concat(frames, ignore_index=True)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "metrics_beyond.csv", index=False)
    print(f"\nwrote {out}/metrics_beyond.csv")


# --------------------------------------------------------------------------- #
# argument parsing
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(
        prog="reclab",
        description="Two-stage recommender evaluation on RetailRocket (Stage 1).",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in [
        ("eda", "dataset summary + filter-threshold grid"),
        ("tune", "grid search on the validation window"),
        ("evaluate", "full-catalogue evaluation of all models"),
        ("sampled", "sampled-negative evaluation + disagreement table"),
        ("protocols", "temporal vs leave-one-out comparison"),
        ("beyond", "coverage / Gini / popularity-bias metrics"),
        ("all", "run the whole pipeline (reproduce the README)"),
    ]:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--out", type=Path, default=Path("outputs"))

    args = parser.parse_args()
    print(f"[reclab {args.command}] loading data...")
    pairs = load_pairs()

    if args.command == "eda":
        run_eda(pairs, args.out)
    elif args.command == "tune":
        run_tune(pairs, args.out)
    elif args.command == "evaluate":
        run_evaluate(pairs, args.out)
    elif args.command == "sampled":
        run_sampled(pairs, args.out)
    elif args.command == "protocols":
        run_protocols(pairs, args.out)
    elif args.command == "beyond":
        run_beyond(pairs, args.out)
    elif args.command == "all":
        run_eda(pairs, args.out)
        run_evaluate(pairs, args.out)
        run_sampled(pairs, args.out)
        run_protocols(pairs, args.out)
        run_beyond(pairs, args.out)
        print("\n=== reproduce complete ===")


if __name__ == "__main__":
    main()
