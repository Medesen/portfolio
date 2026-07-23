"""CLI for the three-stage recommender study.

Stage 1 (classical): ``eda tune evaluate sampled protocols beyond``.
Stage 2 (neural):    ``features neural ablations ceiling cold-start stage2``.
Stage 3 (two-stage): ``evaluate-e2e ann-sweep scaling build-service serve latency stage3``.

Each subcommand writes CSV(s) to ``--out`` (default ``outputs/``) and prints a
human-readable summary. ``all`` runs the full pipeline across all three stages and
regenerates every reported number in the README **given the frozen tuned parameters**;
hyperparameter selection itself is a separate, auditable step (``reclab tune``), not
part of ``all`` (see the tuning note below).

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
from reclab.evaluation.full_catalogue import (
    evaluate,
    operational_bootstrap_ci,
    paired_bootstrap_diff,
)
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
    # alpha is ALS's confidence scale (confidence = 1 + alpha*value); on this sparse
    # implicit data the validation optimum is far above the 20-40 first tried, so the
    # grid runs up to 1280 (the optimum, alpha=640/reg=0.1, is interior — validation
    # NDCG@20 rises to alpha=640 and falls again by 1280). Under-tuning alpha here is exactly the
    # Dacrema-et-al. failure the project exists to avoid, so it is tuned as widely as
    # the neighbourhood models are.
    "als": {
        "factors": [64, 128],
        "regularization": [0.1, 1.0, 10.0],
        "alpha": [40.0, 80.0, 160.0, 320.0, 640.0, 1280.0],
    },
}
ALS_FIXED = {"iterations": 15, "num_threads": 4, "seed": 0}

# Winners from ``reclab tune``, selected on the validation window by NDCG@20.
# Reproduce with ``reclab tune``; grid points are in outputs/tuning_<model>.csv.
TUNED_PARAMS: dict[str, dict] = {
    "popularity": {},
    "itemknn": {"k": 300, "shrink": 100.0},
    "ease": {"reg": 500.0},
    "als": {"factors": 128, "regularization": 0.1, "alpha": 640.0, **ALS_FIXED},
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
    cohort = split.cohort
    n_forced_miss = cohort.n_cold_target if cohort else 0

    # Cohort flow: account for every post-cutoff session before quoting any accuracy.
    print("\n=== Test cohort flow (temporal split) ===")
    if cohort:
        print(f"  post-cutoff sessions:        {cohort.n_post_cutoff:,}")
        print(f"  warm-target (headline):      {cohort.n_warm_target:,} "
              f"({cohort.n_warm_target / cohort.n_post_cutoff:.1%})")
        print(f"  cold-target (forced miss):   {cohort.n_cold_target:,} "
              f"({cohort.n_cold_target / cohort.n_post_cutoff:.1%})")
        print(f"  insufficient warm prefix:    {cohort.n_insufficient:,}")
        flow = {"n_straddling_dropped": split.n_straddling_dropped, **cohort.as_dict()}
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([flow]).to_csv(out / "cohort_flow.csv", index=False)

    rows, op_rows = [], []
    ndcg20_per_session: dict[str, np.ndarray] = {}
    print("\n=== Full-catalogue evaluation (honest protocol) ===")
    print("  (conditional = warm-target headline; operational = actual next item, "
          "cold targets counted as misses)")
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
                if metric in ("hit_rate", "ndcg") and k in (10, 20):
                    op_m, op_lo, op_hi = operational_bootstrap_ci(
                        result.per_session[(metric, k)], n_forced_miss
                    )
                    op_rows.append({
                        "model": name, "metric": metric, "k": k,
                        "value": op_m, "ci_low": op_lo, "ci_high": op_hi,
                        "n_sessions": result.n_sessions + n_forced_miss,
                    })
        ndcg20_per_session[name] = result.per_session[("ndcg", 20)]
        m, lo, hi = result.bootstrap_ci("ndcg", 20)
        op_m, op_lo, op_hi = operational_bootstrap_ci(
            result.per_session[("ndcg", 20)], n_forced_miss
        )
        print(f"  {name:11s} NDCG@20 conditional={m:.4f} [{lo:.4f}, {hi:.4f}]  "
              f"operational={op_m:.4f} [{op_lo:.4f}, {op_hi:.4f}]")
        del model
    table = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "metrics_full.csv", index=False)
    pd.DataFrame(op_rows).to_csv(out / "metrics_operational.csv", index=False)

    # The two front-runners (ItemKNN, EASE) are within a hair on the marginal means,
    # and their marginal CIs overlap — but overlapping marginal CIs are not a test of
    # the difference. Both models score the *same* sessions, so the honest question is
    # whether the paired per-session difference excludes zero. The interval and verdict
    # are computed below and reported as whatever the data supports — "resolves a winner"
    # when it excludes zero, "does not resolve" when it straddles it — never asserted.
    diff_mean, diff_lo, diff_hi = paired_bootstrap_diff(
        ndcg20_per_session["itemknn"], ndcg20_per_session["ease"]
    )
    resolves = not (diff_lo <= 0.0 <= diff_hi)
    print(f"\n  paired NDCG@20 (ItemKNN - EASE) = {diff_mean:+.4f} "
          f"[{diff_lo:+.4f}, {diff_hi:+.4f}] "
          f"-> {'resolves a winner' if resolves else 'includes zero: no resolved winner'}")
    pd.DataFrame([{
        "comparison": "itemknn_minus_ease", "metric": "ndcg", "k": 20,
        "diff": diff_mean, "ci_low": diff_lo, "ci_high": diff_hi,
        "resolves_winner": resolves, "n_sessions": len(ndcg20_per_session["itemknn"]),
    }]).to_csv(out / "paired_ci_full.csv", index=False)
    print(f"\nwrote {out}/metrics_full.csv, {out}/metrics_operational.csv, "
          f"{out}/cohort_flow.csv, {out}/paired_ci_full.csv")
    return table


def run_als_count_ablation(pairs: pd.DataFrame, out: Path) -> pd.DataFrame:
    """Does ALS repeat-count confidence weighting actually beat binary on this data?

    ALS's confidence is ``1 + alpha*value``; with counts an item viewed k times in a
    session gets ``1 + alpha*k``. ``alpha`` is therefore the hyperparameter that
    interacts with the count scale, so it is retuned *per variant* on the validation
    window (factors/regularization held at their Stage-1 tuned values); each variant's
    validation winner is then compared once on the test split with a paired bootstrap.
    The default ALS stays binary unless count weighting wins on *validation* — selection
    never touches test.
    """
    from reclab.tuning import validation_split

    val = validation_split(
        pairs, test_days=TEST_DAYS, val_days=VAL_DAYS,
        min_session_items=MIN_SESSION_ITEMS, min_item_sessions=MIN_ITEM_SESSIONS,
    )
    split = build_split(pairs)
    if val.train_counts is None or split.train_counts is None:
        raise RuntimeError("count matrix unavailable; n_events missing from the frame")

    factors, reg = 128, 0.1            # Stage-1 tuned values, held fixed
    alpha_grid = [160.0, 320.0, 640.0, 1280.0]

    def tune_alpha(train_matrix, use_counts):
        best_a, best_s = None, -float("inf")
        for a in alpha_grid:
            model = ALS(factors=factors, regularization=reg, alpha=a,
                        use_counts=use_counts, **ALS_FIXED).fit(train_matrix)
            s = float(evaluate(model, val, ks=(20,)).per_session[("ndcg", 20)].mean())
            if s > best_s:
                best_s, best_a = s, a
            del model
        return best_a, best_s

    print("\n=== ALS count-weighting ablation (binary vs count confidence) ===")
    a_bin, val_bin = tune_alpha(val.train, use_counts=False)
    a_cnt, val_cnt = tune_alpha(val.train_counts, use_counts=True)
    print(f"  validation NDCG@20: binary={val_bin:.4f} (alpha={a_bin:g})  "
          f"count={val_cnt:.4f} (alpha={a_cnt:g})")
    decision = "count" if val_cnt > val_bin else "binary"

    bin_model = ALS(factors=factors, regularization=reg, alpha=a_bin,
                    use_counts=False, **ALS_FIXED).fit(split.train)
    cnt_model = ALS(factors=factors, regularization=reg, alpha=a_cnt,
                    use_counts=True, **ALS_FIXED).fit(split.train_counts)
    res_bin, res_cnt = evaluate(bin_model, split, ks=(10, 20)), evaluate(cnt_model, split, ks=(10, 20))
    b20, c20 = res_bin.per_session[("ndcg", 20)], res_cnt.per_session[("ndcg", 20)]
    diff, lo, hi = paired_bootstrap_diff(c20, b20)
    resolves = not (lo <= 0.0 <= hi)
    print(f"  test NDCG@20: binary={b20.mean():.4f}  count={c20.mean():.4f}")
    print(f"  paired (count - binary) = {diff:+.4f} [{lo:+.4f}, {hi:+.4f}] "
          f"-> {'resolves a difference' if resolves else 'includes zero: no resolved difference'}")
    print(f"  validation selects: {decision}  (ALS default stays binary unless count wins)")

    table = pd.DataFrame([
        {"variant": "binary", "alpha": a_bin, "val_ndcg@20": val_bin,
         "test_ndcg@20": float(b20.mean()),
         "test_hit_rate@20": float(res_bin.per_session[("hit_rate", 20)].mean())},
        {"variant": "count", "alpha": a_cnt, "val_ndcg@20": val_cnt,
         "test_ndcg@20": float(c20.mean()),
         "test_hit_rate@20": float(res_cnt.per_session[("hit_rate", 20)].mean())},
    ])
    paired = pd.DataFrame([{
        "comparison": "count_minus_binary", "metric": "ndcg", "k": 20,
        "diff": diff, "ci_low": lo, "ci_high": hi, "resolves_winner": resolves,
        "validation_winner": decision, "n_sessions": res_bin.n_sessions,
    }])
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "als_count_ablation.csv", index=False)
    paired.to_csv(out / "als_count_ablation_paired.csv", index=False)
    print(f"\nwrote {out}/als_count_ablation.csv, {out}/als_count_ablation_paired.csv")
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
        description="Two-stage recommender study on RetailRocket: classical baselines, "
        "neural retrieval, reranking and serving (Stages 1-3).",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in [
        ("eda", "dataset summary + filter-threshold grid"),
        ("tune", "grid search on the validation window"),
        ("evaluate", "full-catalogue evaluation of all models"),
        ("als-count-ablation", "ALS repeat-count vs binary confidence weighting"),
        ("sampled", "sampled-negative evaluation + disagreement table"),
        ("protocols", "temporal vs leave-one-out comparison"),
        ("beyond", "coverage / Gini / popularity-bias metrics"),
        # Stage 2
        ("features", "item content-feature coverage (as-of cutoff)"),
        ("neural", "train + evaluate the two-tower and SASRec models"),
        ("ablations", "logQ on/off, id vs content, full vs sampled loss"),
        ("ceiling", "retrieval-ceiling analysis + blending + figure"),
        ("cold-start", "cold-item evaluation vs the category-popularity baseline"),
        # Stage 3
        ("evaluate-e2e", "end-to-end two-stage reranker evaluation"),
        ("ann-sweep", "HNSW recall/latency sweep + Pareto figure"),
        ("scaling", "catalogue-scaling sweep (EASE's memory/time wall, measured)"),
        ("build-service", "fit + persist the serving artifacts to outputs/serving"),
        ("serve", "launch the FastAPI recommender (uvicorn)"),
        ("latency", "per-stage serving latency harness"),
        ("stage3", "run only the Stage 3 pipeline (reranker, ANN, scaling, serving)"),
        ("seed-sensitivity", "training-seed robustness of the neural/reranker claims (P1-5)"),
        ("all", "regenerate every reported number (frozen tuned params; tuning is separate)"),
        ("stage2", "run only the Stage 2 pipeline (neural, ablations, ceiling, cold-start)"),
    ]:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--out", type=Path, default=Path("outputs"))

    args = parser.parse_args()

    if args.command == "serve":
        import uvicorn
        print("Serving on http://0.0.0.0:8000 (POST /recommend, GET /health). "
              "Run `reclab build-service` first if artifacts are missing.")
        uvicorn.run("reclab.serving.app:app", host="0.0.0.0", port=8000)
        return

    print(f"[reclab {args.command}] loading data...")
    pairs = load_pairs()

    if args.command == "eda":
        run_eda(pairs, args.out)
    elif args.command == "tune":
        run_tune(pairs, args.out)
    elif args.command == "evaluate":
        run_evaluate(pairs, args.out)
    elif args.command == "als-count-ablation":
        run_als_count_ablation(pairs, args.out)
    elif args.command == "sampled":
        run_sampled(pairs, args.out)
    elif args.command == "protocols":
        run_protocols(pairs, args.out)
    elif args.command == "beyond":
        run_beyond(pairs, args.out)
    elif args.command == "features":
        import reclab.stage2 as s2
        s2.run_features(build_split(pairs), args.out)
    elif args.command == "neural":
        import reclab.stage2 as s2
        s2.run_neural(build_split(pairs), args.out)
    elif args.command == "ablations":
        import reclab.stage2 as s2
        s2.run_ablations(build_split(pairs), args.out)
    elif args.command == "ceiling":
        run_ceiling_standalone(pairs, args.out)
    elif args.command == "cold-start":
        import reclab.stage2 as s2
        s2.run_cold_start(pairs, build_split(pairs), args.out)
    elif args.command == "stage2":
        run_stage2(pairs, args.out)
    elif args.command == "seed-sensitivity":
        from reclab.sensitivity import run_seed_sensitivity
        run_seed_sensitivity(pairs, args.out)
    elif args.command in ("evaluate-e2e", "ann-sweep", "scaling", "build-service",
                          "latency", "stage3"):
        import reclab.stage3 as s3
        fit = s3.fit_stage3(pairs)
        if args.command == "evaluate-e2e":
            s3.run_evaluate_e2e(fit, args.out)
        elif args.command == "ann-sweep":
            s3.run_ann_sweep(fit, args.out)
        elif args.command == "scaling":
            s3.run_scaling(fit, args.out)
        elif args.command == "build-service":
            s3.run_build_service(fit, args.out)
        elif args.command == "latency":
            service = s3.run_build_service(fit, args.out)
            s3.run_latency(service, args.out, eval_split=fit["eval_split"])
        elif args.command == "stage3":
            s3.run_stage3(fit, args.out)
    elif args.command == "all":
        run_eda(pairs, args.out)
        run_evaluate(pairs, args.out)
        run_als_count_ablation(pairs, args.out)
        run_sampled(pairs, args.out)
        run_protocols(pairs, args.out)
        run_beyond(pairs, args.out)
        run_stage2(pairs, args.out)
        import reclab.stage3 as s3
        s3.run_stage3(s3.fit_stage3(pairs), args.out)
        from reclab.sensitivity import run_seed_sensitivity
        run_seed_sensitivity(pairs, args.out)
        print("\n=== reproduce complete ===")


def run_ceiling_standalone(pairs: pd.DataFrame, out: Path) -> None:
    """Refit the retrievers, then compute the ceiling (slow; `all` reuses fits)."""
    import reclab.stage2 as s2

    split = build_split(pairs)
    features = s2.build_features(split)
    classical = {name: build_model(name).fit(split.train)
                 for name in ("itemknn", "ease", "als")}
    models = s2.fit_stage2_models(split, features)
    s2.run_ceiling(split, out, classical, models)


def run_stage2(pairs: pd.DataFrame, out: Path) -> None:
    """Full Stage 2 pipeline, fitting each model once and reusing the fits."""
    import reclab.stage2 as s2

    split = build_split(pairs)
    features = s2.build_features(split)
    print("\n[stage2] fitting neural models (each unique model once)...")
    models = s2.fit_stage2_models(split, features)
    print("[stage2] fitting classical retrievers for the ceiling...")
    classical = {name: build_model(name).fit(split.train)
                 for name in ("itemknn", "ease", "als")}

    s2.run_features(split, out, features=features)
    s2.run_neural(split, out, features=features, models=models)
    s2.run_ablations(split, out, features=features, models=models)
    s2.run_ceiling(split, out, classical, models)
    s2.run_cold_start(pairs, split, out, two_tower=models["two_tower"], features=features)


if __name__ == "__main__":
    main()
