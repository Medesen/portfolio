"""Stage 3 orchestration: reranker + end-to-end eval, ANN sweep, scaling, serving.

Fits each model once and threads the fits through every analysis, mirroring stage2.py.
The retriever for candidate generation is ALS — the best single retriever at N=500
(Recall@500 ≈ 0.88, from the Stage 2 ceiling) and an embedding model, so the same
vectors drive the ANN index and the serving path.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from reclab.evaluation.beyond_accuracy import beyond_accuracy_metrics
from reclab.features import build_item_features
from reclab.models import ALS
from reclab.ranking.dataset import build_ranker_frame, nested_split
from reclab.ranking.ranker import Reranker, e2e_top_k, evaluate_e2e

N_CANDIDATES = 500
ALS_PARAMS = dict(factors=128, regularization=1.0, alpha=40.0, iterations=15,
                  num_threads=4, seed=0)


def _log(msg):
    print(f"[stage3] {msg}", flush=True)


def fit_stage3(pairs, n_candidates=N_CANDIDATES):
    """Nested pipeline: retrievers, ranker frames, and the trained rerankers.

    Returns a dict with everything the analyses reuse."""
    ranker_train, eval_split = nested_split(pairs, test_days=28)
    _log(f"nested split: A={ranker_train.n_train_sessions} B={ranker_train.n_test_sessions} "
         f"eval-train={eval_split.n_train_sessions} C={eval_split.n_test_sessions} "
         f"(T1={ranker_train.cutoff.date()} T2={eval_split.cutoff.date()})")

    feats_A = build_item_features(ranker_train.cutoff, ranker_train.item_ids)
    feats_T2 = build_item_features(eval_split.cutoff, eval_split.item_ids)

    t = time.time()
    retr_A = ALS(**ALS_PARAMS).fit(ranker_train.train)
    retr_T2 = ALS(**ALS_PARAMS).fit(eval_split.train)
    _log(f"fitted A/T2 retrievers in {time.time()-t:.0f}s; building ranker frames...")

    t = time.time()
    train_frame = build_ranker_frame(retr_A, ranker_train, feats_A, n_candidates,
                                     negatives_per_session=50, seed=0)
    eval_frame = build_ranker_frame(retr_T2, eval_split, feats_T2, n_candidates)
    ceiling = eval_frame.y.sum() / eval_frame.n_sessions
    _log(f"frames built in {time.time()-t:.0f}s; train pos-rate={train_frame.positive_rate:.4f}, "
         f"eval retrieval ceiling (target in {n_candidates})={ceiling:.4f}")

    rerankers = {}
    for obj in ("lambdarank", "binary"):
        t = time.time()
        rerankers[obj] = Reranker(objective=obj, seed=0).fit(train_frame)
        _log(f"trained {obj} reranker in {time.time()-t:.0f}s")

    return dict(ranker_train=ranker_train, eval_split=eval_split, feats_T2=feats_T2,
                retr_T2=retr_T2, eval_frame=eval_frame, rerankers=rerankers,
                retrieval_ceiling=float(ceiling))


def run_evaluate_e2e(fit: dict, out: Path) -> None:
    eval_split, eval_frame = fit["eval_split"], fit["eval_frame"]
    rows = []
    print("\n=== End-to-end two-stage evaluation ===")
    for obj, rk in fit["rerankers"].items():
        res = evaluate_e2e(rk, eval_frame, eval_split.test_target, ks=(10, 20),
                           label=f"two_stage_{obj}")
        rows.append(res)
    table = pd.concat(rows, ignore_index=True).drop_duplicates(["model", "metric", "k"])

    # Beyond-accuracy for the two-stage system (does reranking collapse coverage?).
    item_pop = np.asarray(eval_split.train.sum(axis=0)).ravel()
    top20 = e2e_top_k(fit["rerankers"]["lambdarank"], eval_frame, k=20)
    top_k_arr = np.vstack([top20[s] for s in sorted(top20)])
    ba = beyond_accuracy_metrics(top_k_arr, eval_split.n_items, item_pop, k=20)

    def g(m, metric, k):
        r = table[(table.model == m) & (table.metric == metric) & (table.k == k)]
        return r.value.iloc[0] if len(r) else float("nan")

    print(f"  retrieval ceiling (target in {N_CANDIDATES} candidates): {fit['retrieval_ceiling']:.4f}")
    for m in ("retrieval_only", "two_stage_binary", "two_stage_lambdarank"):
        print(f"  {m:22s} NDCG@20={g(m,'ndcg',20):.4f} HR@20={g(m,'hit_rate',20):.4f} "
              f"MRR@20={g(m,'mrr',20):.4f}")
    print(f"  reference: best single-stage ItemKNN NDCG@20=0.3194 (Stage 1)")
    print(f"  two-stage (lambdarank) coverage@20={ba['coverage']:.3f} "
          f"gini@20={ba['gini']:.3f} pop%@20={ba['mean_pop_percentile']:.3f}")
    print("  feature importance (gain):",
          fit["rerankers"]["lambdarank"].feature_importance().round(0).to_dict())

    out.mkdir(parents=True, exist_ok=True)
    table["retrieval_ceiling"] = fit["retrieval_ceiling"]
    table.to_csv(out / "metrics_e2e.csv", index=False)
    pd.DataFrame([{"model": "two_stage_lambdarank", "k": 20, **ba}]).to_csv(
        out / "e2e_beyond.csv", index=False)
    fit["rerankers"]["lambdarank"].feature_importance().rename("gain").to_csv(
        out / "ranker_feature_importance.csv")
    print(f"\nwrote {out}/metrics_e2e.csv, e2e_beyond.csv, ranker_feature_importance.csv")


def run_ann_sweep(fit: dict, out: Path) -> None:
    from reclab.evaluation.full_catalogue import history_chunk
    from reclab.serving.ann import sweep_ann

    eval_split, retr = fit["eval_split"], fit["retr_T2"]
    item_vecs = retr.item_factors_
    user_vecs = retr.user_embeddings(
        history_chunk(eval_split, 0, eval_split.n_test_sessions).matrix)

    print("\n=== ANN sweep (HNSW over ALS embeddings) ===")
    stats, exact = sweep_ann(item_vecs, user_vecs, k=200,
                             ef_searches=(16, 32, 64, 128, 256), n_latency=1500, seed=0)
    # Exact brute-force per-query latency, the baseline ANN is compared against.
    rng = np.random.default_rng(0)
    sample = user_vecs[rng.integers(0, len(user_vecs), 1500)]
    ex_ms = []
    for v in sample:
        t = time.perf_counter()
        np.argpartition(-(v @ item_vecs.T), 199)[:200]
        ex_ms.append((time.perf_counter() - t) * 1000)
    exact_p50, exact_p95 = float(np.percentile(ex_ms, 50)), float(np.percentile(ex_ms, 95))

    rows = [{"ef_search": s.ef_search, "recall_at_200": s.recall_at_k,
             "p50_ms": s.p50_ms, "p95_ms": s.p95_ms, "p99_ms": s.p99_ms,
             "build_s": s.build_s} for s in stats]
    table = pd.DataFrame(rows)
    table["exact_p50_ms"], table["exact_p95_ms"] = exact_p50, exact_p95
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"  exact brute-force per-query: p50={exact_p50:.3f}ms p95={exact_p95:.3f}ms "
          f"— already sub-millisecond at {len(item_vecs):,} items, so ANN is unnecessary here")

    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "ann_sweep.csv", index=False)
    _plot_ann(table, exact_p95, out / "ann_recall_latency.png")
    print(f"\nwrote {out}/ann_sweep.csv, ann_recall_latency.png")


def _plot_ann(table, exact_p95, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(table["p95_ms"], table["recall_at_200"], "o-", label="HNSW (ANN)")
    for _, r in table.iterrows():
        ax.annotate(f"ef={int(r.ef_search)}", (r.p95_ms, r.recall_at_200),
                    textcoords="offset points", xytext=(6, -8), fontsize=8)
    ax.axvline(exact_p95, color="grey", ls="--", label=f"exact p95 = {exact_p95:.2f} ms")
    ax.set_xlabel("p95 query latency (ms, single-threaded)")
    ax.set_ylabel("Recall@200 vs exact")
    ax.set_title("ANN recall vs latency — unnecessary at 14k items, shown for the shape")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def run_scaling(fit: dict, out: Path) -> None:
    from reclab.scaling import catalogue_scaling_sweep, extrapolate_ease_memory

    print("\n=== Catalogue-scaling sweep (EASE vs ALS fit time) ===")
    sweep = catalogue_scaling_sweep(fit["eval_split"],
                                    sizes=(2000, 5000, 8000, 11000, 13754))
    print(sweep.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    extra = extrapolate_ease_memory()
    print("\nEASE memory extrapolation (arithmetic, beyond this catalogue):")
    print(extra.to_string(index=False))

    out.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(out / "scaling_sweep.csv", index=False)
    extra.to_csv(out / "scaling_extrapolation.csv", index=False)
    _plot_scaling(sweep, out / "scaling_fit_time.png")
    print(f"\nwrote {out}/scaling_sweep.csv, scaling_extrapolation.csv, scaling_fit_time.png")


def _plot_scaling(sweep, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(sweep["n_items"], sweep["ease_fit_s"], "o-", label="EASE (dense solve, ~O(K³))")
    ax.plot(sweep["n_items"], sweep["als_fit_s"], "s-", label="ALS (embeddings, ~linear)")
    ax.set_xlabel("catalogue size K (items)")
    ax.set_ylabel("fit time (s)")
    ax.set_title("Fit time vs catalogue size — EASE's wall, measured")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def run_build_service(fit: dict, out: Path, service_dir: Path | None = None):
    from reclab.serving.pipeline import build_service

    service_dir = service_dir or (out / "serving")
    service = build_service(fit["eval_split"], fit["feats_T2"], fit["retr_T2"],
                            fit["rerankers"]["lambdarank"], n_candidates=N_CANDIDATES)
    service.save(service_dir)
    _log(f"built + persisted service to {service_dir}")
    return service


def run_latency(service, out: Path, eval_split=None, n_requests: int = 300) -> None:
    """Per-stage latency over sample requests drawn from real session histories."""
    print("\n=== Per-stage serving latency ===")
    rng = np.random.default_rng(0)
    # Build sample histories from the test prefixes (real, warm sessions).
    if eval_split is not None:
        prefixes = eval_split.test_prefix
        ids = np.asarray(eval_split.item_ids)
        sample_rows = rng.integers(0, prefixes.shape[0], n_requests)
        histories = [[int(ids[c]) for c in prefixes[r].indices] for r in sample_rows]
        histories = [h for h in histories if h] or [[int(ids[0])]]
    else:
        histories = [[int(service.item_ids[0])]] * n_requests

    stages = {}
    for h in histories:
        rec = service.recommend(h, k=10)
        for stage, ms in rec.timings_ms.items():
            stages.setdefault(stage, []).append(ms)

    rows = []
    for stage, vals in stages.items():
        v = np.array(vals)
        rows.append({"stage": stage, "p50_ms": float(np.percentile(v, 50)),
                     "p95_ms": float(np.percentile(v, 95)),
                     "share_pct": 100 * v.mean() / np.mean(stages["total"])})
    table = pd.DataFrame(rows)
    order = ["user_embedding", "retrieval", "features", "ranking", "filter_topk", "total"]
    table["_o"] = table["stage"].map({s: i for i, s in enumerate(order)})
    table = table.sort_values("_o").drop(columns="_o")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "latency.csv", index=False)
    print(f"\nwrote {out}/latency.csv")


def run_stage3(fit: dict, out: Path) -> None:
    run_evaluate_e2e(fit, out)
    run_ann_sweep(fit, out)
    run_scaling(fit, out)
    service = run_build_service(fit, out)
    run_latency(service, out, eval_split=fit["eval_split"])
