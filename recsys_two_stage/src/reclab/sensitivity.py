"""Training-seed sensitivity for the stochastic Stage 2/3 claims (P1-5).

The bootstrap intervals elsewhere resample *sessions* — they quantify evaluation-sample
uncertainty and say nothing about *training* variance. But the neural retrievers, the
reranker, and their negative sampling are all seeded, so a single seed can make a small
margin look resolved by luck. This retrains the reported models across several seeds and
reports the mean and range, separating training variance from the session bootstrap, so
the headline neural/reranker claims are shown seed-robust rather than asserted.

Scope is deliberately the *reported* configurations only (the main two-tower and its
logQ-off twin, SASRec under each loss, and the two rerankers), not the whole ablation
grid — the point is the claims, and the deterministic pieces (the ALS retrievers, the
evaluation frame) are built once and reused across seeds.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from reclab.evaluation.full_catalogue import evaluate


def run_seed_sensitivity(pairs, out: Path, seeds: tuple[int, ...] = (0, 1, 2)) -> pd.DataFrame:
    import reclab.stage2 as s2
    from reclab.features import build_item_features
    from reclab.models import ALS
    from reclab.ranking.dataset import FEATURES, build_ranker_frame, nested_split
    from reclab.ranking.ranker import Reranker, e2e_per_session
    from reclab.splitting import temporal_split

    split = temporal_split(pairs, test_days=28, min_session_items=2, min_item_sessions=10)
    features = build_item_features(split.cutoff, split.item_ids)

    # Stage-3 windows: the retrievers and the evaluation frame are deterministic (fixed ALS
    # seed, no sampling), so they are built once and reused; only the training seeds vary.
    ranker_train, eval_split = nested_split(pairs, test_days=28)
    feats_A = build_item_features(ranker_train.cutoff, ranker_train.item_ids)
    feats_T2 = build_item_features(eval_split.cutoff, eval_split.item_ids)
    als_params = dict(factors=128, regularization=0.1, alpha=640.0, iterations=15,
                      num_threads=4, seed=0)
    retr_A = ALS(**als_params).fit(ranker_train.train)
    retr_T2 = ALS(**als_params).fit(eval_split.train)
    eval_frame = build_ranker_frame(retr_T2, eval_split, feats_T2, 500)
    targets = eval_split.test_target

    def e2e_ndcg(scores):
        return float(e2e_per_session(scores, eval_frame, targets, "ndcg", 20).mean())

    retr_ndcg = e2e_ndcg(eval_frame.X[:, FEATURES.index("retr_score")])

    def full_ndcg(model):
        return float(evaluate(model, split, ks=(20,)).per_session[("ndcg", 20)].mean())

    print(f"\n=== Training-seed sensitivity (P1-5, seeds={list(seeds)}) ===")
    rows = []
    for seed in seeds:
        tt = s2.fit_two_tower(split, features, seed=seed)
        tt_no_logq = s2.fit_two_tower(split, features, logq_correction=False, seed=seed)
        sr_full = s2.fit_sasrec(split, loss="full_softmax", seed=seed)
        sr_samp = s2.fit_sasrec(split, loss="sampled_bce", seed=seed)
        train_frame = build_ranker_frame(retr_A, ranker_train, feats_A, 500,
                                         negatives_per_session=50, seed=seed)
        lam = Reranker(objective="lambdarank", seed=seed).fit(train_frame)
        binr = Reranker(objective="binary", seed=seed).fit(train_frame)

        vals = {
            "two_tower_logq_on": full_ndcg(tt),
            "two_tower_logq_off": full_ndcg(tt_no_logq),
            "sasrec_full_softmax": full_ndcg(sr_full),
            "sasrec_sampled_bce": full_ndcg(sr_samp),
            "two_stage_lambdarank": e2e_ndcg(lam.predict(eval_frame.X)),
            "two_stage_binary": e2e_ndcg(binr.predict(eval_frame.X)),
            "retrieval_only": retr_ndcg,
        }
        for metric, v in vals.items():
            rows.append({"seed": seed, "metric": metric, "ndcg@20": v})
        print(f"  seed {seed}: two_tower={vals['two_tower_logq_on']:.4f}  "
              f"sasrec_full={vals['sasrec_full_softmax']:.4f}  "
              f"sasrec_sampled={vals['sasrec_sampled_bce']:.4f}  "
              f"two_stage={vals['two_stage_lambdarank']:.4f}")

    per_seed = pd.DataFrame(rows)
    summary = (per_seed.groupby("metric")["ndcg@20"]
               .agg(["mean", "std", "min", "max"]).reset_index())

    # Headline deltas paired *within* each seed, then aggregated — the seed-robust form of
    # "logQ helps", "listwise beats pointwise", "reranking beats the retriever".
    wide = per_seed.pivot(index="seed", columns="metric", values="ndcg@20")
    deltas = pd.DataFrame({
        "logq_on_minus_off": wide["two_tower_logq_on"] - wide["two_tower_logq_off"],
        "lambdarank_minus_binary": wide["two_stage_lambdarank"] - wide["two_stage_binary"],
        "rerank_minus_retrieval": wide["two_stage_lambdarank"] - wide["retrieval_only"],
    })
    delta_summary = (deltas.agg(["mean", "std", "min", "max"]).T
                     .reset_index().rename(columns={"index": "delta"}))

    out.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(out / "seed_sensitivity.csv", index=False)
    summary.to_csv(out / "seed_sensitivity_summary.csv", index=False)
    delta_summary.to_csv(out / "seed_sensitivity_deltas.csv", index=False)
    print("\nper-metric NDCG@20 across seeds (mean/std/min/max):")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nheadline deltas across seeds:")
    print(delta_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nwrote {out}/seed_sensitivity.csv, seed_sensitivity_summary.csv, "
          "seed_sensitivity_deltas.csv")
    return per_seed
