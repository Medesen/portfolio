#!/usr/bin/env python3
"""
Retrieval ablation of the default query pipeline.

The headline chunking comparison in the README isolates chunking strategies
under plain semantic retrieval. This script answers the complementary
question: what do the default pipeline's other components contribute? It
evaluates the stepwise ladder from plain dense retrieval up to the shipped
default query path, one component at a time:

    dense         semantic-only retrieval (the published table's configuration)
    hybrid        + BM25 keyword fusion via Reciprocal Rank Fusion
    hybrid_rerank + cross-encoder reranking of the fused candidates
    full_default  + LLM query rewriting  (== the shipped `make query` path)

A dense+rerank arm is deliberately absent: in this architecture the reranker
only runs on the hybrid path, so that combination does not exist as a shipped
configuration.

Every arm returns RETRIEVAL_DEPTH (20) chunks — the same depth as the
published chunking evaluation — so all arms are scored on identically deep
lists. (The shipped default cuts to final_top_k=10 for display; that cut is
downstream of the ranking being measured.)

Metrics per question: Recall@5/10, MRR, NDCG@10 via the same metrics
implementation as the main evaluation, plus wall-clock retrieval latency
(including rewrite time where enabled). Uncertainty: paired bootstrap over
questions (default 10,000 resamples, seed 42) for each arm's mean and for
each ladder step's delta.

Usage (local, with Ollama running and the index built):
    python scripts/ablation_eval.py --ollama-url http://localhost:11434
    python scripts/ablation_eval.py --max-questions 5   # smoke run
"""

import argparse
import copy
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import calculate_all_metrics, extract_doc_ids_from_chunks  # noqa: E402
from src.evaluation.test_loader import TestLoader  # noqa: E402
from src.generation.llm_client import OllamaClient  # noqa: E402
from src.retrieval.bm25_index import BM25Index  # noqa: E402
from src.retrieval.embedder import Embedder  # noqa: E402
from src.retrieval.query_processor import QueryProcessor  # noqa: E402
from src.retrieval.query_rewriter import QueryRewriter  # noqa: E402
from src.retrieval.vector_store import VectorStore  # noqa: E402
from src.utils.config import Config, load_config  # noqa: E402

RETRIEVAL_DEPTH = 20  # chunks returned per arm; matches the published evaluation
K_VALUES = [5, 10, 20]
REPORTED = ["recall@10", "mrr", "ndcg@10"]  # headline metrics in the README
ARMS = ["dense", "hybrid", "hybrid_rerank", "full_default"]


def build_arm_config(base_config: Config, arm: str) -> Config:
    """Return a Config for one ablation arm, derived from the base config."""
    cfg = copy.deepcopy(base_config._config)
    cfg.setdefault("retrieval", {})
    cfg.setdefault("reranking", {})
    cfg.setdefault("query_rewriting", {})

    cfg["retrieval"]["search_mode"] = "semantic" if arm == "dense" else "hybrid"
    cfg["reranking"]["enabled"] = arm in ("hybrid_rerank", "full_default")
    cfg["query_rewriting"]["enabled"] = arm == "full_default"
    return Config(cfg, base_config.base_path)


def build_processor(
    arm: str, arm_config: Config, embedder, vector_store, bm25_index, ollama_url: str
) -> QueryProcessor:
    """Assemble the QueryProcessor for one arm (mirrors main.py's cmd_query)."""
    rewriter = None
    if arm_config.get("query_rewriting.enabled", False):
        llm_client = OllamaClient(
            base_url=ollama_url,
            model=arm_config.get("generation.model", "llama3.2:3b"),
            timeout=arm_config.get("query_rewriting.timeout", 30),
            logger_name=f"ablation_rewrite_llm_{arm}",
        )
        rewriter = QueryRewriter(
            llm_client=llm_client, config=arm_config, logger_name=f"ablation_rewriter_{arm}"
        )

    return QueryProcessor(
        config=arm_config,
        embedder=embedder,
        vector_store=vector_store,
        bm25_index=bm25_index if arm != "dense" else None,
        query_rewriter=rewriter,
        logger_name=f"ablation_qp_{arm}",
    )


def run_arm(arm: str, processor: QueryProcessor, questions, strategy: str) -> list[dict]:
    """Score every test question under one arm; return per-question records."""
    records = []

    # Warm-up (model loads, index mmap) so timed queries measure steady state
    if arm == "dense":
        processor.process_query("warm up", strategy=strategy, top_k=RETRIEVAL_DEPTH)
    else:
        processor.process_query_hybrid("warm up", strategy=strategy, top_k=RETRIEVAL_DEPTH)

    for i, q in enumerate(questions, 1):
        start = time.time()
        if arm == "dense":
            result = processor.process_query(
                q.question, strategy=strategy, top_k=RETRIEVAL_DEPTH
            )
        else:
            result = processor.process_query_hybrid(
                q.question, strategy=strategy, top_k=RETRIEVAL_DEPTH
            )
        latency = time.time() - start

        doc_ids = extract_doc_ids_from_chunks(result.get("results", []))
        metrics = calculate_all_metrics(doc_ids, q.relevant_doc_ids, K_VALUES)
        metrics["latency_s"] = latency
        records.append({"question_id": q.id, "metrics": metrics})
        print(
            f"  [{arm}] {i}/{len(questions)} {q.id}: "
            f"R@10={metrics['recall@10']:.2f} MRR={metrics['mrr']:.2f} "
            f"({latency:.2f}s)",
            flush=True,
        )
    return records


def bootstrap_ci(values: np.ndarray, rng, n_boot: int) -> tuple[float, float]:
    """Percentile 95% CI for the mean of `values` via bootstrap."""
    n = len(values)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_delta_ci(a: np.ndarray, b: np.ndarray, rng, n_boot: int) -> tuple[float, float]:
    """Percentile 95% CI for mean(b - a) with paired (same-question) resampling."""
    return bootstrap_ci(b - a, rng, n_boot)


def summarize(all_records: dict, n_boot: int, seed: int) -> dict:
    """Aggregate per-question records into means, CIs, and ladder-step deltas."""
    rng = np.random.default_rng(seed)
    summary: dict = {"arms": {}, "deltas": {}}

    per_metric: dict[str, dict[str, np.ndarray]] = {}
    for arm, records in all_records.items():
        arm_summary = {}
        for metric in REPORTED + ["latency_s"]:
            vals = np.array([r["metrics"][metric] for r in records])
            per_metric.setdefault(metric, {})[arm] = vals
            lo, hi = bootstrap_ci(vals, rng, n_boot)
            arm_summary[metric] = {
                "mean": float(vals.mean()),
                "ci95": [lo, hi],
                **({"median": float(np.median(vals))} if metric == "latency_s" else {}),
            }
        summary["arms"][arm] = arm_summary

    # Each ladder step's marginal contribution, plus full-vs-dense overall
    steps = list(zip(ARMS[:-1], ARMS[1:])) + [("dense", "full_default")]
    for base, arm in steps:
        if base == arm or base not in all_records or arm not in all_records:
            continue
        key = f"{arm} - {base}"
        summary["deltas"][key] = {}
        for metric in REPORTED:
            a, b = per_metric[metric][base], per_metric[metric][arm]
            lo, hi = paired_delta_ci(a, b, rng, n_boot)
            summary["deltas"][key][metric] = {"mean": float((b - a).mean()), "ci95": [lo, hi]}
    return summary


def format_markdown(summary: dict, n_questions: int, n_boot: int) -> str:
    """Render the summary as the markdown tables used in the README."""
    lines = [
        f"Ablation of the default retrieval pipeline ({n_questions} questions, "
        f"fixed chunking, depth {RETRIEVAL_DEPTH}; 95% CIs from {n_boot:,} "
        "paired bootstrap resamples)",
        "",
        "| Configuration | Recall@10 | MRR | NDCG@10 | Median latency |",
        "|---|---|---|---|---|",
    ]
    labels = {
        "dense": "Dense (semantic-only)",
        "hybrid": "+ BM25 fusion (RRF)",
        "hybrid_rerank": "+ cross-encoder reranking",
        "full_default": "+ LLM query rewriting (= shipped default)",
    }
    for arm in ARMS:
        if arm not in summary["arms"]:
            continue
        s = summary["arms"][arm]

        def cell(metric):
            m = s[metric]
            return f"{m['mean']:.3f} [{m['ci95'][0]:.3f}, {m['ci95'][1]:.3f}]"

        lat = s["latency_s"]
        lines.append(
            f"| {labels[arm]} | {cell('recall@10')} | {cell('mrr')} | "
            f"{cell('ndcg@10')} | {lat['median']:.2f}s |"
        )

    lines += ["", "Per-step deltas (paired):", "", "| Step | ΔRecall@10 | ΔMRR | ΔNDCG@10 |", "|---|---|---|---|"]
    for key, metrics in summary["deltas"].items():
        def dcell(metric):
            m = metrics[metric]
            return f"{m['mean']:+.3f} [{m['ci95'][0]:+.3f}, {m['ci95'][1]:+.3f}]"

        lines.append(f"| {key} | {dcell('recall@10')} | {dcell('mrr')} | {dcell('ndcg@10')} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--config", default=None, help="Path to config YAML")
    parser.add_argument("--strategy", default="fixed", help="Chunking strategy to query")
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--ollama-url",
        default=None,
        help="Override generation.ollama_base_url (e.g. http://localhost:11434)",
    )
    parser.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    base_config = load_config(args.config)
    ollama_url = args.ollama_url or base_config.get(
        "generation.ollama_base_url", "http://ollama:11434"
    )

    questions = TestLoader(base_config).load_test_set()
    if args.max_questions:
        questions = questions[: args.max_questions]
    print(f"Loaded {len(questions)} test questions")

    # Shared read-only components (identical across arms)
    embedder = Embedder(
        model_name=base_config.get("embeddings.model", "all-MiniLM-L6-v2"),
        device=base_config.get("embeddings.device", "cpu"),
        batch_size=base_config.get("embeddings.batch_size", 32),
    )
    vector_store_dir = base_config.get_path("paths.vector_store_dir")
    vector_store = VectorStore(persist_directory=vector_store_dir)
    bm25_index = BM25Index(persist_directory=vector_store_dir / "bm25")

    all_records = {}
    for arm in args.arms:
        print(f"\n=== Arm: {arm} ===", flush=True)
        arm_config = build_arm_config(base_config, arm)
        processor = build_processor(
            arm, arm_config, embedder, vector_store, bm25_index, ollama_url
        )
        all_records[arm] = run_arm(arm, processor, questions, args.strategy)

    summary = summarize(all_records, args.bootstrap, args.seed)
    markdown = format_markdown(summary, len(questions), args.bootstrap)
    print("\n" + markdown)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output:
        out_path = Path(args.output)
    else:
        results_dir = base_config.base_path / "data/evaluation/results"
        results_dir.mkdir(parents=True, exist_ok=True)
        out_path = results_dir / f"ablation_{timestamp}.json"
    payload = {
        "timestamp": timestamp,
        "n_questions": len(questions),
        "strategy": args.strategy,
        "retrieval_depth": RETRIEVAL_DEPTH,
        "bootstrap_resamples": args.bootstrap,
        "seed": args.seed,
        "summary": summary,
        "per_question": all_records,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with open(out_path.with_suffix(".md"), "w", encoding="utf-8") as f:
        f.write(markdown + "\n")
    print(f"\nResults saved to {out_path} (+ .md)")


if __name__ == "__main__":
    main()
