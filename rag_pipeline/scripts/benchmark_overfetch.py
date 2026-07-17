#!/usr/bin/env python3
"""
Benchmark script for evaluating cross-encoder reranking with different overfetch values.

This script tests different overfetch_k values (number of documents retrieved before
reranking) and measures both retrieval quality metrics and latency.

Usage:
    python scripts/benchmark_overfetch.py [--strategy fixed] [--max-questions 10]
    python scripts/benchmark_overfetch.py --overfetch-values 30 50 60 80
    python scripts/benchmark_overfetch.py --output results.json
"""

from __future__ import annotations
import argparse
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.utils.logger import get_logger
from src.retrieval.embedder import Embedder
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_index import BM25Index
from src.retrieval.hybrid_searcher import HybridSearcher
from src.retrieval.reranker import CrossEncoderReranker
from src.evaluation.test_loader import TestLoader
from src.evaluation import metrics as retrieval_metrics
from src.evaluation.metrics import extract_doc_ids_from_chunks


logger = get_logger("benchmark_overfetch")


def run_benchmark(
    config,
    overfetch_values: List[int],
    strategy: str,
    final_top_k: int,
    max_questions: Optional[int] = None
) -> Dict[str, Any]:
    """
    Run benchmark comparing different overfetch_k values.
    
    Args:
        config: Configuration object
        overfetch_values: List of overfetch_k values to test
        strategy: Chunking strategy to use
        final_top_k: Number of results to return after reranking
        max_questions: Maximum number of questions to evaluate
        
    Returns:
        Benchmark results dictionary
    """
    logger.info("=" * 70)
    logger.info("CROSS-ENCODER RERANKING OVERFETCH BENCHMARK")
    logger.info("=" * 70)
    logger.info(f"Strategy: {strategy}")
    logger.info(f"Final top_k: {final_top_k}")
    logger.info(f"Overfetch values to test: {overfetch_values}")
    logger.info("=" * 70)
    
    # Initialize components
    logger.info("\nInitializing components...")
    
    # Embedder
    embedder = Embedder(
        model_name=config.get("embeddings.model", "all-MiniLM-L6-v2"),
        device=config.get("embeddings.device", "cpu"),
        batch_size=config.get("embeddings.batch_size", 32)
    )
    
    # Vector store
    vector_store = VectorStore(
        persist_directory=config.get_path("paths.vector_store_dir"),
        logger_name="benchmark_vector_store"
    )
    
    # BM25 index — lives under the "bm25" subdirectory of the vector store,
    # matching where the Indexer writes it (see Indexer.__init__ / main.py)
    bm25_index = BM25Index(
        persist_directory=config.get_path("paths.vector_store_dir") / "bm25"
    )
    if not bm25_index.load_index(strategy):
        logger.error(f"Failed to load BM25 index for strategy '{strategy}'")
        sys.exit(1)
    
    # Reranker
    reranker = CrossEncoderReranker(
        model_name=config.get("reranking.model", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        device=config.get("reranking.device", config.get("embeddings.device", "cpu")),
        batch_size=config.get("reranking.batch_size", 32)
    )
    
    # Hybrid searcher with reranker
    hybrid_searcher = HybridSearcher(
        vector_store=vector_store,
        bm25_index=bm25_index,
        embedder=embedder,
        alpha=config.get("retrieval.hybrid_alpha", 0.7),
        rrf_k=config.get("retrieval.rrf_k", 60),
        reranker=reranker
    )
    
    # Load test questions
    test_loader = TestLoader(config)
    questions = test_loader.load_test_set()
    
    if max_questions:
        questions = questions[:max_questions]
    
    logger.info(f"Loaded {len(questions)} test questions")
    
    # Metrics calculator
    
    # Run benchmark for each overfetch value
    results_by_overfetch = {}
    
    for overfetch_k in overfetch_values:
        logger.info(f"\n{'='*70}")
        logger.info(f"Testing overfetch_k = {overfetch_k}")
        logger.info(f"{'='*70}")
        
        per_question_results = []
        rerank_times_ms = []
        total_times_ms = []
        
        for i, question in enumerate(questions, 1):
            query_preview = question.question[:50] + "..." if len(question.question) > 50 else question.question
            logger.info(f"  [{i}/{len(questions)}] {query_preview}")
            
            try:
                # Run hybrid search with reranking
                start_time = time.time()
                result = hybrid_searcher.search(
                    query=question.question,
                    strategy=strategy,
                    top_k=final_top_k,
                    overfetch_k=overfetch_k,
                    rerank_top_k=final_top_k
                )
                total_time_ms = (time.time() - start_time) * 1000
                
                # Extract timing
                rerank_time_ms = result['metadata']['timing'].get('rerank_ms', 0)
                rerank_times_ms.append(rerank_time_ms)
                total_times_ms.append(total_time_ms)
                
                # Extract retrieved doc IDs
                retrieved_chunks = result['results']
                retrieved_doc_ids = extract_doc_ids_from_chunks(retrieved_chunks)
                
                # Calculate metrics
                k_values = [5, 10, final_top_k] if final_top_k not in [5, 10] else [5, 10]
                metrics = retrieval_metrics.calculate_all_metrics(
                    retrieved_doc_ids,
                    question.relevant_doc_ids,
                    k_values
                )
                
                # Topic coverage
                topic_coverage = retrieval_metrics.topic_coverage(
                    retrieved_chunks,
                    question.expected_topics
                )
                metrics['topic_coverage'] = topic_coverage
                
                per_question_results.append({
                    'question_id': question.id,
                    'metrics': metrics,
                    'rerank_time_ms': rerank_time_ms,
                    'total_time_ms': total_time_ms,
                    'reranked': result['metadata'].get('reranked', False)
                })
                
            except Exception as e:
                logger.error(f"    Error: {e}")
                per_question_results.append({
                    'question_id': question.id,
                    'error': str(e)
                })
        
        # Aggregate results for this overfetch value
        successful_results = [r for r in per_question_results if 'metrics' in r]
        
        if successful_results:
            # Average metrics
            avg_metrics = retrieval_metrics.average_metrics(
                [r['metrics'] for r in successful_results]
            )
            
            # Average timing
            avg_rerank_time_ms = sum(rerank_times_ms) / len(rerank_times_ms) if rerank_times_ms else 0
            avg_total_time_ms = sum(total_times_ms) / len(total_times_ms) if total_times_ms else 0
            
            results_by_overfetch[overfetch_k] = {
                'overfetch_k': overfetch_k,
                'num_questions': len(successful_results),
                'avg_metrics': avg_metrics,
                'timing': {
                    'avg_rerank_time_ms': round(avg_rerank_time_ms, 2),
                    'avg_total_time_ms': round(avg_total_time_ms, 2),
                    'min_rerank_time_ms': round(min(rerank_times_ms), 2) if rerank_times_ms else 0,
                    'max_rerank_time_ms': round(max(rerank_times_ms), 2) if rerank_times_ms else 0,
                },
                'per_question': per_question_results
            }
            
            # Print summary for this overfetch value
            def _fmt(metric_key: str, fallback_key: str) -> str:
                value = avg_metrics.get(metric_key, avg_metrics.get(fallback_key))
                return f"{value:.4f}" if isinstance(value, (int, float)) else "N/A"

            logger.info(f"\n  Results for overfetch_k={overfetch_k}:")
            logger.info(f"    Recall@{final_top_k}: {_fmt(f'recall@{final_top_k}', 'recall@10')}")
            logger.info(f"    MRR: {avg_metrics.get('mrr', 0):.4f}")
            logger.info(f"    NDCG@{final_top_k}: {_fmt(f'ndcg@{final_top_k}', 'ndcg@10')}")
            logger.info(f"    Topic Coverage: {avg_metrics.get('topic_coverage', 0):.4f}")
            logger.info(f"    Avg Rerank Time: {avg_rerank_time_ms:.1f}ms")
            logger.info(f"    Avg Total Time: {avg_total_time_ms:.1f}ms")
    
    # Build comparison summary
    comparison = build_comparison(results_by_overfetch, final_top_k)
    
    return {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'strategy': strategy,
            'final_top_k': final_top_k,
            'overfetch_values': overfetch_values,
            'num_questions': len(questions),
            'reranker_model': config.get("reranking.model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        },
        'results_by_overfetch': results_by_overfetch,
        'comparison': comparison
    }


def build_comparison(results_by_overfetch: Dict[int, Dict], final_top_k: int) -> Dict[str, Any]:
    """Build comparison summary across overfetch values."""
    
    if not results_by_overfetch:
        return {}
    
    comparison = {
        'metrics_comparison': {},
        'timing_comparison': {},
        'best_overfetch_by_metric': {}
    }
    
    # Key metrics to compare
    key_metrics = ['mrr', f'recall@{final_top_k}', f'ndcg@{final_top_k}', 'topic_coverage']
    # Fallback if final_top_k not in standard values
    if f'recall@{final_top_k}' not in results_by_overfetch.get(list(results_by_overfetch.keys())[0], {}).get('avg_metrics', {}):
        key_metrics = ['mrr', 'recall@10', 'ndcg@10', 'topic_coverage']
    
    for metric in key_metrics:
        metric_values = {}
        for overfetch_k, result in results_by_overfetch.items():
            value = result.get('avg_metrics', {}).get(metric)
            if value is not None:
                metric_values[overfetch_k] = round(value, 4)
        
        comparison['metrics_comparison'][metric] = metric_values
        
        if metric_values:
            best_k = max(metric_values, key=metric_values.get)
            comparison['best_overfetch_by_metric'][metric] = {
                'overfetch_k': best_k,
                'value': metric_values[best_k]
            }
    
    # Timing comparison
    for overfetch_k, result in results_by_overfetch.items():
        comparison['timing_comparison'][overfetch_k] = result.get('timing', {})
    
    return comparison


def print_comparison_table(results: Dict[str, Any]) -> None:
    """Print a formatted comparison table."""
    
    comparison = results.get('comparison', {})
    metrics_comparison = comparison.get('metrics_comparison', {})
    timing_comparison = comparison.get('timing_comparison', {})
    
    if not metrics_comparison:
        print("\nNo results to compare.")
        return
    
    overfetch_values = sorted(results['metadata']['overfetch_values'])
    
    print("\n" + "=" * 80)
    print("BENCHMARK COMPARISON TABLE")
    print("=" * 80)
    
    # Header
    header = f"{'Metric':<25}"
    for k in overfetch_values:
        header += f" | overfetch={k:>3}"
    print(header)
    print("-" * 80)
    
    # Metrics rows
    for metric, values in metrics_comparison.items():
        row = f"{metric:<25}"
        for k in overfetch_values:
            val = values.get(k, 'N/A')
            if isinstance(val, float):
                row += f" |     {val:>7.4f}"
            else:
                row += f" |     {str(val):>7}"
        print(row)
    
    # Timing rows
    print("-" * 80)
    row = f"{'avg_rerank_time_ms':<25}"
    for k in overfetch_values:
        timing = timing_comparison.get(k, {})
        val = timing.get('avg_rerank_time_ms', 'N/A')
        if isinstance(val, (int, float)):
            row += f" |   {val:>7.1f}ms"
        else:
            row += f" |     {str(val):>7}"
    print(row)
    
    row = f"{'avg_total_time_ms':<25}"
    for k in overfetch_values:
        timing = timing_comparison.get(k, {})
        val = timing.get('avg_total_time_ms', 'N/A')
        if isinstance(val, (int, float)):
            row += f" |   {val:>7.1f}ms"
        else:
            row += f" |     {str(val):>7}"
    print(row)
    
    print("=" * 80)
    
    # Best performers
    best_by_metric = comparison.get('best_overfetch_by_metric', {})
    if best_by_metric:
        print("\nBest overfetch_k by metric:")
        for metric, info in best_by_metric.items():
            print(f"  {metric}: overfetch_k={info['overfetch_k']} (value={info['value']:.4f})")
    
    print()


def main():
    """Main benchmark runner."""
    parser = argparse.ArgumentParser(
        description="Benchmark cross-encoder reranking with different overfetch values",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--overfetch-values",
        type=int,
        nargs="+",
        default=[30, 50, 60],
        help="List of overfetch_k values to benchmark (default: 30 50 60)"
    )
    
    parser.add_argument(
        "--strategy",
        choices=["fixed", "semantic", "hierarchical"],
        default="fixed",
        help="Chunking strategy to use (default: fixed)"
    )
    
    parser.add_argument(
        "--final-top-k",
        type=int,
        default=10,
        help="Number of results to return after reranking (default: 10)"
    )
    
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Maximum number of questions to evaluate"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output path for results JSON"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress detailed output, show only summary"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print("Loading configuration...")
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading configuration: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Run benchmark
    try:
        results = run_benchmark(
            config=config,
            overfetch_values=args.overfetch_values,
            strategy=args.strategy,
            final_top_k=args.final_top_k,
            max_questions=args.max_questions
        )
        
        # Print comparison table
        print_comparison_table(results)
        
        # Save results
        if args.output:
            output_path = args.output
        else:
            results_dir = config.base_path / "data/evaluation/results"
            results_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = results_dir / f"benchmark_overfetch_{timestamp}.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            # Remove per_question details for cleaner output (optional)
            clean_results = {
                'metadata': results['metadata'],
                'comparison': results['comparison'],
                'summary_by_overfetch': {
                    k: {
                        'overfetch_k': v['overfetch_k'],
                        'num_questions': v['num_questions'],
                        'avg_metrics': v['avg_metrics'],
                        'timing': v['timing']
                    }
                    for k, v in results['results_by_overfetch'].items()
                }
            }
            json.dump(clean_results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Results saved to: {output_path}")
        
        print("\n" + "=" * 70)
        print("BENCHMARK COMPLETE")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
