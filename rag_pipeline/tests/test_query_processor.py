"""Tests for QueryProcessor retrieval correctness.

Covers the distance->similarity conversion, honoring an explicit top_k under
reranking, and not rewriting an already-rewritten query twice on hybrid fallback.
"""

from unittest.mock import Mock

import pytest

from src.retrieval.query_processor import QueryProcessor
from src.retrieval.vector_store import distance_to_similarity


def _config():
    """Config mock whose .get returns the caller-supplied default."""
    config = Mock()
    config.get.side_effect = lambda key, default=None: default
    return config


def _rewriter(rewritten="rewritten query"):
    rewriter = Mock()
    rewriter.rewrite.return_value = {
        "rewritten_query": rewritten,
        "original_query": "original",
        "from_cache": False,
        "rewrite_failed": False,
        "rewrite_skipped": False,
    }
    return rewriter


# --- finding 9: distance conversion -----------------------------------------


def test_distance_to_similarity_matches_squared_l2_convention():
    """ChromaDB l2 distance is squared, so similarity = 1 - distance/2."""
    assert distance_to_similarity(0.0) == 1.0  # identical vectors
    assert distance_to_similarity(2.0) == 0.0  # orthogonal (squared L2 = 2)
    assert distance_to_similarity(1.0) == pytest.approx(0.5)
    # never negative (opposite vectors, squared L2 = 4)
    assert distance_to_similarity(4.0) == 0.0


# --- finding 11: no double rewrite on hybrid fallback ------------------------


def _processor_without_hybrid(rewriter=None):
    qp = QueryProcessor(
        config=_config(),
        embedder=Mock(),
        vector_store=Mock(),
        bm25_index=None,  # -> hybrid_searcher stays None -> hybrid falls back
        query_rewriter=rewriter,
    )
    qp._query_single_strategy = Mock(return_value=[])  # skip real retrieval
    return qp


def test_hybrid_fallback_does_not_double_rewrite():
    rewriter = _rewriter()
    qp = _processor_without_hybrid(rewriter)

    result = qp.process_query_hybrid("original question", strategy="fixed")

    # Rewrite must happen exactly once (in the hybrid path), not again in the
    # semantic fallback.
    assert rewriter.rewrite.call_count == 1
    # The rewrite metadata is still carried through to the fallback result.
    assert "query_rewriting" in result["metadata"]


def test_process_query_skip_rewrite_flag():
    rewriter = _rewriter()
    qp = _processor_without_hybrid(rewriter)

    qp.process_query("q", strategy="fixed", skip_rewrite=True)
    rewriter.rewrite.assert_not_called()

    qp.process_query("q", strategy="fixed", skip_rewrite=False)
    assert rewriter.rewrite.call_count == 1


# --- finding 8: explicit top_k honored as the post-rerank cut ----------------


def test_explicit_top_k_used_as_rerank_top_k():
    qp = QueryProcessor(
        config=_config(),
        embedder=Mock(),
        vector_store=Mock(),
        bm25_index=None,
        query_rewriter=None,
    )
    # Simulate a reranking-enabled hybrid setup without loading real models.
    qp.reranking_enabled = True
    qp.reranker = Mock()
    qp.reranking_final_top_k = 10
    qp.hybrid_searcher = Mock()
    qp.hybrid_searcher.search.return_value = {"results": [], "metadata": {}}
    qp.bm25_index = Mock()
    qp.bm25_index._loaded_strategy = "fixed"  # matches strategy -> no reload
    qp.search_mode = "hybrid"

    qp.process_query_hybrid("q", strategy="fixed", top_k=5)

    _, kwargs = qp.hybrid_searcher.search.call_args
    assert kwargs["rerank_top_k"] == 5  # explicit top_k, not final_top_k=10


def test_default_top_k_falls_back_to_final_top_k():
    qp = QueryProcessor(
        config=_config(),
        embedder=Mock(),
        vector_store=Mock(),
        bm25_index=None,
        query_rewriter=None,
    )
    qp.reranking_enabled = True
    qp.reranker = Mock()
    qp.reranking_final_top_k = 10
    qp.hybrid_searcher = Mock()
    qp.hybrid_searcher.search.return_value = {"results": [], "metadata": {}}
    qp.bm25_index = Mock()
    qp.bm25_index._loaded_strategy = "fixed"
    qp.search_mode = "hybrid"

    qp.process_query_hybrid("q", strategy="fixed")  # no explicit top_k

    _, kwargs = qp.hybrid_searcher.search.call_args
    assert kwargs["rerank_top_k"] == 10  # configured default
