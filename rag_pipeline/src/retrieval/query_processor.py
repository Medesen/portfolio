"""Query processing and retrieval interface."""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Union
import time
from pathlib import Path
import numpy as np

from .embedder import Embedder
from .vector_store import VectorStore, distance_to_similarity
from .bm25_index import BM25Index
from .hybrid_searcher import HybridSearcher
from .query_rewriter import QueryRewriter
from .reranker import CrossEncoderReranker
from ..utils.logger import get_logger


class QueryProcessor:
    """
    Query processor for RAG pipeline.
    
    Handles query preprocessing, embedding generation, vector retrieval,
    and result formatting. Supports querying single or multiple chunking
    strategies with result merging.
    """
    
    def __init__(
        self,
        config,
        embedder: Embedder,
        vector_store: VectorStore,
        bm25_index: Optional[BM25Index] = None,
        query_rewriter: Optional[QueryRewriter] = None,
        logger_name: str = "query_processor"
    ):
        """
        Initialize query processor.
        
        Args:
            config: Configuration object
            embedder: Embedder instance for generating query embeddings
            vector_store: VectorStore instance for retrieval
            bm25_index: Optional BM25Index instance for hybrid search
            query_rewriter: Optional QueryRewriter instance for LLM query rewriting
            logger_name: Logger name
        """
        self.config = config
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.query_rewriter = query_rewriter
        self.logger = get_logger(logger_name)
        
        # Load retrieval configuration
        self.default_top_k = config.get("retrieval.top_k", 20)
        self.min_similarity = config.get("retrieval.min_similarity", 0.0)

        # Hybrid search configuration
        self.search_mode = config.get("retrieval.search_mode", "semantic")
        self.hybrid_alpha = config.get("retrieval.hybrid_alpha", 0.7)
        self.rrf_k = config.get("retrieval.rrf_k", 60)
        self.overfetch_factor = config.get("retrieval.overfetch_factor", 3)

        # Reranking configuration
        self.reranking_enabled = config.get("reranking.enabled", False)
        self.reranking_model = config.get("reranking.model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.reranking_overfetch_k = config.get("reranking.overfetch_k", 50)
        self.reranking_final_top_k = config.get("reranking.final_top_k", 10)
        self.reranking_batch_size = config.get("reranking.batch_size", 32)
        self.reranking_device = config.get("reranking.device", config.get("embeddings.device", "cpu"))

        # Fail fast on invalid config values: zero/negative counts or an
        # out-of-range alpha would otherwise surface as opaque errors deep
        # inside ChromaDB, the RRF fusion math, or the cross-encoder.
        for name, value in [
            ("retrieval.top_k", self.default_top_k),
            ("retrieval.rrf_k", self.rrf_k),
            ("retrieval.overfetch_factor", self.overfetch_factor),
            ("reranking.overfetch_k", self.reranking_overfetch_k),
            ("reranking.final_top_k", self.reranking_final_top_k),
            ("reranking.batch_size", self.reranking_batch_size),
        ]:
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"Config {name} must be a positive integer, got {value!r}")
        if not 0.0 <= self.hybrid_alpha <= 1.0:
            raise ValueError(
                f"Config retrieval.hybrid_alpha must be in [0.0, 1.0], got {self.hybrid_alpha!r}"
            )
        
        # Initialize reranker if enabled
        self.reranker: Optional[CrossEncoderReranker] = None
        if self.reranking_enabled and self.reranking_model:
            self.reranker = CrossEncoderReranker(
                model_name=self.reranking_model,
                device=self.reranking_device,
                batch_size=self.reranking_batch_size,
                logger_name="reranker"
            )
            self.logger.info(
                f"Reranker initialized (model={self.reranking_model}, "
                f"overfetch_k={self.reranking_overfetch_k}, "
                f"final_top_k={self.reranking_final_top_k})"
            )
        
        # Initialize hybrid searcher if BM25 index is available
        self.hybrid_searcher: Optional[HybridSearcher] = None
        if bm25_index is not None:
            self.hybrid_searcher = HybridSearcher(
                vector_store=vector_store,
                bm25_index=bm25_index,
                embedder=embedder,
                alpha=self.hybrid_alpha,
                rrf_k=self.rrf_k,
                reranker=self.reranker,
                logger_name="hybrid_searcher"
            )
            self.logger.info("Hybrid searcher initialized")
        
        self.logger.info(f"Query processor initialized (search_mode={self.search_mode})")
    
    def process_query(
        self,
        query_text: str,
        strategy: Optional[str] = None,
        top_k: Optional[int] = None,
        show_full_content: bool = False,
        skip_rewrite: bool = False
    ) -> Dict[str, Any]:
        """
        Process a query and retrieve relevant chunks.
        
        Args:
            query_text: Natural language query
            strategy: Chunking strategy to query ("fixed", "semantic", "hierarchical", or None for default)
            top_k: Number of results to return (None for default)
            show_full_content: Whether to include full chunk content
            
        Returns:
            Dictionary with query results
        """
        start_time = time.time()
        
        # Rewrite query with LLM if available. skip_rewrite is set by callers
        # (e.g. hybrid fallback) that already rewrote the query, to avoid
        # rewriting an already-rewritten query a second time.
        rewrite_metadata = None
        if self.query_rewriter is not None and not skip_rewrite:
            rewrite_result = self.query_rewriter.rewrite(query_text)
            query_text = rewrite_result["rewritten_query"]
            rewrite_metadata = {
                "original_query": rewrite_result["original_query"],
                "rewritten_query": rewrite_result["rewritten_query"],
                "from_cache": rewrite_result["from_cache"],
                "rewrite_failed": rewrite_result["rewrite_failed"],
                "rewrite_skipped": rewrite_result["rewrite_skipped"],
            }

        # Normalize query
        query_text = self._normalize_query(query_text)
        self.logger.info(f"Processing query: '{query_text}'")
        
        # Determine strategy
        if strategy is None:
            strategy = self.config.get("retrieval.strategy", "fixed")
        
        # Determine top_k: reranking.final_top_k takes precedence when reranking is enabled
        if top_k is None:
            if self.reranking_enabled:
                top_k = self.reranking_final_top_k
            else:
                top_k = self.default_top_k
        
        # Generate query embedding
        self.logger.info("Generating query embedding...")
        embed_start = time.time()
        query_embedding = self.embedder.embed(query_text, show_progress=False, normalize=True)
        embed_time = time.time() - embed_start
        self.logger.info(f"Query embedding generated in {embed_time:.3f}s")
        
        # Retrieve from strategy
        retrieval_start = time.time()
        results = self._query_single_strategy(
            query_embedding, strategy, top_k
        )
        retrieval_time = time.time() - retrieval_start
        
        total_time = time.time() - start_time
        
        # Format results
        formatted_results = {
            "query": query_text,
            "strategy": strategy,
            "results": results,
            "metadata": {
                "total_results": len(results),
                "top_k_requested": top_k,
                "min_similarity": self.min_similarity,
                "timing": {
                    "embedding_time": round(embed_time, 3),
                    "retrieval_time": round(retrieval_time, 3),
                    "total_time": round(total_time, 3)
                }
            }
        }
        
        # Add rewrite metadata if query was rewritten
        if rewrite_metadata is not None:
            formatted_results["metadata"]["query_rewriting"] = rewrite_metadata
        
        self.logger.info(
            f"Query completed: {len(results)} results in {total_time:.3f}s"
        )
        
        return formatted_results
    
    def process_query_hybrid(
        self,
        query_text: str,
        strategy: str,
        top_k: Optional[int] = None,
        alpha: Optional[float] = None,
        search_mode: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a query using hybrid search (semantic + keyword with RRF).
        
        Args:
            query_text: Natural language query
            strategy: Chunking strategy to query ("fixed", "semantic", "hierarchical")
            top_k: Number of results to return (None for default)
            alpha: Override default alpha weight (0.0-1.0)
            search_mode: Override default search mode ("semantic", "keyword", "hybrid")
            
        Returns:
            Dictionary with query results
        """
        # Rewrite query with LLM if available
        rewrite_metadata = None
        if self.query_rewriter is not None:
            rewrite_result = self.query_rewriter.rewrite(query_text)
            query_text = rewrite_result["rewritten_query"]
            rewrite_metadata = {
                "original_query": rewrite_result["original_query"],
                "rewritten_query": rewrite_result["rewritten_query"],
                "from_cache": rewrite_result["from_cache"],
                "rewrite_failed": rewrite_result["rewrite_failed"],
                "rewrite_skipped": rewrite_result["rewrite_skipped"],
            }
        
        # Normalize query
        query_text = self._normalize_query(query_text)
        self.logger.info(f"Processing hybrid query: '{query_text}'")
        
        # Remember whether the caller asked for a specific top_k before defaulting,
        # so an explicit --top-k is honored as the post-rerank cut (see below).
        explicit_top_k = top_k
        # Determine parameters: reranking.final_top_k takes precedence when reranking is enabled
        if top_k is None:
            top_k = self.reranking_final_top_k if self.reranking_enabled else self.default_top_k
        search_mode = search_mode or self.search_mode
        
        # Check if hybrid searcher is available
        if self.hybrid_searcher is None:
            self.logger.warning(
                "Hybrid searcher not available (BM25 index not loaded). "
                "Falling back to semantic search."
            )
            # query_text is already rewritten+normalized here; skip a second rewrite
            # and carry the rewrite metadata through to the fallback result.
            result = self.process_query(
                query_text, strategy=strategy, top_k=explicit_top_k, skip_rewrite=True
            )
            if rewrite_metadata is not None:
                result["metadata"]["query_rewriting"] = rewrite_metadata
            return result
        
        # Load BM25 index for the strategy if not already loaded or different strategy
        if self.bm25_index._loaded_strategy != strategy:
            if not self.bm25_index.load_index(strategy):
                self.logger.warning(
                    f"BM25 index not found for strategy '{strategy}'. "
                    "Falling back to semantic search."
                )
                # Already rewritten+normalized; skip the second rewrite and carry
                # the rewrite metadata through.
                result = self.process_query(
                    query_text, strategy=strategy, top_k=explicit_top_k, skip_rewrite=True
                )
                if rewrite_metadata is not None:
                    result["metadata"]["query_rewriting"] = rewrite_metadata
                return result
        
        # Perform search based on mode
        if search_mode == "keyword":
            result = self.hybrid_searcher.search_keyword_only(
                query=query_text,
                top_k=top_k
            )
        elif search_mode == "semantic":
            result = self.hybrid_searcher.search_semantic_only(
                query=query_text,
                strategy=strategy,
                top_k=top_k
            )
        else:  # hybrid
            # When reranking is enabled, use reranking parameters
            if self.reranking_enabled and self.reranker is not None:
                # Honor an explicit top_k as the number of results kept after
                # reranking; fall back to the configured final_top_k otherwise.
                rerank_top_k = (
                    explicit_top_k if explicit_top_k is not None else self.reranking_final_top_k
                )
                result = self.hybrid_searcher.search(
                    query=query_text,
                    strategy=strategy,
                    top_k=top_k,
                    overfetch_k=self.reranking_overfetch_k,
                    alpha=alpha,
                    rerank_top_k=rerank_top_k
                )
            else:
                result = self.hybrid_searcher.search(
                    query=query_text,
                    strategy=strategy,
                    top_k=top_k,
                    overfetch_factor=self.overfetch_factor,
                    alpha=alpha
                )
        
        # Add search_mode to metadata
        result['metadata']['search_mode'] = search_mode
        
        # Add rewrite metadata if query was rewritten
        if rewrite_metadata is not None:
            result['metadata']['query_rewriting'] = rewrite_metadata
        
        # Normalize timing keys for compatibility with format_console_output
        if 'timing' in result['metadata']:
            timing = result['metadata']['timing']
            # Map hybrid search timing keys to expected format
            normalized_timing = {
                'total_time': timing.get('total', 0),
                'retrieval_time': timing.get('semantic_search', 0) + timing.get('keyword_search', 0) + timing.get('fusion', 0),
                'embedding_time': 0,  # Embedding is included in semantic_search time for hybrid
            }
            # Preserve additional timing info
            if 'rerank_ms' in timing:
                normalized_timing['rerank_ms'] = timing['rerank_ms']
            result['metadata']['timing'] = normalized_timing
        
        self.logger.info(
            f"Hybrid query completed: {len(result['results'])} results "
            f"(mode={search_mode})"
        )
        
        return result
    
    def _normalize_query(self, query_text: str) -> str:
        """
        Normalize and validate query text.
        
        Args:
            query_text: Raw query text
            
        Returns:
            Normalized query text
            
        Raises:
            ValueError: If query is empty after normalization
        """
        # Basic normalization: strip whitespace
        query_text = query_text.strip()
        
        # Check for empty query
        if not query_text:
            raise ValueError("Query cannot be empty")
        
        # Remove extra whitespace
        query_text = " ".join(query_text.split())
        
        # Limit query length to prevent performance issues
        max_query_length = self.config.get("retrieval.max_query_length", 1000)
        if len(query_text) > max_query_length:
            self.logger.warning(
                f"Query truncated from {len(query_text)} to {max_query_length} characters"
            )
            query_text = query_text[:max_query_length]
        
        return query_text
    
    def _query_single_strategy(
        self,
        query_embedding: Union[List[float], np.ndarray],
        strategy: str,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """
        Query a single chunking strategy.
        
        Args:
            query_embedding: Query embedding vector (list or numpy array)
            strategy: Strategy name
            top_k: Number of results
            
        Returns:
            List of result dictionaries
        """
        self.logger.info(f"Querying strategy: {strategy} (top_k={top_k})")
        
        # Check if collection exists
        collection_info = self.vector_store.get_collection_info(strategy)
        if collection_info is None:
            self.logger.warning(
                f"Strategy '{strategy}' not found or not indexed. "
                f"Run 'index --strategy {strategy}' first."
            )
            return []
        
        # Query vector store
        raw_results = self.vector_store.query(
            collection_name=strategy,
            query_embedding=query_embedding.tolist(),
            n_results=top_k,
        )
        
        # Format results
        results = self._format_results(raw_results, strategy)
        
        # Filter by minimum similarity
        if self.min_similarity > 0.0:
            results = [
                r for r in results
                if r["similarity_score"] >= self.min_similarity
            ]
        
        self.logger.info(f"Retrieved {len(results)} results from '{strategy}'")
        return results
    
    def _format_results(
        self,
        raw_results: Dict[str, Any],
        strategy_name: str
    ) -> List[Dict[str, Any]]:
        """
        Format ChromaDB results into readable format.
        
        Args:
            raw_results: Raw ChromaDB query results
            strategy_name: Name of the strategy
            
        Returns:
            List of formatted result dictionaries
        """
        formatted = []
        
        # ChromaDB returns results as lists within the dictionary
        ids = raw_results.get("ids", [[]])[0]
        documents = raw_results.get("documents", [[]])[0]
        metadatas = raw_results.get("metadatas", [[]])[0]
        distances = raw_results.get("distances", [[]])[0]
        
        for i, (chunk_id, content, metadata, distance) in enumerate(
            zip(ids, documents, metadatas, distances)
        ):
            # ChromaDB's default l2 space returns the SQUARED distance, so for
            # normalized vectors cosine_similarity = 1 - distance / 2.
            similarity_score = distance_to_similarity(distance)
            
            result = {
                "rank": i + 1,
                "chunk_id": chunk_id,
                "doc_id": metadata.get("doc_id", "unknown"),
                "content": content,
                "similarity_score": round(similarity_score, 4),
                "metadata": metadata,
                "strategy": strategy_name
            }
            
            formatted.append(result)
        
        return formatted
    
    def format_console_output(
        self,
        results: Dict[str, Any],
        show_full_content: bool = False,
        max_excerpt_length: int = 1000
    ) -> str:
        """
        Format results for console display.
        
        Args:
            results: Query results dictionary
            show_full_content: Whether to show full content
            max_excerpt_length: Maximum length for content excerpts
            
        Returns:
            Formatted string for console output
        """
        lines = []
        lines.append("=" * 80)
        lines.append("QUERY RESULTS")
        lines.append("=" * 80)

        rewrite_info = results.get('metadata', {}).get('query_rewriting')

        if rewrite_info:
            original = rewrite_info.get('original_query', '')
            rewritten = rewrite_info.get('rewritten_query', '')

            if rewrite_info.get('rewrite_skipped'):
                lines.append(f"Query: \"{original}\" [rewriting disabled]")
            elif rewrite_info.get('rewrite_failed'):
                lines.append(f"Query: \"{original}\" [rewrite failed, using original]")
            elif original != rewritten:
                lines.append(f"Query: \"{rewritten}\" [rewritten]")
                lines.append(f"Original: \"{original}\"")
            else:
                lines.append(f"Query: \"{original}\"")
        else:
            lines.append(f"Query: \"{results['query']}\"")
            lines.append("NB: Query rewriting not performed.")


        lines.append(f"Strategy: {results['strategy']}")
        lines.append(f"Results: {results['metadata']['total_results']}")

        timing = results['metadata']['timing']

        lines.append(
            f"Time: {timing['total_time']}s "
            f"(embedding: {timing['embedding_time']}s, "
            f"retrieval: {timing['retrieval_time']}s)"
        )

        if results['metadata'].get('strategies_queried'):
            strategies = ", ".join(results['metadata']['strategies_queried'])
            lines.append(f"Strategies queried: {strategies}")

        lines.append("-" * 80)

        # Display results
        if not results['results']:
            lines.append("\nNo results found.")
        else:
            for result in results['results']:
                # Get strategy from result or top-level results dict
                strategy = result.get('strategy', results.get('strategy', 'unknown'))

                # Build score string showing available scores
                score_parts = []
                if 'rrf_score' in result:
                    score_parts.append(f"rrf: {result['rrf_score']:.4f}")
                if 'rerank_score' in result:
                    score_parts.append(f"rerank: {result['rerank_score']:.4f}")
                if 'similarity_score' in result and not score_parts:
                    # Only show similarity if no hybrid/rerank scores
                    score_parts.append(f"similarity: {result['similarity_score']:.4f}")

                score_str = ", ".join(score_parts) if score_parts else "no score"

                lines.append(f"\nRank {result['rank']} [{strategy}] ({score_str})")
                lines.append(f"Doc: {result['doc_id']}")
                lines.append(f"Chunk: {result['chunk_id']}")

                # Content display
                content = result['content']
                if show_full_content:
                    lines.append(f"\n{content}")
                else:
                    excerpt = self._create_excerpt(content, max_excerpt_length)
                    lines.append(f"\n{excerpt}")

                lines.append("")  # Blank line between results

        lines.append("-" * 80)
        lines.append(f"Retrieved {results['metadata']['total_results']} results")
        lines.append("=" * 80)

        return "\n".join(lines)

    def _create_excerpt(self, text: str, max_length: int) -> str:
        """
        Create an excerpt from text.
        
        Args:
            text: Full text
            max_length: Maximum length
            
        Returns:
            Truncated text with ellipsis
        """
        if len(text) <= max_length:
            return text
        
        # Try to break at sentence boundary
        excerpt = text[:max_length]
        last_period = excerpt.rfind(". ")
        
        if last_period > max_length * 0.7:  # At least 70% of max_length
            return excerpt[:last_period + 1] + ".."
        
        # Otherwise, break at word boundary
        last_space = excerpt.rfind(" ")
        if last_space > 0:
            return excerpt[:last_space] + "..."
        
        return excerpt + "..."

