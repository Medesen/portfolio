"""Hybrid search combining semantic and keyword search with RRF."""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Set, TYPE_CHECKING
import time

from .vector_store import VectorStore, distance_to_similarity
from .bm25_index import BM25Index
from .embedder import Embedder
from ..utils.logger import get_logger

if TYPE_CHECKING:
    from .reranker import CrossEncoderReranker


class HybridSearcher:
    """
    Hybrid search combining semantic (dense) and keyword (sparse) search.
    
    Uses Reciprocal Rank Fusion (RRF) to combine rankings from both
    search methods into a single, higher-quality ranking.
    
    RRF Score = alpha * (1 / (k + semantic_rank)) + (1 - alpha) * (1 / (k + keyword_rank))
    
    Where:
    - alpha: Weight for semantic search (0.0-1.0). 1.0 = pure semantic, 0.0 = pure keyword
    - k: RRF constant (typically 60). Higher values reduce impact of top-ranked documents
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        bm25_index: BM25Index,
        embedder: Embedder,
        alpha: float = 0.7,
        rrf_k: int = 60,
        reranker: Optional["CrossEncoderReranker"] = None,
        logger_name: str = "hybrid_searcher"
    ):
        """
        Initialize hybrid searcher.
        
        Args:
            vector_store: VectorStore for semantic search
            bm25_index: BM25Index for keyword search
            embedder: Embedder for generating query embeddings
            alpha: Weight for semantic search (0.0-1.0). 
                   1.0 = pure semantic, 0.0 = pure keyword
            rrf_k: RRF constant (typically 60). Higher values reduce 
                   the impact of top-ranked documents.
            reranker: Optional CrossEncoderReranker for reranking results
            logger_name: Logger name
            
        Raises:
            ValueError: If alpha is not between 0.0 and 1.0
        """
        # Validate alpha
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be between 0.0 and 1.0, got {alpha}")
        
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.embedder = embedder
        self.alpha = alpha
        self.rrf_k = rrf_k
        self.reranker = reranker
        self.logger = get_logger(logger_name)
        
        reranker_status = "enabled" if reranker else "disabled"
        self.logger.info(
            f"Hybrid searcher initialized (alpha={alpha}, rrf_k={rrf_k}, reranker={reranker_status})"
        )
    
    def search(
        self,
        query: str,
        strategy: str,
        top_k: int = 20,
        overfetch_factor: int = 3,
        overfetch_k: Optional[int] = None,
        alpha: Optional[float] = None,
        rerank_top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Perform hybrid search combining semantic and keyword search.
        
        Args:
            query: Search query
            strategy: Chunking strategy/collection name
            top_k: Number of final results to return (used if reranker not active)
            overfetch_factor: Multiplier for initial retrieval 
                              (retrieves top_k * overfetch_factor from each method)
            overfetch_k: Explicit number of documents to fetch before reranking.
                         When set, overrides overfetch_factor calculation.
            alpha: Override default alpha weight
            rerank_top_k: Number of results to return after reranking.
                          When set with reranker, this takes precedence over top_k.
            
        Returns:
            Dictionary with results, timing, and metadata
        """
        start_time = time.time()
        alpha = alpha if alpha is not None else self.alpha
        
        # Validate alpha is in valid range
        if not 0.0 <= alpha <= 1.0:
            self.logger.warning(
                f"Invalid alpha value {alpha}, clamping to [0.0, 1.0]"
            )
            alpha = max(0.0, min(1.0, alpha))
        
        # Determine fetch_k: explicit overfetch_k takes precedence
        if overfetch_k is not None:
            fetch_k = overfetch_k
        else:
            fetch_k = top_k * overfetch_factor
        
        # Determine final_top_k: rerank_top_k takes precedence when reranker is active
        if self.reranker is not None and rerank_top_k is not None:
            final_top_k = rerank_top_k
        else:
            final_top_k = top_k
        
        query_preview = query[:50] + "..." if len(query) > 50 else query
        self.logger.info(
            f"Hybrid search: query='{query_preview}', strategy={strategy}, "
            f"top_k={final_top_k}, fetch_k={fetch_k}, alpha={alpha}, "
            f"reranker={'enabled' if self.reranker else 'disabled'}"
        )
        
        # 1. Semantic search
        semantic_start = time.time()
        semantic_results = self._semantic_search(query, strategy, fetch_k)
        semantic_time = time.time() - semantic_start
        
        # 2. Keyword search
        keyword_start = time.time()
        keyword_results = self._keyword_search(query, fetch_k)
        keyword_time = time.time() - keyword_start
        
        # 3. Reciprocal Rank Fusion
        # When reranker is active, fuse all candidates (no truncation yet)
        fusion_top_k = fetch_k if self.reranker is not None else final_top_k
        fusion_start = time.time()
        fused_results = self._reciprocal_rank_fusion(
            semantic_results=semantic_results,
            keyword_results=keyword_results,
            alpha=alpha,
            top_k=fusion_top_k
        )
        fusion_time = time.time() - fusion_start
        
        # 4. Reranking (if enabled)
        rerank_time_ms = None
        reranked = False
        if self.reranker is not None:
            rerank_result = self.reranker.rerank(
                query=query,
                results=fused_results,
                top_k=final_top_k
            )
            fused_results = rerank_result['results']
            rerank_time_ms = rerank_result['rerank_time_ms']
            reranked = rerank_result['reranked']
        
        total_time = time.time() - start_time
        
        # Build timing metadata
        timing = {
            'semantic_search': round(semantic_time, 4),
            'keyword_search': round(keyword_time, 4),
            'fusion': round(fusion_time, 4),
            'total': round(total_time, 4)
        }
        if rerank_time_ms is not None:
            timing['rerank_ms'] = rerank_time_ms
        
        return {
            'query': query,
            'strategy': strategy,
            'results': fused_results,
            'metadata': {
                'total_results': len(fused_results),
                'top_k': final_top_k,
                'alpha': alpha,
                'rrf_k': self.rrf_k,
                'semantic_candidates': len(semantic_results),
                'keyword_candidates': len(keyword_results),
                'reranked': reranked,
                'timing': timing
            }
        }
    
    def search_semantic_only(
        self,
        query: str,
        strategy: str,
        top_k: int = 20
    ) -> Dict[str, Any]:
        """
        Perform semantic-only search (alpha=1.0 equivalent).
        
        Args:
            query: Search query
            strategy: Chunking strategy/collection name
            top_k: Number of results to return
            
        Returns:
            Dictionary with results and metadata
        """
        start_time = time.time()
        
        results = self._semantic_search(query, strategy, top_k)
        
        # Add rank field to match hybrid output format
        for i, result in enumerate(results, start=1):
            result['rank'] = i
            result['rrf_score'] = 1.0 / (self.rrf_k + i)  # Pure semantic RRF score
            result['in_semantic'] = True
            result['in_keyword'] = False
        
        return {
            'query': query,
            'strategy': strategy,
            'results': results,
            'metadata': {
                'total_results': len(results),
                'top_k': top_k,
                'alpha': 1.0,
                'search_mode': 'semantic',
                'timing': {
                    'total': round(time.time() - start_time, 4)
                }
            }
        }
    
    def search_keyword_only(
        self,
        query: str,
        top_k: int = 20
    ) -> Dict[str, Any]:
        """
        Perform keyword-only search (alpha=0.0 equivalent).
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            Dictionary with results and metadata
        """
        start_time = time.time()
        
        results = self._keyword_search(query, top_k)
        
        # Add rank field to match hybrid output format
        for i, result in enumerate(results, start=1):
            result['rank'] = i
            result['rrf_score'] = 1.0 / (self.rrf_k + i)  # Pure keyword RRF score
            result['in_semantic'] = False
            result['in_keyword'] = True
        
        return {
            'query': query,
            'strategy': None,
            'results': results,
            'metadata': {
                'total_results': len(results),
                'top_k': top_k,
                'alpha': 0.0,
                'search_mode': 'keyword',
                'timing': {
                    'total': round(time.time() - start_time, 4)
                }
            }
        }
    
    def _semantic_search(
        self,
        query: str,
        strategy: str,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Perform semantic search using vector store."""
        # Generate query embedding
        query_embedding = self.embedder.embed(query, show_progress=False, normalize=True)
        
        # Query vector store
        raw_results = self.vector_store.query(
            collection_name=strategy,
            query_embedding=query_embedding.tolist(),
            n_results=top_k
        )
        
        # Format results
        results = []
        ids = raw_results.get('ids', [[]])[0]
        documents = raw_results.get('documents', [[]])[0]
        metadatas = raw_results.get('metadatas', [[]])[0]
        distances = raw_results.get('distances', [[]])[0]
        
        for rank, (chunk_id, content, metadata, distance) in enumerate(
            zip(ids, documents, metadatas, distances), start=1
        ):
            # ChromaDB's default l2 space returns the SQUARED distance, so for
            # normalized vectors cosine_similarity = 1 - distance / 2.
            similarity = distance_to_similarity(distance)
            
            results.append({
                'chunk_id': chunk_id,
                'doc_id': metadata.get('doc_id', 'unknown'),
                'content': content,
                'semantic_score': similarity,
                'semantic_rank': rank,
                'metadata': metadata
            })
        
        return results
    
    def _keyword_search(
        self,
        query: str,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Perform keyword search using BM25 index."""
        return self.bm25_index.search(query, top_k=top_k)
    
    def _reciprocal_rank_fusion(
        self,
        semantic_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]],
        alpha: float,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Combine rankings using weighted Reciprocal Rank Fusion.
        
        RRF Score = alpha * (1 / (k + semantic_rank)) + (1 - alpha) * (1 / (k + keyword_rank))
        
        Documents appearing in only one list receive score only from that list.
        """
        # Build lookup dictionaries by chunk_id
        semantic_lookup: Dict[str, Dict] = {
            r['chunk_id']: r for r in semantic_results
        }
        keyword_lookup: Dict[str, Dict] = {
            r['chunk_id']: r for r in keyword_results
        }
        
        # Collect all unique chunk_ids
        all_chunk_ids: Set[str] = set(semantic_lookup.keys()) | set(keyword_lookup.keys())
        
        # Calculate RRF scores
        scored_results = []
        
        for chunk_id in all_chunk_ids:
            semantic_data = semantic_lookup.get(chunk_id)
            keyword_data = keyword_lookup.get(chunk_id)
            
            # Calculate RRF score components
            semantic_rrf = 0.0
            keyword_rrf = 0.0
            
            if semantic_data:
                semantic_rank = semantic_data['semantic_rank']
                semantic_rrf = alpha * (1.0 / (self.rrf_k + semantic_rank))
            
            if keyword_data:
                keyword_rank = keyword_data['bm25_rank']
                keyword_rrf = (1.0 - alpha) * (1.0 / (self.rrf_k + keyword_rank))
            
            rrf_score = semantic_rrf + keyword_rrf
            
            # Merge data from both sources
            merged = {
                'chunk_id': chunk_id,
                'rrf_score': rrf_score,
                'in_semantic': semantic_data is not None,
                'in_keyword': keyword_data is not None,
            }
            
            # Prefer semantic data for content/metadata (usually more complete)
            if semantic_data:
                merged['doc_id'] = semantic_data['doc_id']
                merged['content'] = semantic_data['content']
                merged['metadata'] = semantic_data['metadata']
                merged['semantic_score'] = semantic_data['semantic_score']
                merged['semantic_rank'] = semantic_data['semantic_rank']
            
            if keyword_data:
                merged['doc_id'] = merged.get('doc_id') or keyword_data['doc_id']
                merged['content'] = merged.get('content') or keyword_data['content']
                merged['metadata'] = merged.get('metadata') or keyword_data.get('metadata', {})
                merged['bm25_score'] = keyword_data['bm25_score']
                merged['bm25_rank'] = keyword_data['bm25_rank']
            
            scored_results.append(merged)
        
        # Sort by RRF score descending
        scored_results.sort(key=lambda x: x['rrf_score'], reverse=True)
        
        # Take top_k and assign final ranks
        final_results = scored_results[:top_k]
        for rank, result in enumerate(final_results, start=1):
            result['rank'] = rank
        
        return final_results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get searcher statistics and configuration."""
        stats = {
            'alpha': self.alpha,
            'rrf_k': self.rrf_k,
            'vector_store_collections': self.vector_store.list_collections(),
            'bm25_index': self.bm25_index.get_stats(),
            'reranker_enabled': self.reranker is not None
        }
        if self.reranker is not None:
            stats['reranker'] = self.reranker.get_stats()
        return stats
