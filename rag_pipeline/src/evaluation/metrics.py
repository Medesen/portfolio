"""Retrieval metrics for RAG evaluation."""

from __future__ import annotations
from typing import List, Dict, Any, Set
import math

from ..utils.logger import get_logger


class RetrievalMetrics:
    """Calculator for information retrieval metrics."""
    
    def __init__(self, logger_name: str = "retrieval_metrics"):
        """
        Initialize metrics calculator.
        
        Args:
            logger_name: Logger name
        """
        self.logger = get_logger(logger_name)
    
    def recall_at_k(
        self,
        retrieved_doc_ids: List[str],
        relevant_doc_ids: List[str],
        k: int
    ) -> float:
        """
        Calculate Recall@k: proportion of relevant docs in top-k results.
        
        Args:
            retrieved_doc_ids: List of retrieved document IDs (in rank order)
            relevant_doc_ids: List of relevant document IDs
            k: Number of top results to consider
            
        Returns:
            Recall@k score (0.0 to 1.0)
        """
        if not relevant_doc_ids:
            return 0.0
        
        # Get top-k retrieved docs
        top_k = set(retrieved_doc_ids[:k])
        relevant = set(relevant_doc_ids)
        
        # Count relevant docs in top-k
        relevant_retrieved = len(top_k & relevant)
        
        recall = relevant_retrieved / len(relevant)
        return recall
    
    def reciprocal_rank(
        self,
        retrieved_doc_ids: List[str],
        relevant_doc_ids: List[str]
    ) -> float:
        """
        Calculate the reciprocal rank for a single query:
        1 / rank_of_first_relevant_doc (rank starts at 1).

        Averaging this value across queries (``average_metrics``) yields
        Mean Reciprocal Rank — the "mrr" key in the metrics dictionary.

        Args:
            retrieved_doc_ids: List of retrieved document IDs (in rank order)
            relevant_doc_ids: List of relevant document IDs

        Returns:
            Reciprocal rank (0.0 to 1.0)
        """
        if not relevant_doc_ids:
            return 0.0
        
        relevant = set(relevant_doc_ids)
        
        # Find rank of first relevant document
        for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
            if doc_id in relevant:
                return 1.0 / rank
        
        # No relevant document found
        return 0.0
    
    def ndcg_at_k(
        self,
        retrieved_doc_ids: List[str],
        relevant_doc_ids: List[str],
        k: int
    ) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG@k).
        
        Uses binary relevance (1 if relevant, 0 otherwise).
        
        Args:
            retrieved_doc_ids: List of retrieved document IDs (in rank order)
            relevant_doc_ids: List of relevant document IDs
            k: Number of top results to consider
            
        Returns:
            NDCG@k score (0.0 to 1.0)
        """
        if not relevant_doc_ids:
            return 0.0
        
        relevant = set(relevant_doc_ids)
        
        # Calculate DCG@k (Discounted Cumulative Gain)
        # DCG penalizes relevant documents that appear lower in the ranking
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_doc_ids[:k], start=1):
            relevance = 1.0 if doc_id in relevant else 0.0
            # DCG formula: sum(rel_i / log2(i + 1))
            # log2(i+1) provides the "discount" - documents at position 1 contribute more
            # Example: pos 1 → 1/log2(2)=1.0, pos 2 → 1/log2(3)=0.63, pos 10 → 1/log2(11)=0.29
            dcg += relevance / math.log2(i + 1)
        
        # Calculate Ideal DCG (IDCG) - best possible DCG if all relevant docs were ranked first
        # This normalizes DCG to be between 0 and 1
        idcg = 0.0
        for i in range(1, min(len(relevant), k) + 1):
            idcg += 1.0 / math.log2(i + 1)
        
        # Normalize
        if idcg == 0:
            return 0.0
        
        ndcg = dcg / idcg
        return ndcg
    
    def precision_at_k(
        self,
        retrieved_doc_ids: List[str],
        relevant_doc_ids: List[str],
        k: int
    ) -> float:
        """
        Calculate Precision@k: proportion of relevant docs among top-k results.
        
        Args:
            retrieved_doc_ids: List of retrieved document IDs (in rank order)
            relevant_doc_ids: List of relevant document IDs
            k: Number of top results to consider
            
        Returns:
            Precision@k score (0.0 to 1.0)
        """
        if not retrieved_doc_ids[:k]:
            return 0.0
        
        top_k = set(retrieved_doc_ids[:k])
        relevant = set(relevant_doc_ids)
        
        # Count relevant docs in top-k
        relevant_retrieved = len(top_k & relevant)
        
        precision = relevant_retrieved / min(k, len(retrieved_doc_ids))
        return precision
    
    def calculate_all_metrics(
        self,
        retrieved_doc_ids: List[str],
        relevant_doc_ids: List[str],
        k_values: List[int] = [5, 10, 20]
    ) -> Dict[str, Any]:
        """
        Calculate all retrieval metrics for a query.
        
        Args:
            retrieved_doc_ids: List of retrieved document IDs (in rank order)
            relevant_doc_ids: List of relevant document IDs
            k_values: List of k values to compute metrics for
            
        Returns:
            Dictionary with all metrics
        """
        metrics = {
            "mrr": self.reciprocal_rank(retrieved_doc_ids, relevant_doc_ids)
        }
        
        for k in k_values:
            metrics[f"recall@{k}"] = self.recall_at_k(retrieved_doc_ids, relevant_doc_ids, k)
            metrics[f"precision@{k}"] = self.precision_at_k(retrieved_doc_ids, relevant_doc_ids, k)
            metrics[f"ndcg@{k}"] = self.ndcg_at_k(retrieved_doc_ids, relevant_doc_ids, k)
        
        return metrics

    def average_metrics(
        self,
        metrics_list: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Average metrics across multiple queries.
        
        Args:
            metrics_list: List of metric dictionaries
            
        Returns:
            Dictionary with averaged metrics
        """
        if not metrics_list:
            return {}
        
        # Get all metric names
        metric_names = set()
        for metrics in metrics_list:
            metric_names.update(metrics.keys())
        
        # Calculate averages
        averaged = {}
        for metric_name in metric_names:
            values = [m.get(metric_name, 0.0) for m in metrics_list if metric_name in m]
            if values:  # Only compute if we have actual values
                averaged[metric_name] = sum(values) / len(values)
            else:
                averaged[metric_name] = 0.0  # Or you could skip this metric entirely
        
        return averaged

    def topic_coverage(
        self,
        retrieved_chunks: List[Dict[str, Any]],
        expected_topics: List[str],
        content_field: str = "content"
    ) -> float:
        """
        Calculate what proportion of expected topics appear in retrieved content.
        
        This is a softer metric than exact doc_id matching.
        
        Args:
            retrieved_chunks: List of retrieved chunk dictionaries
            expected_topics: List of expected topic keywords
            content_field: Field name containing chunk content
            
        Returns:
            Topic coverage score (0.0 to 1.0)
        """
        if not expected_topics:
            return 0.0
        
        # Concatenate all retrieved content
        all_content = " ".join(
            chunk.get(content_field, "").lower()
            for chunk in retrieved_chunks
        )
        
        # Check which topics are covered
        topics_found = 0
        for topic in expected_topics:
            if topic.lower() in all_content:
                topics_found += 1
        
        coverage = topics_found / len(expected_topics)
        return coverage


def extract_doc_ids_from_chunks(chunks: List[Dict[str, Any]]) -> List[str]:
    """
    Extract document IDs from retrieved chunks.
    
    Args:
        chunks: List of chunk dictionaries
        
    Returns:
        List of unique document IDs in order of appearance
    """
    doc_ids = []
    seen = set()
    
    for chunk in chunks:
        doc_id = chunk.get("doc_id", chunk.get("metadata", {}).get("doc_id"))
        if doc_id and doc_id not in seen:
            doc_ids.append(doc_id)
            seen.add(doc_id)
    
    return doc_ids

