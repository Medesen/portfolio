"""
Test evaluation metrics.

These tests verify that:
- Recall@k is calculated correctly
- MRR is calculated correctly
- NDCG is calculated correctly
- Edge cases are handled properly
"""

from src.evaluation import metrics


def test_recall_at_k_perfect_retrieval():
    """Test Recall@k with perfect retrieval (all relevant docs found)."""
    relevant_docs = ["doc1", "doc2", "doc3"]
    retrieved_docs = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    
    recall = metrics.recall_at_k(retrieved_docs, relevant_docs, k=5)
    
    # Should find all 3 relevant docs = 3/3 = 1.0
    assert recall == 1.0


def test_recall_at_k_partial_retrieval():
    """Test Recall@k with partial retrieval."""
    relevant_docs = ["doc1", "doc2", "doc3"]
    retrieved_docs = ["doc1", "doc4", "doc2", "doc5", "doc6"]
    
    recall = metrics.recall_at_k(retrieved_docs, relevant_docs, k=5)
    
    # Should find 2 out of 3 relevant docs = 2/3 ≈ 0.667
    assert abs(recall - 0.6667) < 0.001


def test_recall_at_k_no_relevant():
    """Test Recall@k when no relevant docs are found."""
    relevant_docs = ["doc1", "doc2", "doc3"]
    retrieved_docs = ["doc4", "doc5", "doc6"]
    
    recall = metrics.recall_at_k(retrieved_docs, relevant_docs, k=3)
    
    # Should find 0 out of 3 = 0.0
    assert recall == 0.0


def test_rr_first_position():
    """Test reciprocal rank when relevant doc is at first position."""
    relevant_docs = ["doc1"]
    retrieved_docs = ["doc1", "doc2", "doc3"]

    rr = metrics.reciprocal_rank(retrieved_docs, relevant_docs)

    # Relevant doc at position 1: RR = 1/1 = 1.0
    assert rr == 1.0


def test_rr_second_position():
    """Test reciprocal rank when relevant doc is at second position."""
    relevant_docs = ["doc2"]
    retrieved_docs = ["doc1", "doc2", "doc3"]

    rr = metrics.reciprocal_rank(retrieved_docs, relevant_docs)

    # Relevant doc at position 2: RR = 1/2 = 0.5
    assert rr == 0.5


def test_ndcg_perfect_ranking():
    """Test NDCG with perfect ranking (all relevant docs at top)."""
    relevant_docs = ["doc1", "doc2"]
    retrieved_docs = ["doc1", "doc2", "doc3"]

    ndcg = metrics.ndcg_at_k(retrieved_docs, relevant_docs, k=3)

    # Perfect ranking should give NDCG = 1.0
    assert ndcg == 1.0


def test_precision_at_k_counts_relevant_in_top_k():
    """Test Precision@k with more retrieved docs than k."""
    relevant_docs = ["doc1", "doc2", "doc3"]
    retrieved_docs = ["doc1", "doc4", "doc2", "doc5", "doc6"]

    precision = metrics.precision_at_k(retrieved_docs, relevant_docs, k=5)

    # 2 relevant docs in the top 5 = 2/5 = 0.4
    assert precision == 0.4


def test_precision_at_k_divides_by_k_when_fewer_returned():
    """Precision@k uses k as the denominator even when fewer docs are returned.

    Dividing by the number returned would reward a system for returning fewer
    results — with chunk→document deduplication, fewer than k documents is the
    common case, so this distinction directly affects reported numbers.
    """
    relevant_docs = ["doc1", "doc2"]
    retrieved_docs = ["doc1", "doc2"]  # only 2 unique documents returned

    precision = metrics.precision_at_k(retrieved_docs, relevant_docs, k=10)

    # 2 relevant in the top 10 = 2/10 — NOT 2/2
    assert precision == 0.2

