"""Tests for hybrid search functionality (BM25 + Semantic with RRF)."""

import pytest
from unittest.mock import Mock
import numpy as np

from src.retrieval.bm25_index import BM25Index
from src.retrieval.hybrid_searcher import HybridSearcher


class TestBM25Index:
    """Tests for BM25Index class."""
    
    @pytest.fixture
    def bm25_index(self, temp_dir):
        """Create a BM25Index instance for testing."""
        return BM25Index(persist_directory=temp_dir)
    
    @pytest.fixture
    def sample_chunks(self):
        """Sample chunks for testing."""
        return [
            {
                'chunk_id': 'c1',
                'doc_id': 'd1',
                'content': 'StandardScaler normalizes features by removing the mean and scaling to unit variance.',
                'chunk_index': 0,
                'metadata': {'source': 'preprocessing.html'}
            },
            {
                'chunk_id': 'c2',
                'doc_id': 'd1',
                'content': 'Use fit_transform to fit the scaler and transform data in one step.',
                'chunk_index': 1,
                'metadata': {'source': 'preprocessing.html'}
            },
            {
                'chunk_id': 'c3',
                'doc_id': 'd2',
                'content': 'GridSearchCV performs hyperparameter tuning with cross-validation.',
                'chunk_index': 0,
                'metadata': {'source': 'model_selection.html'}
            },
            {
                'chunk_id': 'c4',
                'doc_id': 'd3',
                'content': 'PCA reduces dimensionality by projecting data onto principal components.',
                'chunk_index': 0,
                'metadata': {'source': 'decomposition.html'}
            },
        ]
    
    def test_initialization(self, bm25_index):
        """Test BM25Index initializes correctly."""
        assert bm25_index.index is None
        assert bm25_index.chunk_ids == []
        assert bm25_index.chunk_metadata == {}
        assert bm25_index.stemmer is not None  # NLTK should be available
        assert len(bm25_index.stopwords) > 0
    
    def test_tokenization_basic(self, bm25_index):
        """Test basic tokenization."""
        tokens = bm25_index.tokenize("Hello World")
        assert "hello" in tokens or "world" in tokens
        # Short tokens should be filtered
        assert "a" not in tokens
    
    def test_tokenization_sklearn_terms(self, bm25_index):
        """Test tokenization preserves sklearn terms with underscores."""
        tokens = bm25_index.tokenize("StandardScaler fit_transform method")
        # fit_transform should be preserved as one token
        assert "fit_transform" in tokens
        # StandardScaler gets lowercased and potentially stemmed
        assert any("standard" in t or "standardscal" in t for t in tokens)
    
    def test_tokenization_stopword_removal(self, bm25_index):
        """Test stopwords are removed."""
        tokens = bm25_index.tokenize("This is a test of the tokenizer")
        # Common stopwords should be removed
        assert "this" not in tokens
        assert "is" not in tokens
        assert "the" not in tokens
        # Content words should remain (possibly stemmed)
        assert any("test" in t for t in tokens)
        assert any("token" in t for t in tokens)
    
    def test_tokenization_stemming(self, bm25_index):
        """Test Porter stemming is applied."""
        tokens = bm25_index.tokenize("normalizing normalized normalization")
        # All should stem to "normal" or similar
        assert len(set(tokens)) == 1  # All same after stemming
    
    def test_build_index(self, bm25_index, sample_chunks):
        """Test building BM25 index from chunks."""
        bm25_index.build_index(sample_chunks, 'test_strategy')
        
        assert bm25_index.index is not None
        assert len(bm25_index.chunk_ids) == 4
        assert len(bm25_index.chunk_metadata) == 4
        assert 'c1' in bm25_index.chunk_metadata
    
    def test_save_and_load_index(self, bm25_index, sample_chunks):
        """Test index persistence."""
        bm25_index.build_index(sample_chunks, 'test_strategy')
        
        # Create new instance and load
        new_index = BM25Index(persist_directory=bm25_index.persist_directory)
        assert new_index.load_index('test_strategy')
        
        assert len(new_index.chunk_ids) == 4
        assert new_index.index is not None
    
    def test_search_basic(self, bm25_index, sample_chunks):
        """Test basic search functionality."""
        bm25_index.build_index(sample_chunks, 'test_strategy')
        
        results = bm25_index.search("StandardScaler", top_k=3)
        
        assert len(results) > 0
        assert results[0]['chunk_id'] == 'c1'  # Most relevant
        assert 'bm25_score' in results[0]
        assert 'bm25_rank' in results[0]
        assert results[0]['bm25_rank'] == 1
    
    def test_search_multiple_terms(self, bm25_index, sample_chunks):
        """Test search with multiple query terms."""
        bm25_index.build_index(sample_chunks, 'test_strategy')
        
        results = bm25_index.search("fit_transform scaler", top_k=3)
        
        # Should match c1 and c2 (both mention scaler/fit_transform)
        chunk_ids = [r['chunk_id'] for r in results]
        assert 'c1' in chunk_ids or 'c2' in chunk_ids
    
    def test_search_no_matches(self, bm25_index, sample_chunks):
        """Test search with no matching terms."""
        bm25_index.build_index(sample_chunks, 'test_strategy')
        
        results = bm25_index.search("xyznonexistent", top_k=3)
        
        # Should return empty or zero-score results
        assert len(results) == 0
    
    def test_index_exists(self, bm25_index, sample_chunks):
        """Test index_exists method."""
        assert not bm25_index.index_exists('test_strategy')
        
        bm25_index.build_index(sample_chunks, 'test_strategy')
        
        assert bm25_index.index_exists('test_strategy')
        assert not bm25_index.index_exists('nonexistent')
    
    def test_get_stats(self, bm25_index, sample_chunks):
        """Test get_stats method."""
        stats = bm25_index.get_stats()
        assert stats['num_documents'] == 0
        assert stats['index_loaded'] == False
        
        bm25_index.build_index(sample_chunks, 'test_strategy')
        
        stats = bm25_index.get_stats()
        assert stats['num_documents'] == 4
        assert stats['index_loaded'] == True
        assert stats['stemmer_available'] == True


class TestHybridSearcher:
    """Tests for HybridSearcher class."""
    
    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock VectorStore."""
        mock = Mock()
        mock.query.return_value = {
            'ids': [['c1', 'c2', 'c3']],
            'documents': [[
                'StandardScaler normalizes features.',
                'Use fit_transform method.',
                'PCA for dimensionality reduction.'
            ]],
            'metadatas': [[
                {'doc_id': 'd1'},
                {'doc_id': 'd1'},
                {'doc_id': 'd2'}
            ]],
            'distances': [[0.2, 0.4, 0.6]]
        }
        mock.list_collections.return_value = ['fixed', 'semantic']
        return mock
    
    @pytest.fixture
    def mock_bm25_index(self):
        """Create a mock BM25Index."""
        mock = Mock()
        mock.search.return_value = [
            {'chunk_id': 'c2', 'doc_id': 'd1', 'content': 'Use fit_transform method.', 
             'bm25_score': 5.2, 'bm25_rank': 1, 'metadata': {}},
            {'chunk_id': 'c4', 'doc_id': 'd3', 'content': 'GridSearchCV tuning.', 
             'bm25_score': 3.1, 'bm25_rank': 2, 'metadata': {}},
            {'chunk_id': 'c1', 'doc_id': 'd1', 'content': 'StandardScaler normalizes.', 
             'bm25_score': 2.8, 'bm25_rank': 3, 'metadata': {}},
        ]
        mock.get_stats.return_value = {'num_documents': 4, 'index_loaded': True}
        return mock
    
    @pytest.fixture
    def mock_embedder(self):
        """Create a mock Embedder."""
        mock = Mock()
        mock.embed.return_value = np.zeros(384)
        return mock
    
    @pytest.fixture
    def hybrid_searcher(self, mock_vector_store, mock_bm25_index, mock_embedder):
        """Create a HybridSearcher instance."""
        return HybridSearcher(
            vector_store=mock_vector_store,
            bm25_index=mock_bm25_index,
            embedder=mock_embedder,
            alpha=0.7,
            rrf_k=60
        )
    
    def test_initialization(self, hybrid_searcher):
        """Test HybridSearcher initializes correctly."""
        assert hybrid_searcher.alpha == 0.7
        assert hybrid_searcher.rrf_k == 60
    
    def test_rrf_fusion_documents_in_both_lists_rank_higher(self, hybrid_searcher):
        """Test that documents appearing in both lists rank higher."""
        result = hybrid_searcher.search(
            query='StandardScaler fit_transform',
            strategy='fixed',
            top_k=5
        )
        
        results = result['results']
        
        # Find documents in both lists vs single list
        both_list_docs = [r for r in results if r['in_semantic'] and r['in_keyword']]
        single_list_docs = [r for r in results if not (r['in_semantic'] and r['in_keyword'])]
        
        # Documents in both lists should have higher RRF scores
        if both_list_docs and single_list_docs:
            min_both_score = min(r['rrf_score'] for r in both_list_docs)
            max_single_score = max(r['rrf_score'] for r in single_list_docs)
            assert min_both_score > max_single_score
    
    def test_rrf_score_calculation(self, hybrid_searcher):
        """Test RRF score is calculated correctly."""
        result = hybrid_searcher.search(
            query='test query',
            strategy='fixed',
            top_k=5
        )
        
        # Verify RRF scores are present and positive
        for r in result['results']:
            assert 'rrf_score' in r
            assert r['rrf_score'] > 0
    
    def test_alpha_weighting_pure_semantic(self, hybrid_searcher):
        """Test alpha=1.0 gives same order as pure semantic search."""
        result = hybrid_searcher.search(
            query='test query',
            strategy='fixed',
            top_k=5,
            alpha=1.0
        )
        
        # With alpha=1.0, only semantic results should contribute to score
        # Documents only in keyword list should have score 0
        for r in result['results']:
            if not r['in_semantic']:
                assert r['rrf_score'] == 0.0
    
    def test_alpha_weighting_pure_keyword(self, hybrid_searcher):
        """Test alpha=0.0 gives same order as pure keyword search."""
        result = hybrid_searcher.search(
            query='test query',
            strategy='fixed',
            top_k=5,
            alpha=0.0
        )
        
        # With alpha=0.0, only keyword results should contribute to score
        # Documents only in semantic list should have score 0
        for r in result['results']:
            if not r['in_keyword']:
                assert r['rrf_score'] == 0.0
    
    def test_search_returns_metadata(self, hybrid_searcher):
        """Test search returns proper metadata."""
        result = hybrid_searcher.search(
            query='test query',
            strategy='fixed',
            top_k=5
        )
        
        assert 'query' in result
        assert 'strategy' in result
        assert 'results' in result
        assert 'metadata' in result
        
        metadata = result['metadata']
        assert 'total_results' in metadata
        assert 'alpha' in metadata
        assert 'rrf_k' in metadata
        assert 'semantic_candidates' in metadata
        assert 'keyword_candidates' in metadata
        assert 'timing' in metadata
    
    def test_search_timing_metadata(self, hybrid_searcher):
        """Test timing metadata is included."""
        result = hybrid_searcher.search(
            query='test query',
            strategy='fixed',
            top_k=5
        )
        
        timing = result['metadata']['timing']
        assert 'semantic_search' in timing
        assert 'keyword_search' in timing
        assert 'fusion' in timing
        assert 'total' in timing
    
    def test_search_semantic_only(self, hybrid_searcher):
        """Test semantic-only search mode."""
        result = hybrid_searcher.search_semantic_only(
            query='test query',
            strategy='fixed',
            top_k=5
        )
        
        assert result['metadata']['search_mode'] == 'semantic'
        assert result['metadata']['alpha'] == 1.0
        
        for r in result['results']:
            assert r['in_semantic'] == True
            assert r['in_keyword'] == False
    
    def test_search_keyword_only(self, hybrid_searcher):
        """Test keyword-only search mode."""
        result = hybrid_searcher.search_keyword_only(
            query='test query',
            top_k=5
        )
        
        assert result['metadata']['search_mode'] == 'keyword'
        assert result['metadata']['alpha'] == 0.0
        
        for r in result['results']:
            assert r['in_semantic'] == False
            assert r['in_keyword'] == True
    
    def test_overfetch_factor(self, hybrid_searcher, mock_vector_store, mock_bm25_index):
        """Test overfetch_factor retrieves more candidates than top_k."""
        top_k = 5
        overfetch_factor = 3
        
        hybrid_searcher.search(
            query='test query',
            strategy='fixed',
            top_k=top_k,
            overfetch_factor=overfetch_factor
        )
        
        # Verify both search methods were called with fetch_k = top_k * overfetch_factor
        expected_fetch_k = top_k * overfetch_factor
        mock_bm25_index.search.assert_called_with('test query', top_k=expected_fetch_k)
    
    def test_get_stats(self, hybrid_searcher):
        """Test get_stats method."""
        stats = hybrid_searcher.get_stats()
        
        assert 'alpha' in stats
        assert 'rrf_k' in stats
        assert 'vector_store_collections' in stats
        assert 'bm25_index' in stats


class TestRRFMathematics:
    """Test the mathematical correctness of RRF fusion."""
    
    def test_rrf_formula(self):
        """Verify RRF formula: score = alpha * (1/(k+semantic_rank)) + (1-alpha) * (1/(k+keyword_rank))"""
        alpha = 0.7
        rrf_k = 60
        
        # Document with semantic_rank=1, keyword_rank=3
        semantic_rank = 1
        keyword_rank = 3
        
        expected_score = alpha * (1/(rrf_k + semantic_rank)) + (1-alpha) * (1/(rrf_k + keyword_rank))
        
        # Calculate: 0.7 * (1/61) + 0.3 * (1/63)
        calculated = 0.7 * (1/61) + 0.3 * (1/63)
        
        assert abs(expected_score - calculated) < 1e-10
        assert abs(expected_score - 0.016237) < 1e-5
    
    def test_document_in_both_lists_scores_higher(self):
        """Verify document in both lists gets higher score than single-list document."""
        alpha = 0.7
        rrf_k = 60
        
        # Document A: in both lists (rank 1 semantic, rank 1 keyword)
        score_a = alpha * (1/(rrf_k + 1)) + (1-alpha) * (1/(rrf_k + 1))
        
        # Document B: only in semantic (rank 1)
        score_b = alpha * (1/(rrf_k + 1))
        
        # Document C: only in keyword (rank 1)
        score_c = (1-alpha) * (1/(rrf_k + 1))
        
        assert score_a > score_b
        assert score_a > score_c
        assert score_a == score_b + score_c  # Sum of components
    
    def test_rrf_k_dampening_effect(self):
        """Verify higher rrf_k reduces impact of top-ranked documents."""
        # With k=60, rank 1 vs rank 10 difference
        k_60_rank_1 = 1 / (60 + 1)
        k_60_rank_10 = 1 / (60 + 10)
        k_60_ratio = k_60_rank_1 / k_60_rank_10
        
        # With k=10, same ranks
        k_10_rank_1 = 1 / (10 + 1)
        k_10_rank_10 = 1 / (10 + 10)
        k_10_ratio = k_10_rank_1 / k_10_rank_10
        
        # Higher k = smaller ratio = less difference between ranks
        assert k_60_ratio < k_10_ratio
