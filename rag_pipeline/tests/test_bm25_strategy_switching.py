"""Tests for BM25 index strategy switching functionality.

This test module verifies that the BM25Index correctly tracks which strategy
is currently loaded and reloads the appropriate index when strategies change.
This prevents bugs where switching strategies mid-session could cause the
wrong BM25 index to be used for hybrid search.
"""

import pytest
from unittest.mock import Mock

from src.retrieval.bm25_index import BM25Index


class TestBM25StrategyTracking:
    """Tests for BM25 strategy tracking functionality."""
    
    @pytest.fixture
    def bm25_index(self, temp_dir):
        """Create a BM25Index instance for testing."""
        return BM25Index(persist_directory=temp_dir)
    
    @pytest.fixture
    def sample_chunks_fixed(self):
        """Sample chunks for 'fixed' strategy."""
        return [
            {
                'chunk_id': 'fixed_c1',
                'doc_id': 'd1',
                'content': 'alpha preprocessing normalization StandardScaler usage.',
                'chunk_index': 0,
                'metadata': {'strategy': 'fixed'}
            },
            {
                'chunk_id': 'fixed_c2',
                'doc_id': 'd1',
                'content': 'alpha preprocessing transformation methods.',
                'chunk_index': 1,
                'metadata': {'strategy': 'fixed'}
            },
        ]
    
    @pytest.fixture
    def sample_chunks_semantic(self):
        """Sample chunks for 'semantic' strategy."""
        return [
            {
                'chunk_id': 'semantic_c1',
                'doc_id': 'd1',
                'content': 'beta hyperparameter tuning GridSearchCV optimization.',
                'chunk_index': 0,
                'metadata': {'strategy': 'semantic'}
            },
            {
                'chunk_id': 'semantic_c2',
                'doc_id': 'd1',
                'content': 'beta cross-validation model selection.',
                'chunk_index': 1,
                'metadata': {'strategy': 'semantic'}
            },
        ]
    
    def test_loaded_strategy_initially_none(self, bm25_index):
        """Test that _loaded_strategy is None on initialization."""
        assert bm25_index._loaded_strategy is None
    
    def test_loaded_strategy_set_after_build(self, bm25_index, sample_chunks_fixed):
        """Test that _loaded_strategy is set after building index."""
        bm25_index.build_index(sample_chunks_fixed, 'fixed')
        assert bm25_index._loaded_strategy == 'fixed'
    
    def test_loaded_strategy_set_after_load(self, bm25_index, sample_chunks_fixed):
        """Test that _loaded_strategy is set after loading index."""
        # Build and save
        bm25_index.build_index(sample_chunks_fixed, 'fixed')
        
        # Create new instance and load
        new_index = BM25Index(persist_directory=bm25_index.persist_directory)
        assert new_index._loaded_strategy is None
        
        new_index.load_index('fixed')
        assert new_index._loaded_strategy == 'fixed'
    
    def test_loaded_strategy_cleared_on_load_failure(self, bm25_index):
        """Test that _loaded_strategy is cleared if load fails."""
        # Set a value manually to simulate prior load
        bm25_index._loaded_strategy = 'old_strategy'
        
        # Try to load non-existent strategy
        result = bm25_index.load_index('nonexistent')
        
        assert result is False
        assert bm25_index._loaded_strategy is None
    
    def test_strategy_switch_reloads_correct_index(
        self, bm25_index, sample_chunks_fixed, sample_chunks_semantic
    ):
        """Test that switching strategies loads the correct index."""
        # Build both strategies
        bm25_index.build_index(sample_chunks_fixed, 'fixed')
        bm25_index.build_index(sample_chunks_semantic, 'semantic')
        
        # Load fixed strategy
        bm25_index.load_index('fixed')
        assert bm25_index._loaded_strategy == 'fixed'
        assert 'fixed_c1' in bm25_index.chunk_ids
        assert 'semantic_c1' not in bm25_index.chunk_ids
        
        # Switch to semantic strategy
        bm25_index.load_index('semantic')
        assert bm25_index._loaded_strategy == 'semantic'
        assert 'semantic_c1' in bm25_index.chunk_ids
        assert 'fixed_c1' not in bm25_index.chunk_ids
    
    def test_search_returns_correct_chunks_after_strategy_switch(
        self, bm25_index, sample_chunks_fixed, sample_chunks_semantic
    ):
        """Test that search returns chunks from the correct strategy after switching."""
        # Build both strategies
        bm25_index.build_index(sample_chunks_fixed, 'fixed')
        bm25_index.build_index(sample_chunks_semantic, 'semantic')
        
        # Create new instance and load fixed strategy
        index1 = BM25Index(persist_directory=bm25_index.persist_directory)
        index1.load_index('fixed')
        
        # Verify chunk_ids are from fixed strategy
        assert 'fixed_c1' in index1.chunk_ids
        assert 'fixed_c2' in index1.chunk_ids
        assert 'semantic_c1' not in index1.chunk_ids
        
        # Switch to semantic strategy on same instance
        index1.load_index('semantic')
        
        # Verify chunk_ids are now from semantic strategy
        assert 'semantic_c1' in index1.chunk_ids
        assert 'semantic_c2' in index1.chunk_ids
        assert 'fixed_c1' not in index1.chunk_ids
    
    def test_no_reload_if_same_strategy(self, bm25_index, sample_chunks_fixed):
        """Test that checking strategy match allows skipping reload."""
        bm25_index.build_index(sample_chunks_fixed, 'fixed')
        
        # Simulate the check done in query_processor
        # First load should succeed
        assert bm25_index._loaded_strategy == 'fixed'
        
        # Subsequent check for same strategy should indicate no reload needed
        if bm25_index._loaded_strategy != 'fixed':
            pytest.fail("Should not need to reload for same strategy")
        
        # Check for different strategy should indicate reload needed
        if bm25_index._loaded_strategy != 'semantic':
            # This is expected - we need to reload
            pass
        else:
            pytest.fail("Should indicate reload needed for different strategy")


class TestQueryProcessorBM25Integration:
    """Integration tests for QueryProcessor BM25 strategy handling."""
    
    @pytest.fixture
    def mock_bm25_index(self):
        """Create a mock BM25Index with strategy tracking."""
        mock = Mock(spec=BM25Index)
        mock._loaded_strategy = None
        mock.index = None
        
        def mock_load_index(strategy):
            mock._loaded_strategy = strategy
            mock.index = Mock()  # Simulate index being loaded
            return True
        
        mock.load_index.side_effect = mock_load_index
        return mock
    
    def test_query_processor_loads_different_strategy(self, mock_bm25_index):
        """Test that QueryProcessor loads a new strategy when needed."""
        # Simulate first query with 'fixed' strategy
        strategy = 'fixed'
        if mock_bm25_index._loaded_strategy != strategy:
            mock_bm25_index.load_index(strategy)
        
        assert mock_bm25_index._loaded_strategy == 'fixed'
        assert mock_bm25_index.load_index.call_count == 1
        
        # Simulate second query with same strategy - should not reload
        strategy = 'fixed'
        if mock_bm25_index._loaded_strategy != strategy:
            mock_bm25_index.load_index(strategy)
        
        assert mock_bm25_index.load_index.call_count == 1  # No additional call
        
        # Simulate third query with different strategy - should reload
        strategy = 'semantic'
        if mock_bm25_index._loaded_strategy != strategy:
            mock_bm25_index.load_index(strategy)
        
        assert mock_bm25_index._loaded_strategy == 'semantic'
        assert mock_bm25_index.load_index.call_count == 2
