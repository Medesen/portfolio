"""Tests for LLM-based query rewriting."""

import pytest
from unittest.mock import Mock

from src.retrieval.query_rewriter import QueryRewriter


class TestQueryRewriter:
    """Tests for QueryRewriter class."""
    
    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock OllamaClient."""
        mock = Mock()
        mock.generate.return_value = {
            "response": "Principal Component Analysis PCA dimensionality reduction",
            "model": "llama3.2:3b",
            "done": True,
        }
        return mock
    
    @pytest.fixture
    def query_rewriter(self, mock_llm_client):
        """Create a QueryRewriter instance with mocked LLM."""
        return QueryRewriter(
            llm_client=mock_llm_client,
            enabled=True,
            temperature=0.3,
            max_tokens=100,
            cache_size=10,
        )
    
    @pytest.fixture
    def disabled_rewriter(self, mock_llm_client):
        """Create a disabled QueryRewriter instance."""
        return QueryRewriter(
            llm_client=mock_llm_client,
            enabled=False,
        )
    
    def test_initialization(self, query_rewriter):
        """Test QueryRewriter initializes correctly."""
        assert query_rewriter.enabled == True
        assert query_rewriter.temperature == 0.3
        assert query_rewriter.max_tokens == 100
        assert query_rewriter.cache_size == 10
        assert query_rewriter.llm_client is not None
    
    def test_rewrite_basic(self, query_rewriter, mock_llm_client):
        """Test basic query rewriting."""
        result = query_rewriter.rewrite("How do I use PCA?")
        
        assert result["original_query"] == "How do I use PCA?"
        assert result["rewritten_query"] == "Principal Component Analysis PCA dimensionality reduction"
        assert result["from_cache"] == False
        assert result["rewrite_failed"] == False
        assert result["rewrite_skipped"] == False
        
        # Verify LLM was called
        mock_llm_client.generate.assert_called_once()
    
    def test_rewrite_abbreviation_expansion(self, query_rewriter, mock_llm_client):
        """Test that abbreviations are expanded."""
        mock_llm_client.generate.return_value = {
            "response": "Principal Component Analysis PCA sklearn decomposition",
            "model": "llama3.2:3b",
            "done": True,
        }
        
        result = query_rewriter.rewrite("PCA")
        
        # Should contain expanded form
        assert "Principal Component Analysis" in result["rewritten_query"]
        # Should also keep original abbreviation
        assert "PCA" in result["rewritten_query"]
    
    def test_rewrite_filler_removal(self, query_rewriter, mock_llm_client):
        """Test that conversational filler is removed."""
        mock_llm_client.generate.return_value = {
            "response": "StandardScaler normalize features preprocessing",
            "model": "llama3.2:3b",
            "done": True,
        }
        
        result = query_rewriter.rewrite("Um, like, how do I normalize my data?")
        
        # Should not contain filler words
        assert "um" not in result["rewritten_query"].lower()
        assert "like" not in result["rewritten_query"].lower()
        # Should contain technical terms
        assert "StandardScaler" in result["rewritten_query"] or "normalize" in result["rewritten_query"]
    
    def test_rewrite_synonym_addition(self, query_rewriter, mock_llm_client):
        """Test that relevant synonyms are added."""
        mock_llm_client.generate.return_value = {
            "response": "cross-validation GridSearchCV hyperparameter tuning model selection",
            "model": "llama3.2:3b",
            "done": True,
        }
        
        result = query_rewriter.rewrite("cross validation")
        
        # Should contain synonyms/related terms
        assert "GridSearchCV" in result["rewritten_query"] or "hyperparameter" in result["rewritten_query"]
    
    def test_cache_hit(self, query_rewriter, mock_llm_client):
        """Test that repeated queries hit the cache."""
        # First call - cache miss
        result1 = query_rewriter.rewrite("How do I use PCA?")
        assert result1["from_cache"] == False
        assert mock_llm_client.generate.call_count == 1
        
        # Second call with same query - cache hit
        result2 = query_rewriter.rewrite("How do I use PCA?")
        assert result2["from_cache"] == True
        assert result2["rewritten_query"] == result1["rewritten_query"]
        # LLM should NOT be called again
        assert mock_llm_client.generate.call_count == 1
    
    def test_cache_case_insensitive(self, query_rewriter, mock_llm_client):
        """Test that cache lookup is case-insensitive."""
        # First call
        result1 = query_rewriter.rewrite("How do I use PCA?")
        assert result1["from_cache"] == False
        
        # Second call with different case - should still hit cache
        result2 = query_rewriter.rewrite("how do i use pca?")
        assert result2["from_cache"] == True
        assert mock_llm_client.generate.call_count == 1
    
    def test_cache_eviction(self, query_rewriter, mock_llm_client):
        """Test LRU cache eviction when full."""
        # Fill cache (size=10)
        for i in range(10):
            query_rewriter.rewrite(f"query {i}")
        
        assert mock_llm_client.generate.call_count == 10
        
        # Add one more - should evict oldest
        query_rewriter.rewrite("query 10")
        assert mock_llm_client.generate.call_count == 11
        
        # First query should be evicted, so this should miss cache
        result = query_rewriter.rewrite("query 0")
        assert result["from_cache"] == False
        assert mock_llm_client.generate.call_count == 12
    
    def test_fallback_on_exception(self, query_rewriter, mock_llm_client):
        """Test graceful fallback when LLM call fails."""
        mock_llm_client.generate.side_effect = Exception("Connection timeout")
        
        result = query_rewriter.rewrite("How do I use PCA?")
        
        # Should return original query
        assert result["original_query"] == "How do I use PCA?"
        assert result["rewritten_query"] == "How do I use PCA?"
        assert result["rewrite_failed"] == True
        assert result["from_cache"] == False
    
    def test_fallback_on_empty_response(self, query_rewriter, mock_llm_client):
        """Test fallback when LLM returns empty response."""
        mock_llm_client.generate.return_value = {
            "response": "",
            "model": "llama3.2:3b",
            "done": True,
        }
        
        result = query_rewriter.rewrite("How do I use PCA?")
        
        assert result["rewritten_query"] == "How do I use PCA?"
        assert result["rewrite_failed"] == True
    
    def test_fallback_on_too_long_response(self, query_rewriter, mock_llm_client):
        """Test fallback when LLM returns overly long response (likely explanation)."""
        mock_llm_client.generate.return_value = {
            "response": "A" * 600,  # Too long, probably an explanation
            "model": "llama3.2:3b",
            "done": True,
        }
        
        result = query_rewriter.rewrite("How do I use PCA?")
        
        assert result["rewritten_query"] == "How do I use PCA?"
        assert result["rewrite_failed"] == True
    
    def test_disabled_rewriter(self, disabled_rewriter, mock_llm_client):
        """Test that disabled rewriter returns original query."""
        result = disabled_rewriter.rewrite("How do I use PCA?")
        
        assert result["original_query"] == "How do I use PCA?"
        assert result["rewritten_query"] == "How do I use PCA?"
        assert result["rewrite_skipped"] == True
        assert result["rewrite_failed"] == False
        
        # LLM should NOT be called
        mock_llm_client.generate.assert_not_called()
    
    def test_no_llm_client(self):
        """Test rewriter without LLM client."""
        rewriter = QueryRewriter(llm_client=None, enabled=True)
        
        result = rewriter.rewrite("How do I use PCA?")
        
        assert result["rewritten_query"] == "How do I use PCA?"
        assert result["rewrite_skipped"] == True
    
    def test_strips_quotes_from_response(self, query_rewriter, mock_llm_client):
        """Test that quotes are stripped from LLM response."""
        mock_llm_client.generate.return_value = {
            "response": '"Principal Component Analysis PCA"',
            "model": "llama3.2:3b",
            "done": True,
        }
        
        result = query_rewriter.rewrite("PCA")
        
        # Should not have quotes
        assert result["rewritten_query"] == "Principal Component Analysis PCA"
    
    def test_clear_cache(self, query_rewriter, mock_llm_client):
        """Test cache clearing."""
        # Populate cache
        query_rewriter.rewrite("query 1")
        query_rewriter.rewrite("query 2")
        
        stats = query_rewriter.get_cache_stats()
        assert stats["cache_size"] == 2
        
        # Clear cache
        query_rewriter.clear_cache()
        
        stats = query_rewriter.get_cache_stats()
        assert stats["cache_size"] == 0
        
        # Next call should miss cache
        result = query_rewriter.rewrite("query 1")
        assert result["from_cache"] == False
    
    def test_get_stats(self, query_rewriter):
        """Test get_stats method."""
        stats = query_rewriter.get_stats()
        
        assert "enabled" in stats
        assert "temperature" in stats
        assert "max_tokens" in stats
        assert "llm_available" in stats
        assert "cache_size" in stats
        assert "max_cache_size" in stats
        assert "cache_utilization" in stats
        
        assert stats["enabled"] == True
        assert stats["llm_available"] == True
        assert stats["max_cache_size"] == 10


class TestQueryRewriterWithConfig:
    """Tests for QueryRewriter with config object."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock config object."""
        config = Mock()
        config.get.side_effect = lambda key, default=None: {
            "query_rewriting.enabled": True,
            "query_rewriting.temperature": 0.2,
            "query_rewriting.max_tokens": 150,
            "query_rewriting.cache_size": 256,
        }.get(key, default)
        return config
    
    def test_config_loading(self, mock_config):
        """Test that config values are loaded correctly."""
        rewriter = QueryRewriter(config=mock_config)
        
        assert rewriter.enabled == True
        assert rewriter.temperature == 0.2
        assert rewriter.max_tokens == 150
        assert rewriter.cache_size == 256
    
    def test_config_disabled(self):
        """Test disabled via config."""
        config = Mock()
        config.get.side_effect = lambda key, default=None: {
            "query_rewriting.enabled": False,
        }.get(key, default)
        
        rewriter = QueryRewriter(config=config)
        
        assert rewriter.enabled == False


class TestQueryRewriterPrompt:
    """Tests for the rewrite prompt content."""
    
    def test_prompt_contains_sklearn_context(self):
        """Test that prompt mentions scikit-learn."""
        prompt = QueryRewriter.REWRITE_PROMPT
        
        assert "scikit-learn" in prompt.lower()
    
    def test_prompt_mentions_abbreviation_expansion(self):
        """Test that prompt instructs to expand abbreviations."""
        prompt = QueryRewriter.REWRITE_PROMPT
        
        assert "abbreviation" in prompt.lower()
        assert "PCA" in prompt
    
    def test_prompt_mentions_filler_removal(self):
        """Test that prompt instructs to remove filler."""
        prompt = QueryRewriter.REWRITE_PROMPT
        
        assert "filler" in prompt.lower()
    
    def test_prompt_mentions_synonyms(self):
        """Test that prompt instructs to add synonyms."""
        prompt = QueryRewriter.REWRITE_PROMPT
        
        assert "synonym" in prompt.lower()
    
    def test_prompt_has_query_placeholder(self):
        """Test that prompt has placeholder for query."""
        prompt = QueryRewriter.REWRITE_PROMPT
        
        assert "{query}" in prompt
