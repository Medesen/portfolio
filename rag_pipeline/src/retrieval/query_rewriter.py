"""Query rewriting with LLM for improved retrieval."""

from __future__ import annotations
from typing import Dict, Any, Optional
import hashlib

from ..generation.llm_client import OllamaClient
from ..utils.logger import get_logger


class QueryRewriter:
    """
    LLM-based query rewriter for scikit-learn documentation search.
    
    Rewrites queries to improve retrieval by:
    - Clarifying ambiguous phrases
    - Adding relevant synonyms (e.g., "normalize" → "normalize StandardScaler")
    - Removing conversational filler ("um", "like", "I want to know")
    - Expanding abbreviations (e.g., "PCA" → "Principal Component Analysis PCA")
    
    Includes LRU caching for repeated queries and graceful fallback on LLM failure.
    """
    
    # Written flush-left: a class-level indented literal would send 4+ spaces
    # of leading whitespace on every line to the LLM.
    REWRITE_PROMPT = """You are a query rewriter for a scikit-learn documentation search system.

Rewrite the following query to improve search results:
- Clarify ambiguous phrases
- Add relevant synonyms for scikit-learn concepts
- Remove conversational filler words (um, like, I want to know, how do I, etc.)
- Expand abbreviations while keeping the original
- Keep the query concise and focused on technical terms
- Preserve the core technical topic from the original query
- Ignore irrelevant content and focus only on the machine learning question

Examples:
- "um how do i normalize stuff" → "StandardScaler normalize features preprocessing"
- "SVM" → "Support Vector Machine SVM classification"
- "so I was eating pizza yesterday and wondered about cross validation lol" → "cross-validation model evaluation KFold"
- "RF for classification" → "Random Forest RandomForestClassifier ensemble"
- "my cat walked on my keyboard but anyway how to handle missing data" → "missing values imputation SimpleImputer"
- "PCA dimensionality" → "Principal Component Analysis PCA dimensionality reduction"
- "logistic regression example" → "LogisticRegression binary classification linear model"
- "k means clustering" → "KMeans clustering unsupervised learning"
- "yo bro I need help with decision trees and overfitting, gonna grab coffee brb" → "DecisionTreeClassifier overfitting pruning max_depth"
- "train test split" → "train_test_split model evaluation data splitting"
- "gradient boosting vs random forest" → "GradientBoostingClassifier RandomForestClassifier ensemble comparison"
- "how to tune hyperparameters" → "GridSearchCV hyperparameter tuning model selection"
- "neural network mlp" → "MLPClassifier neural network multi-layer perceptron"
- "I was at the gym thinking about feature selection what are the methods" → "feature selection SelectKBest RFE mutual_info"
- "confusion matrix accuracy" → "confusion_matrix accuracy_score precision recall metrics"
- "pipeline preprocessing" → "Pipeline ColumnTransformer preprocessing workflow"

Return ONLY the rewritten query, nothing else. No explanations, no quotes.

Original query: {query}
Rewritten query:"""

    def __init__(
        self,
        llm_client: Optional[OllamaClient] = None,
        config: Optional[Any] = None,
        enabled: bool = True,
        temperature: float = 0.3,
        max_tokens: int = 100,
        cache_size: int = 128,
        logger_name: str = "query_rewriter"
    ):
        """
        Initialize query rewriter.
        
        Args:
            llm_client: OllamaClient instance for LLM calls
            config: Configuration object (overrides other params if provided)
            enabled: Whether query rewriting is enabled
            temperature: LLM temperature (lower = more deterministic)
            max_tokens: Maximum tokens for rewritten query
            cache_size: Maximum number of cached rewrites
            logger_name: Logger name
        """
        self.logger = get_logger(logger_name)
        
        # Load from config if provided
        if config is not None:
            self.enabled = config.get("query_rewriting.enabled", enabled)
            self.temperature = config.get("query_rewriting.temperature", temperature)
            self.max_tokens = config.get("query_rewriting.max_tokens", max_tokens)
            self.cache_size = config.get("query_rewriting.cache_size", cache_size)
        else:
            self.enabled = enabled
            self.temperature = temperature
            self.max_tokens = max_tokens
            self.cache_size = cache_size
        
        self.llm_client = llm_client
        
        # Simple dict-based cache with LRU-like behavior
        self._cache: Dict[str, str] = {}
        self._cache_order: list = []
        
        self.logger.info(
            f"Query rewriter initialized (enabled={self.enabled}, "
            f"temperature={self.temperature}, cache_size={self.cache_size})"
        )
    
    def _get_cache_key(self, query: str) -> str:
        """Generate a cache key for a query."""
        return hashlib.md5(query.lower().strip().encode()).hexdigest()
    
    def _get_from_cache(self, query: str) -> Optional[str]:
        """Get a cached rewrite if available, refreshing its recency.

        Moving the key to the end of the eviction order on read is what makes
        the cache LRU rather than FIFO: an entry that keeps being used is not
        the one evicted at capacity.
        """
        key = self._get_cache_key(query)
        if key in self._cache:
            self._cache_order.remove(key)
            self._cache_order.append(key)
        return self._cache.get(key)
    
    def _add_to_cache(self, query: str, rewritten: str) -> None:
        """Add a rewrite to the cache with LRU eviction."""
        key = self._get_cache_key(query)
        
        # If key exists, update value and move to end of order
        if key in self._cache:
            self._cache_order.remove(key)
            self._cache_order.append(key)
            self._cache[key] = rewritten  # Update the value
            return
        
        # Evict oldest if at capacity
        while len(self._cache) >= self.cache_size and self._cache_order:
            oldest_key = self._cache_order.pop(0)
            self._cache.pop(oldest_key, None)
        
        # Add new entry
        self._cache[key] = rewritten
        self._cache_order.append(key)
    
    def rewrite(self, query: str) -> Dict[str, Any]:
        """
        Rewrite a query for improved retrieval.
        
        Args:
            query: Original query text
            
        Returns:
            Dictionary with:
                - original_query: The original query
                - rewritten_query: The rewritten query (or original on failure)
                - from_cache: Whether result was from cache
                - rewrite_failed: Whether LLM rewriting failed
                - rewrite_skipped: Whether rewriting was skipped (disabled or no client)
        """
        result = {
            "original_query": query,
            "rewritten_query": query,
            "from_cache": False,
            "rewrite_failed": False,
            "rewrite_skipped": False,
        }
        
        # Check if rewriting is enabled
        if not self.enabled:
            self.logger.debug("Query rewriting is disabled")
            result["rewrite_skipped"] = True
            return result

        # Check if LLM client is available
        if self.llm_client is None:
            self.logger.warning("No LLM client available for query rewriting")
            result["rewrite_skipped"] = True
            return result

        # Normalize query for caching
        normalized_query = query.lower().strip()

        # Check cache first
        cached_result = self._get_from_cache(normalized_query)
        if cached_result is not None:
            self.logger.debug(f"Cache hit for query: '{normalized_query[:50]}...'")
            result["rewritten_query"] = cached_result
            result["from_cache"] = True
            return result

        # Generate rewritten query with LLM
        try:
            self.logger.info(f"Rewriting query: '{normalized_query[:50]}...'")

            prompt = self.REWRITE_PROMPT.format(query=normalized_query)

            response = self.llm_client.generate(
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            rewritten = response.get("response", "").strip()
            
            # Validate the rewritten query
            if rewritten and len(rewritten) > 0:
                # Clean up any quotes or extra whitespace
                rewritten = rewritten.strip('"\'').strip()
                
                # Don't use if it's too long (likely an explanation instead of a query)
                if len(rewritten) <= 512:
                    result["rewritten_query"] = rewritten
                    self._add_to_cache(normalized_query, rewritten)
                    self.logger.info(f"Rewritten to: '{rewritten[:50]}...'")
                else:
                    self.logger.warning(
                        f"Rewritten query too long ({len(rewritten)} chars), using original"
                    )
                    result["rewrite_failed"] = True
            else:
                self.logger.warning("Empty response from LLM, using original query")
                result["rewrite_failed"] = True
                
        except Exception as e:
            self.logger.warning(f"Query rewriting failed: {e}. Using original query.")
            result["rewrite_failed"] = True
        
        return result
    
    def clear_cache(self) -> None:
        """Clear the query cache."""
        self._cache.clear()
        self._cache_order.clear()
        self.logger.info("Query cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cache_size": len(self._cache),
            "max_cache_size": self.cache_size,
            "cache_utilization": len(self._cache) / self.cache_size if self.cache_size > 0 else 0,
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rewriter statistics."""
        return {
            "enabled": self.enabled,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "llm_available": self.llm_client is not None,
            **self.get_cache_stats(),
        }
