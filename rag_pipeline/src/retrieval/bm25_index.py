"""BM25 keyword search implementation."""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Set
from pathlib import Path
import json
import pickle
import re

from rank_bm25 import BM25Okapi

from ..utils.logger import get_logger


class BM25Index:
    """
    BM25 index for keyword-based document retrieval.
    
    Builds an inverted index over document chunks and provides
    BM25-scored keyword search. Uses Porter stemming and stopword
    removal for improved matching.
    """
    
    # Default English stopwords (fallback if NLTK unavailable)
    DEFAULT_STOPWORDS: Set[str] = {
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
        "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself',
        'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her',
        'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them',
        'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom',
        'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was',
        'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
        'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
        'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with',
        'about', 'against', 'between', 'into', 'through', 'during', 'before',
        'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out',
        'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
        'here', 'there', 'when', 'where', 'why', 'how', 'all', 'each', 'few',
        'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
        'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will',
        'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm',
        'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't",
        'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn',
        "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn',
        "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't",
        'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won',
        "won't", 'wouldn', "wouldn't"
    }
    
    def __init__(
        self,
        persist_directory: Path,
        logger_name: str = "bm25_index"
    ):
        """
        Initialize BM25 index.
        
        Args:
            persist_directory: Directory to save/load the index
            logger_name: Logger name
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(logger_name)
        
        self.index: Optional[BM25Okapi] = None
        self.chunk_ids: List[str] = []  # Parallel list mapping index position to chunk_id
        self.chunk_metadata: Dict[str, Dict] = {}  # chunk_id -> metadata
        self._loaded_strategy: Optional[str] = None  # Track which strategy is currently loaded
        
        # Initialize stemmer and stopwords
        self.stemmer = None
        self.stopwords: Set[str] = self.DEFAULT_STOPWORDS
        self._initialize_nlp()

    @property
    def loaded_strategy(self) -> Optional[str]:
        """Name of the strategy whose index is currently loaded, or None."""
        return self._loaded_strategy

    def _initialize_nlp(self) -> None:
        """Initialize NLTK stemmer and stopwords with fallback."""
        try:
            import nltk
            from nltk.stem import PorterStemmer
            
            # Download stopwords if not already present
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                self.logger.info("Downloading NLTK stopwords...")
                nltk.download('stopwords', quiet=True)
            
            from nltk.corpus import stopwords
            self.stopwords = set(stopwords.words('english'))
            self.stemmer = PorterStemmer()
            self.logger.info("NLTK stemmer and stopwords initialized")
            
        except ImportError:
            self.logger.warning(
                "NLTK not available, using default stopwords and no stemming"
            )
        except Exception as e:
            self.logger.warning(
                f"Failed to initialize NLTK: {e}. Using fallback."
            )
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25 indexing.
        
        Applies lowercasing, regex tokenization, stopword removal,
        and Porter stemming for improved keyword matching.
        
        Args:
            text: Input text
            
        Returns:
            List of processed tokens
        """
        # Lowercase
        text = text.lower()
        
        # Split on non-alphanumeric characters, preserving underscores
        # (important for sklearn terms like fit_transform)
        tokens = re.findall(r'\b[a-z0-9_]+\b', text)
        
        # Remove stopwords
        tokens = [t for t in tokens if t not in self.stopwords]
        
        # Apply stemming if available
        if self.stemmer is not None:
            tokens = [self.stemmer.stem(t) for t in tokens]
        
        # Remove very short tokens (likely not meaningful)
        tokens = [t for t in tokens if len(t) > 1]
        
        return tokens
    
    def build_index(
        self,
        chunks: List[Dict[str, Any]],
        strategy_name: str
    ) -> None:
        """
        Build BM25 index from chunks.
        
        Args:
            chunks: List of chunk dictionaries with 'chunk_id', 'content', 'doc_id', etc.
            strategy_name: Name of chunking strategy (for file naming)
        """
        self.logger.info(f"Building BM25 index for {len(chunks)} chunks...")
        
        # Tokenize all documents
        tokenized_docs = []
        self.chunk_ids = []
        self.chunk_metadata = {}
        
        for chunk in chunks:
            chunk_id = chunk['chunk_id']
            content = chunk.get('content', '')
            
            tokens = self.tokenize(content)
            tokenized_docs.append(tokens)
            self.chunk_ids.append(chunk_id)
            
            # Store metadata for retrieval
            self.chunk_metadata[chunk_id] = {
                'doc_id': chunk.get('doc_id'),
                'content': content,
                'chunk_index': chunk.get('chunk_index'),
                'metadata': chunk.get('metadata', {})
            }
        
        # Build BM25 index
        self.index = BM25Okapi(tokenized_docs)
        self._loaded_strategy = strategy_name
        
        self.logger.info(f"BM25 index built with {len(tokenized_docs)} documents")
        
        # Persist to disk
        self._save_index(strategy_name)
    
    def _save_index(self, strategy_name: str) -> None:
        """Save index to disk."""
        index_path = self.persist_directory / f"bm25_{strategy_name}.pkl"
        metadata_path = self.persist_directory / f"bm25_{strategy_name}_metadata.json"
        
        # Save BM25 index and chunk_ids
        with open(index_path, 'wb') as f:
            pickle.dump({
                'index': self.index,
                'chunk_ids': self.chunk_ids
            }, f)
        
        # Save metadata as JSON (more readable/debuggable)
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.chunk_metadata, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"BM25 index saved to {index_path}")
    
    def load_index(self, strategy_name: str) -> bool:
        """
        Load index from disk.
        
        Args:
            strategy_name: Name of chunking strategy
            
        Returns:
            True if loaded successfully, False otherwise
        """
        index_path = self.persist_directory / f"bm25_{strategy_name}.pkl"
        metadata_path = self.persist_directory / f"bm25_{strategy_name}_metadata.json"
        
        if not index_path.exists() or not metadata_path.exists():
            self.logger.warning(f"BM25 index not found for strategy '{strategy_name}'")
            self._loaded_strategy = None
            return False
        
        try:
            with open(index_path, 'rb') as f:
                data = pickle.load(f)
                self.index = data['index']
                self.chunk_ids = data['chunk_ids']
            
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.chunk_metadata = json.load(f)
            
            self._loaded_strategy = strategy_name
            self.logger.info(f"BM25 index loaded: {len(self.chunk_ids)} documents")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load BM25 index: {e}")
            self._loaded_strategy = None
            return False
    
    def search(
        self,
        query: str,
        top_k: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Search the index using BM25 scoring.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of result dictionaries with chunk_id, content, score, etc.
        """
        if self.index is None:
            raise RuntimeError("BM25 index not loaded. Call load_index() first.")
        
        # Tokenize query using same preprocessing as documents
        query_tokens = self.tokenize(query)
        
        if not query_tokens:
            self.logger.warning("Query tokenized to empty list")
            return []
        
        # Get BM25 scores for all documents
        scores = self.index.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]
        
        # Build results
        results = []
        for rank, idx in enumerate(top_indices, start=1):
            chunk_id = self.chunk_ids[idx]
            score = scores[idx]
            
            if score <= 0:
                continue  # Skip zero-score documents
            
            metadata = self.chunk_metadata.get(chunk_id, {})
            
            results.append({
                'chunk_id': chunk_id,
                'doc_id': metadata.get('doc_id'),
                'content': metadata.get('content', ''),
                'bm25_score': float(score),
                'bm25_rank': rank,
                'metadata': metadata.get('metadata', {})
            })
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            'num_documents': len(self.chunk_ids) if self.chunk_ids else 0,
            'index_loaded': self.index is not None,
            'stemmer_available': self.stemmer is not None,
            'stopwords_count': len(self.stopwords)
        }
    
    def index_exists(self, strategy_name: str) -> bool:
        """
        Check if a BM25 index exists for the given strategy.
        
        Args:
            strategy_name: Name of chunking strategy
            
        Returns:
            True if index files exist, False otherwise
        """
        index_path = self.persist_directory / f"bm25_{strategy_name}.pkl"
        metadata_path = self.persist_directory / f"bm25_{strategy_name}_metadata.json"
        return index_path.exists() and metadata_path.exists()
