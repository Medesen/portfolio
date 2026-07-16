"""Indexing orchestrator for building the vector database."""

from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional

from ..utils.config import Config
from ..utils.logger import get_logger
from ..chunking import FixedSizeChunker, SemanticChunker, HierarchicalChunker
from .embedder import Embedder
from .vector_store import VectorStore
from .bm25_index import BM25Index


class IndexingStateTracker:
    """Track indexing state to avoid redundant operations."""
    
    def __init__(self, state_path: Path, vector_store_dir: Optional[Path] = None):
        """
        Initialize state tracker.
        
        Args:
            state_path: Path to state JSON file
            vector_store_dir: Path to vector store directory for validation
        """
        self.state_path = state_path
        self.vector_store_dir = vector_store_dir
        self.state = self._load_state()
    
    def _load_state(self) -> Dict:
        """Load state from file or return empty state."""
        if self.state_path.exists():
            try:
                with open(self.state_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return self._empty_state()
        return self._empty_state()
    
    def _empty_state(self) -> Dict:
        """Return empty state structure."""
        return {
            "strategies": {},  # strategy_name -> {completed, timestamp, chunk_count, doc_count}
            "bm25_strategies": {}  # strategy_name -> {completed, timestamp, chunk_count}
        }
    
    def save_state(self):
        """Save current state to file."""
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2)
    
    def is_strategy_indexed(self, strategy_name: str, vector_store: VectorStore) -> bool:
        """
        Check if a strategy has been indexed.
        
        Validates that state says it's complete AND that the collection exists
        with the expected number of chunks.
        
        Args:
            strategy_name: Name of the chunking strategy
            vector_store: VectorStore instance for validation
            
        Returns:
            True if indexed and validated, False otherwise
        """
        # Check if state says it's complete
        if strategy_name not in self.state["strategies"]:
            return False
        
        strategy_state = self.state["strategies"][strategy_name]
        if not strategy_state.get("completed", False):
            return False
        
        # Validate that collection actually exists with chunks
        if vector_store:
            collection_info = vector_store.get_collection_info(strategy_name)
            if collection_info is None:
                # Collection doesn't exist - mark as incomplete
                self.state["strategies"][strategy_name]["completed"] = False
                self.save_state()
                return False
            
            # Check chunk count matches (exact match required)
            expected_count = strategy_state.get("chunk_count", 0)
            actual_count = collection_info.get("count", 0)
            
            if expected_count > 0 and actual_count != expected_count:
                # Mismatch - mark as incomplete
                self.state["strategies"][strategy_name]["completed"] = False
                self.save_state()
                return False
        
        return True
    
    def is_bm25_indexed(self, strategy_name: str, bm25_index: "BM25Index") -> bool:
        """
        Check if BM25 index has been built for a strategy.
        
        Args:
            strategy_name: Name of the chunking strategy
            bm25_index: BM25Index instance for validation
            
        Returns:
            True if BM25 indexed and validated, False otherwise
        """
        # Ensure bm25_strategies key exists (for backward compatibility)
        if "bm25_strategies" not in self.state:
            self.state["bm25_strategies"] = {}
        
        # Check if state says it's complete
        if strategy_name not in self.state["bm25_strategies"]:
            return False
        
        bm25_state = self.state["bm25_strategies"][strategy_name]
        if not bm25_state.get("completed", False):
            return False
        
        # Validate that BM25 index files exist
        if bm25_index and not bm25_index.index_exists(strategy_name):
            # Index files don't exist - mark as incomplete
            self.state["bm25_strategies"][strategy_name]["completed"] = False
            self.save_state()
            return False
        
        return True
    
    def mark_strategy_completed(
        self, strategy_name: str, chunk_count: int, doc_count: int
    ):
        """Mark a strategy as indexed."""
        self.state["strategies"][strategy_name] = {
            "completed": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "chunk_count": chunk_count,
            "doc_count": doc_count,
        }
        self.save_state()
    
    def mark_bm25_completed(self, strategy_name: str, chunk_count: int):
        """Mark BM25 index as built for a strategy."""
        # Ensure bm25_strategies key exists
        if "bm25_strategies" not in self.state:
            self.state["bm25_strategies"] = {}
        
        self.state["bm25_strategies"][strategy_name] = {
            "completed": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "chunk_count": chunk_count,
        }
        self.save_state()
    
    def reset(self, strategy_name: Optional[str] = None):
        """
        Reset state.
        
        Args:
            strategy_name: If provided, reset only this strategy. Otherwise reset all.
        """
        if strategy_name:
            if strategy_name in self.state["strategies"]:
                del self.state["strategies"][strategy_name]
            # Also reset BM25 state for this strategy
            if "bm25_strategies" in self.state and strategy_name in self.state["bm25_strategies"]:
                del self.state["bm25_strategies"][strategy_name]
        else:
            self.state = self._empty_state()
        self.save_state()


class Indexer:
    """Orchestrate the indexing pipeline: chunk → embed → store."""
    
    def __init__(self, config: Config, logger_name: str = "indexer"):
        """
        Initialize indexer.
        
        Args:
            config: Configuration object
            logger_name: Name for the logger
        """
        self.config = config
        self.logger = get_logger(logger_name)
        
        # Setup paths
        self.processed_dir = config.get_path("paths.processed_dir")
        self.vector_store_dir = config.get_path("paths.vector_store_dir", create=True)
        self.state_dir = config.get_path("paths.state_dir", create=True)
        self.state_file = self.state_dir / "indexing_state.json"
        
        # Initialize components
        self.embedder = None  # Lazy initialization
        self.vector_store = VectorStore(self.vector_store_dir, logger_name="vector_store")
        self.bm25_index = BM25Index(
            persist_directory=self.vector_store_dir / "bm25",
            logger_name="bm25_index"
        )
        self.state_tracker = IndexingStateTracker(self.state_file, self.vector_store_dir)
        
        # Initialize chunkers
        self.chunkers = self._initialize_chunkers()
    
    def _initialize_chunkers(self) -> Dict[str, Any]:
        """Initialize chunking strategies based on config."""
        chunkers = {}
        
        strategies_config = self.config.get("chunking.strategies", {})
        
        if not strategies_config:
            self.logger.warning(
                "No chunking strategies found in config at 'chunking.strategies'. "
                "Indexing will not process any documents."
            )
            return chunkers
        
        # Fixed-size chunker
        fixed_config = strategies_config.get("fixed", {})
        if fixed_config.get("enabled", False):
            chunkers["fixed"] = FixedSizeChunker(fixed_config)
            self.logger.info("Fixed-size chunker enabled")
        elif "fixed" in strategies_config:
            self.logger.info("Fixed-size chunker configured but not enabled")
        
        # Semantic chunker
        semantic_config = strategies_config.get("semantic", {})
        if semantic_config.get("enabled", False):
            chunkers["semantic"] = SemanticChunker(semantic_config)
            self.logger.info("Semantic chunker enabled")
        elif "semantic" in strategies_config:
            self.logger.info("Semantic chunker configured but not enabled")
        
        # Hierarchical chunker
        hierarchical_config = strategies_config.get("hierarchical", {})
        if hierarchical_config.get("enabled", False):
            chunkers["hierarchical"] = HierarchicalChunker(hierarchical_config)
            self.logger.info("Hierarchical chunker enabled")
        elif "hierarchical" in strategies_config:
            self.logger.info("Hierarchical chunker configured but not enabled")
        
        if not chunkers:
            self.logger.warning(
                "No chunking strategies are enabled. "
                "Set 'enabled: true' for at least one strategy in config."
            )
        
        return chunkers

    def _get_embedder(self) -> Embedder:
        """Lazy initialization of embedder."""
        if self.embedder is None:
            model_name = self.config.get("embeddings.model", "all-MiniLM-L6-v2")
            device = self.config.get("embeddings.device", "cpu")
            batch_size = self.config.get("embeddings.batch_size", 32)
            
            self.embedder = Embedder(
                model_name=model_name,
                device=device,
                batch_size=batch_size,
                logger_name="embedder"
            )
        
        return self.embedder
    
    def index(
        self,
        strategy: Optional[str] = None,
        force_reindex: bool = False
    ):
        """
        Run the indexing pipeline.
        
        Args:
            strategy: Specific strategy to index ("fixed", "semantic", "hierarchical")
                     If None, index all enabled strategies
            force_reindex: Force re-indexing even if already done
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting INDEXING pipeline")
        self.logger.info("=" * 60)
        
        if force_reindex:
            self.logger.info("Force reindexing enabled")
            if strategy:
                self.state_tracker.reset(strategy)
            else:
                self.state_tracker.reset()
        
        # Determine which strategies to index
        if strategy:
            if strategy not in self.chunkers:
                raise ValueError(
                    f"Strategy '{strategy}' not enabled. "
                    f"Available: {list(self.chunkers.keys())}"
                )
            strategies_to_index = [strategy]
        else:
            strategies_to_index = list(self.chunkers.keys())
        
        if not strategies_to_index:
            self.logger.warning("No chunking strategies enabled in config")
            return
        
        # Load documents
        self.logger.info("Loading processed documents...")
        documents = self._load_documents()
        self.logger.info(f"Loaded {len(documents)} documents")
        
        # Index each strategy
        for strat_name in strategies_to_index:
            self._index_strategy(strat_name, documents, force_reindex)
        
        self.logger.info("=" * 60)
        self.logger.info("INDEXING pipeline completed")
        self.logger.info("=" * 60)
    
    def _index_strategy(
        self, strategy_name: str, documents: List[Dict], force_reindex: bool
    ):
        """Index documents with a specific chunking strategy."""
        # Check if already indexed
        if not force_reindex and self.state_tracker.is_strategy_indexed(
            strategy_name, self.vector_store
        ):
            self.logger.info(
                f"Strategy '{strategy_name}' already indexed (skipping)"
            )
            return
        
        self.logger.info(f"\nIndexing with strategy: {strategy_name}")
        self.logger.info("-" * 60)

        # Rebuilding: drop any existing collection first. add_chunks uses
        # get_or_create_collection, so without this a re-index after a
        # chunking-config change would interleave new chunk IDs with stale
        # ones in the same collection, and the resulting count would
        # permanently mismatch the state file.
        if self.vector_store.get_collection_info(strategy_name) is not None:
            self.logger.info(
                f"Dropping existing collection '{strategy_name}' before rebuild"
            )
            self.vector_store.delete_collection(strategy_name)

        # Get chunker
        chunker = self.chunkers[strategy_name]
        
        # Chunk all documents
        self.logger.info("Chunking documents...")
        all_chunks = chunker.chunk_documents(documents)
        self.logger.info(f"Created {len(all_chunks)} chunks")
        
        # Generate embeddings
        self.logger.info("Generating embeddings...")
        embedder = self._get_embedder()
        chunks_with_embeddings = embedder.embed_chunks(all_chunks, show_progress=True)
        
        # Store in vector database
        self.logger.info(f"Storing in vector database (collection: {strategy_name})...")
        embedding_dim = embedder.embedding_dim
        num_added = self.vector_store.add_chunks(
            collection_name=strategy_name,
            chunks=chunks_with_embeddings,
            embedding_dimension=embedding_dim
        )
        
        # Build BM25 index — always rebuilt alongside the collection: the
        # chunks it must mirror were just regenerated, so a stale "completed"
        # BM25 state must not leave a keyword index that no longer matches
        # the vector collection.
        self.logger.info(f"Building BM25 index for strategy '{strategy_name}'...")
        self.bm25_index.build_index(chunks_with_embeddings, strategy_name)
        self.state_tracker.mark_bm25_completed(strategy_name, len(all_chunks))
        
        # Mark as completed
        self.state_tracker.mark_strategy_completed(
            strategy_name, len(all_chunks), len(documents)
        )
        
        self.logger.info(
            f"✓ Strategy '{strategy_name}' indexed successfully "
            f"({num_added} chunks from {len(documents)} documents)"
        )
    
    def _load_documents(self) -> List[Dict]:
        """Load all processed documents."""
        documents = []
        failed_files = []
        
        # Walk through all processed JSON files
        for json_file in self.processed_dir.rglob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    doc = json.load(f)
                    documents.append(doc)
            except Exception as e:
                self.logger.warning(f"Failed to load {json_file}: {e}")
                failed_files.append(str(json_file))
        
        # Log summary of failures
        if failed_files:
            self.logger.warning(
                f"Failed to load {len(failed_files)} of {len(documents) + len(failed_files)} files. "
                f"First few failures: {failed_files[:5]}"
            )
        
        return documents
    
    def get_stats(self) -> Dict[str, Any]:
        """Get indexing statistics."""
        stats = {
            "vector_store": self.vector_store.get_stats(),
            "bm25_index": self.bm25_index.get_stats(),
            "strategies": {},
        }
        
        for strategy_name in self.chunkers.keys():
            is_indexed = self.state_tracker.is_strategy_indexed(
                strategy_name, self.vector_store
            )
            is_bm25_indexed = self.state_tracker.is_bm25_indexed(
                strategy_name, self.bm25_index
            )
            strategy_state = self.state_tracker.state.get("strategies", {}).get(
                strategy_name, {}
            )
            bm25_state = self.state_tracker.state.get("bm25_strategies", {}).get(
                strategy_name, {}
            )
            
            stats["strategies"][strategy_name] = {
                "indexed": is_indexed,
                "bm25_indexed": is_bm25_indexed,
                "chunk_count": strategy_state.get("chunk_count", 0),
                "doc_count": strategy_state.get("doc_count", 0),
                "timestamp": strategy_state.get("timestamp"),
                "bm25_timestamp": bm25_state.get("timestamp"),
            }
        
        return stats

