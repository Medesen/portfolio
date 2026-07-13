"""ChromaDB vector database wrapper."""

from __future__ import annotations
from typing import List, Dict, Any, Optional
from pathlib import Path
import chromadb
from chromadb.config import Settings

from ..utils.logger import get_logger


def distance_to_similarity(distance: float) -> float:
    """Convert a ChromaDB distance to a cosine similarity in [0, 1].

    Collections use ChromaDB's default ``l2`` space, whose distance is the
    *squared* Euclidean norm (no square root). For the normalized embeddings we
    store, ``||a - b||^2 = 2 * (1 - cos)``, so ``cos = 1 - distance / 2``.

    The previous inline code computed ``1 - distance ** 2 / 2``, which squared an
    already-squared distance — monotonic (so rankings were unaffected) but the
    reported similarity scores and any ``min_similarity`` filtering were wrong.
    """
    return max(0.0, 1.0 - distance / 2.0)


class VectorStore:
    """
    Wrapper for ChromaDB vector database.
    
    Manages collections for different chunking strategies and provides
    a clean interface for adding/retrieving embeddings.
    """
    
    def __init__(
        self,
        persist_directory: Path,
        logger_name: str = "vector_store"
    ):
        """
        Initialize vector store.
        
        Args:
            persist_directory: Directory to persist the database
            logger_name: Logger name
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(logger_name)
        
        self.logger.info(f"Initializing ChromaDB at: {self.persist_directory}")
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False)
        )
        
        self.logger.info("ChromaDB initialized successfully")
    
    def create_collection(
        self,
        collection_name: str,
        embedding_dimension: int,
        metadata: Dict[str, Any] = None
    ) -> chromadb.Collection:
        """
        Create or get a collection.
        
        Args:
            collection_name: Name of the collection
            embedding_dimension: Dimension of embeddings
            metadata: Collection metadata
            
        Returns:
            ChromaDB collection
        """
        collection_metadata = metadata or {}
        collection_metadata["embedding_dimension"] = embedding_dimension
        
        try:
            collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata=collection_metadata
            )
            self.logger.info(
                f"Collection '{collection_name}' ready "
                f"(count: {collection.count()})"
            )
            return collection
        except Exception as e:
            self.logger.error(f"Error creating collection '{collection_name}': {e}")
            raise
    
    def add_chunks(
        self,
        collection_name: str,
        chunks: List[Dict[str, Any]],
        embedding_dimension: int
    ) -> int:
        """
        Add chunks with embeddings to a collection.
        
        Args:
            collection_name: Name of the collection
            chunks: List of chunk dictionaries with 'embedding', 'chunk_id', 'content', etc.
            embedding_dimension: Dimension of embeddings
            
        Returns:
            Number of chunks added
        """
        if not chunks:
            self.logger.warning("No chunks to add")
            return 0
        
        # Get or create collection
        collection = self.create_collection(collection_name, embedding_dimension)
        
        # Prepare data for ChromaDB
        ids = []
        embeddings = []
        documents = []
        metadatas = []
        
        for chunk in chunks:
            ids.append(chunk["chunk_id"])
            embeddings.append(chunk["embedding"])
            documents.append(chunk["content"])
            
            # Prepare metadata (exclude embedding and content)
            metadata = {
                "doc_id": chunk["doc_id"],
                "chunk_index": chunk["chunk_index"],
            }
            # Add all other metadata fields
            if "metadata" in chunk:
                for key, value in chunk["metadata"].items():
                    # ChromaDB metadata values must be str, int, float, or bool
                    if isinstance(value, (str, int, float, bool)):
                        metadata[key] = value
                    elif value is not None:
                        metadata[key] = str(value)
            
            metadatas.append(metadata)
        
        # Add to collection in batches
        batch_size = 1000
        total_added = 0
        
        for i in range(0, len(ids), batch_size):
            batch_end = min(i + batch_size, len(ids))
            
            collection.add(
                ids=ids[i:batch_end],
                embeddings=embeddings[i:batch_end],
                documents=documents[i:batch_end],
                metadatas=metadatas[i:batch_end]
            )
            
            total_added += (batch_end - i)
            self.logger.info(
                f"Added batch {i // batch_size + 1}: "
                f"{batch_end - i} chunks ({total_added}/{len(ids)} total)"
            )
        
        self.logger.info(
            f"Successfully added {total_added} chunks to collection '{collection_name}'"
        )
        return total_added
    
    def query(
        self,
        collection_name: str,
        query_embedding: List[float],
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Query a collection for similar chunks.
        
        Args:
            collection_name: Name of the collection
            query_embedding: Query embedding vector
            n_results: Number of results to return
            where: Metadata filter (e.g., {"doc_type": "guide"})
            
        Returns:
            Dictionary with ChromaDB query results in the format:
            {
                'ids': [[str, ...]],        # List of list of chunk IDs
                'documents': [[str, ...]],   # List of list of content strings
                'metadatas': [[dict, ...]], # List of list of metadata dicts
                'distances': [[float, ...]] # List of list of L2 distances
            }
            Note: Outer list is for multiple queries; we always query one at a time,
            so results are accessed as results['ids'][0], results['documents'][0], etc.
        """
        try:
            collection = self.client.get_collection(collection_name)
            
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where
            )
            
            return results
        except Exception as e:
            self.logger.error(f"Error querying collection '{collection_name}': {e}")
            raise
    
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """
        Get information about a collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Dictionary with collection information
        """
        try:
            collection = self.client.get_collection(collection_name)
            return {
                "name": collection_name,
                "count": collection.count(),
                "metadata": collection.metadata
            }
        except Exception as e:
            self.logger.warning(f"Collection '{collection_name}' not found: {e}")
            return None
    
    def list_collections(self) -> List[str]:
        """
        List all collections in the database.
        
        Returns:
            List of collection names
        """
        collections = self.client.list_collections()
        return [c.name for c in collections]
    
    def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection.
        
        Args:
            collection_name: Name of the collection to delete
            
        Returns:
            True if successful
        """
        try:
            self.client.delete_collection(collection_name)
            self.logger.info(f"Deleted collection '{collection_name}'")
            return True
        except Exception as e:
            self.logger.error(f"Error deleting collection '{collection_name}': {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get overall statistics about the vector store.
        
        Returns:
            Dictionary with stats
        """
        collections = self.list_collections()
        total_chunks = 0
        collection_stats = {}
        
        for coll_name in collections:
            info = self.get_collection_info(coll_name)
            if info:
                collection_stats[coll_name] = info
                total_chunks += info["count"]
        
        return {
            "total_collections": len(collections),
            "total_chunks": total_chunks,
            "collections": collection_stats
        }

