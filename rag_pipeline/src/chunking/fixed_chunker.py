"""Fixed-size chunking with overlap."""

from __future__ import annotations
from typing import Dict, List, Any

from .base_chunker import BaseChunker, Chunk


class FixedSizeChunker(BaseChunker):
    """
    Chunks documents into fixed-size pieces with overlap.
    
    Uses a simple word-based approximation where ~1 token ≈ 0.75 words.
    For more precise token counting, could integrate tiktoken, but this
    is sufficient for most use cases.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize fixed-size chunker.
        
        Args:
            config: Configuration with 'chunk_size' and 'overlap' (in tokens)
        """
        super().__init__(config)
        # Default: 512 tokens ≈ 384 words
        self.chunk_size = self.config.get("chunk_size", 512)
        self.overlap = self.config.get("overlap", 50)

        # Convert tokens to approximate word count (1 token ≈ 0.75 words)
        self.chunk_size_words = int(self.chunk_size * 0.75)
        self.overlap_words = int(self.overlap * 0.75)

        # The chunk window advances by (chunk_size_words - overlap_words) each
        # iteration; if that is not positive the loop in chunk_document never
        # terminates. Validate on the converted word counts, not the raw token
        # values — the 0.75 conversion can collapse nearby values (e.g.
        # chunk_size=5, overlap=4 both floor to 3 words).
        if self.chunk_size_words < 1:
            raise ValueError(
                f"chunk_size={self.chunk_size} tokens converts to "
                f"{self.chunk_size_words} words; it must be at least 2 tokens"
            )
        if self.overlap < 0:
            raise ValueError(f"overlap must be >= 0 tokens, got {self.overlap}")
        if self.overlap_words >= self.chunk_size_words:
            raise ValueError(
                f"overlap ({self.overlap} tokens ≈ {self.overlap_words} words) must be "
                f"smaller than chunk_size ({self.chunk_size} tokens ≈ "
                f"{self.chunk_size_words} words), or the chunk window cannot advance"
            )
        
    def get_strategy_name(self) -> str:
        """Get strategy name."""
        return "fixed"
    
    def chunk_document(self, document: Dict[str, Any]) -> List[Chunk]:
        """
        Chunk document into fixed-size pieces with overlap.
        
        Args:
            document: Document dictionary with 'content', 'doc_id', etc.
            
        Returns:
            List of Chunk objects
        """
        content = document.get("content", "")
        doc_id = document.get("doc_id", "unknown")
        
        if not content or not content.strip():
            return []
        
        # Split into words
        words = content.split()
        
        if len(words) == 0:
            return []
        
        chunks = []
        chunk_index = 0
        start_idx = 0
        
        while start_idx < len(words):
            # Extract chunk with fixed size (in words)
            # Example: if chunk_size=512 tokens (~384 words), this takes 384 words
            end_idx = start_idx + self.chunk_size_words
            chunk_words = words[start_idx:end_idx]
            chunk_text = " ".join(chunk_words)
            
            # Create chunk object with metadata tracking size and overlap
            base_metadata = self._extract_metadata(document)
            base_metadata.update({
                "chunk_size": self.chunk_size,
                "overlap": self.overlap,
                "word_count": len(chunk_words),
                "char_count": len(chunk_text),
            })
            
            chunk = Chunk(
                content=chunk_text,
                chunk_id=self._create_chunk_id(doc_id, chunk_index),
                doc_id=doc_id,
                chunk_index=chunk_index,
                metadata=base_metadata,
            )
            chunks.append(chunk)
            
            # Move to next chunk with overlap to prevent information loss at boundaries
            # The overlap ensures important info at chunk boundaries isn't split
            # Example: chunk_size=384 words, overlap=38 words
            # Chunk 1: words 0-384, Chunk 2: words 346-730 (38 word overlap)
            if end_idx >= len(words):
                break
            
            start_idx = end_idx - self.overlap_words
            chunk_index += 1
        
        return chunks

