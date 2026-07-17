"""Semantic chunking based on natural language boundaries."""

from __future__ import annotations
import re
from typing import Dict, List, Any

from .base_chunker import BaseChunker, Chunk


class SemanticChunker(BaseChunker):
    """
    Chunks documents based on semantic boundaries (sentences/paragraphs).

    Attempts to keep semantically related content together by splitting
    on natural boundaries and building chunks that don't exceed a maximum size.
    A single unit longer than max_chunk_size is split at word boundaries so
    no emitted chunk ever exceeds the limit.

    Known limitation: sentence splitting is regex-based (punctuation followed
    by whitespace and a capital letter), so abbreviations like "e.g. The" or
    "Fig. 3" can produce false sentence boundaries. Acceptable here because
    chunks are built from *groups* of sentences, so an occasional bad split
    only moves a boundary by a few words; a proper tokenizer (spaCy/nltk)
    would be the upgrade path if unit fidelity started to matter.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize semantic chunker.
        
        Args:
            config: Configuration with 'max_chunk_size' and 'method'
        """
        super().__init__(config)
        self.max_chunk_size = self.config.get("max_chunk_size", 1000)  # in words
        self.method = self.config.get("method", "sentence")  # "sentence" or "paragraph"
        
    def get_strategy_name(self) -> str:
        """Get strategy name."""
        return "semantic"
    
    def chunk_document(self, document: Dict[str, Any]) -> List[Chunk]:
        """
        Chunk document based on semantic boundaries.
        
        Args:
            document: Document dictionary with 'content', 'doc_id', etc.
            
        Returns:
            List of Chunk objects
        """
        content = document.get("content", "")
        doc_id = document.get("doc_id", "unknown")
        
        if not content or not content.strip():
            return []
        
        # Split into semantic units (sentences or paragraphs)
        if self.method == "paragraph":
            units = self._split_paragraphs(content)
        else:  # Default to sentence
            units = self._split_sentences(content)
        
        if not units:
            return []

        # A single unit longer than the limit (e.g. a huge paragraph, or a
        # wall of text the sentence regex could not split) would previously
        # become an oversized chunk; split such units at word boundaries first.
        units = [piece for unit in units for piece in self._split_oversized_unit(unit)]

        # Group units into chunks that don't exceed max size
        chunks = []
        current_chunk_units = []
        current_word_count = 0
        chunk_index = 0

        for unit in units:
            unit_words = len(unit.split())
            
            # If adding this unit would exceed max size and we have content, finalize chunk
            if current_word_count + unit_words > self.max_chunk_size and current_chunk_units:
                chunk_text = " ".join(current_chunk_units)
                chunks.append(self._create_chunk_object(
                    chunk_text, doc_id, chunk_index, document, len(current_chunk_units)
                ))
                
                current_chunk_units = []
                current_word_count = 0
                chunk_index += 1
            
            # Add unit to current chunk
            current_chunk_units.append(unit)
            current_word_count += unit_words
        
        # Add final chunk if there's remaining content
        if current_chunk_units:
            chunk_text = " ".join(current_chunk_units)
            chunks.append(self._create_chunk_object(
                chunk_text, doc_id, chunk_index, document, len(current_chunk_units)
            ))
        
        return chunks
    
    def _create_chunk_object(
        self,
        chunk_text: str,
        doc_id: str,
        chunk_index: int,
        document: Dict[str, Any],
        unit_count: int
    ) -> Chunk:
        """Create a Chunk object with metadata."""
        base_metadata = self._extract_metadata(document)
        base_metadata.update({
            "max_chunk_size": self.max_chunk_size,
            "method": self.method,
            "word_count": len(chunk_text.split()),
            "char_count": len(chunk_text),
            "unit_count": unit_count,  # number of sentences/paragraphs
        })
        
        return Chunk(
            content=chunk_text,
            chunk_id=self._create_chunk_id(doc_id, chunk_index),
            doc_id=doc_id,
            chunk_index=chunk_index,
            metadata=base_metadata,
        )
    
    def _split_oversized_unit(self, unit: str) -> List[str]:
        """
        Split a unit exceeding max_chunk_size into word-boundary pieces.

        Units at or under the limit pass through unchanged (the common case).

        Args:
            unit: A single sentence or paragraph

        Returns:
            List of pieces, each at most max_chunk_size words
        """
        words = unit.split()
        if len(words) <= self.max_chunk_size:
            return [unit]

        return [
            " ".join(words[i : i + self.max_chunk_size])
            for i in range(0, len(words), self.max_chunk_size)
        ]

    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Uses a simple regex-based approach. For better sentence splitting,
        could integrate spaCy or nltk, but this works well for most cases.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        # Simple sentence boundary detection
        # Handles: . ! ? followed by space and capital letter or end of string
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])$'
        sentences = re.split(sentence_pattern, text)
        
        # Clean up and filter empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def _split_paragraphs(self, text: str) -> List[str]:
        """
        Split text into paragraphs.
        
        Args:
            text: Input text
            
        Returns:
            List of paragraphs
        """
        # Split on double newlines or multiple newlines
        paragraphs = re.split(r'\n\s*\n', text)
        
        # Clean up and filter empty paragraphs
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        return paragraphs

