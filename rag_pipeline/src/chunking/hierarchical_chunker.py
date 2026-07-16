"""Hierarchical chunking based on document structure."""

from __future__ import annotations
import re
from typing import Dict, List, Any, Optional, Tuple

from .base_chunker import BaseChunker, Chunk


class HierarchicalChunker(BaseChunker):
    """
    Chunks documents based on their hierarchical structure (sections, subsections).
    
    Uses the section headings extracted during preprocessing to create chunks
    that respect document structure, making them more semantically coherent.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize hierarchical chunker.
        
        Args:
            config: Configuration with 'max_chunk_size'
        """
        super().__init__(config)
        self.max_chunk_size = self.config.get("max_chunk_size", 1000)  # in words
        
    def get_strategy_name(self) -> str:
        """Get strategy name."""
        return "hierarchical"
    
    def chunk_document(self, document: Dict[str, Any]) -> List[Chunk]:
        """
        Chunk document based on hierarchical structure.
        
        Args:
            document: Document dictionary with 'content', 'doc_id', 'metadata.sections', etc.
            
        Returns:
            List of Chunk objects
        """
        content = document.get("content", "")
        doc_id = document.get("doc_id", "unknown")
        sections = document.get("metadata", {}).get("sections", [])
        
        if not content or not content.strip():
            return []
        
        # If no sections found, fall back to paragraph-based chunking
        if not sections:
            return self._chunk_by_paragraphs(content, doc_id, document)
        
        # Split content by section headings
        section_chunks = self._split_by_sections(content, sections)
        
        # Build chunks, splitting large sections if needed
        chunks = []
        chunk_index = 0
        
        for section_heading, section_content in section_chunks:
            # If section is small enough, keep it as one chunk
            word_count = len(section_content.split())
            
            if word_count <= self.max_chunk_size:
                chunk = self._create_chunk_object(
                    section_content,
                    doc_id,
                    chunk_index,
                    document,
                    section_heading
                )
                chunks.append(chunk)
                chunk_index += 1
            else:
                # Section too large, split it further
                sub_chunks = self._split_large_section(
                    section_content, doc_id, chunk_index, document, section_heading
                )
                chunks.extend(sub_chunks)
                chunk_index += len(sub_chunks)
        
        return chunks
    
    def _split_by_sections(
        self, content: str, sections: List[str]
    ) -> List[Tuple[Optional[str], str]]:
        """
        Split content by section headings, scanning left to right.

        Each heading is located strictly AFTER the position where the previous
        heading matched. Technical docs repeat heading strings constantly
        ("Examples", "Notes", "Parameters"), so matching every heading at its
        first occurrence — as an earlier implementation did — could place a
        section's end before its start, yielding empty chunks. Headings that
        cannot be found in the flattened text are skipped, and any text before
        the first matched heading is kept as a heading-less preamble section
        rather than silently dropped.

        Args:
            content: Full document content
            sections: List of section headings, in document order

        Returns:
            List of (section_heading, section_content) tuples; the heading is
            None for the preamble.
        """
        # Locate each heading sequentially so boundaries are monotone.
        positions: List[Tuple[str, int]] = []
        search_from = 0
        for section in sections:
            idx = content.find(section, search_from)
            if idx == -1:
                continue
            positions.append((section, idx))
            search_from = idx + len(section)

        if not positions:
            # Headings exist in metadata but none survive in the flattened
            # text: treat the whole document as one heading-less section.
            return [(None, content)] if content.strip() else []

        result: List[Tuple[Optional[str], str]] = []
        preamble = content[: positions[0][1]].strip()
        if preamble:
            result.append((None, preamble))

        for i, (section, start_pos) in enumerate(positions):
            end_pos = positions[i + 1][1] if i + 1 < len(positions) else len(content)
            section_content = content[start_pos:end_pos].strip()
            if section_content:
                result.append((section, section_content))

        return result
    
    def _split_large_section(
        self,
        section_content: str,
        doc_id: str,
        start_index: int,
        document: Dict[str, Any],
        section_heading: str
    ) -> List[Chunk]:
        """
        Split a large section into smaller chunks.
        
        Args:
            section_content: Content of the section
            doc_id: Document ID
            start_index: Starting chunk index
            document: Original document
            section_heading: The section heading
            
        Returns:
            List of chunks
        """
        # Split by paragraphs first
        paragraphs = re.split(r'\n\s*\n', section_content)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        chunks = []
        current_paragraphs = []
        current_word_count = 0
        chunk_index = start_index
        
        for para in paragraphs:
            para_words = len(para.split())
            
            # If adding this would exceed max, finalize current chunk
            if current_word_count + para_words > self.max_chunk_size and current_paragraphs:
                chunk_text = "\n\n".join(current_paragraphs)
                chunk = self._create_chunk_object(
                    chunk_text, doc_id, chunk_index, document, section_heading
                )
                chunks.append(chunk)
                
                current_paragraphs = []
                current_word_count = 0
                chunk_index += 1
            
            current_paragraphs.append(para)
            current_word_count += para_words
        
        # Add final chunk
        if current_paragraphs:
            chunk_text = "\n\n".join(current_paragraphs)
            chunk = self._create_chunk_object(
                chunk_text, doc_id, chunk_index, document, section_heading
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_by_paragraphs(
        self, content: str, doc_id: str, document: Dict[str, Any]
    ) -> List[Chunk]:
        """
        Fallback to paragraph-based chunking when no sections available.
        
        Args:
            content: Document content
            doc_id: Document ID
            document: Original document
            
        Returns:
            List of chunks
        """
        paragraphs = re.split(r'\n\s*\n', content)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        chunks = []
        current_paragraphs = []
        current_word_count = 0
        chunk_index = 0
        
        for para in paragraphs:
            para_words = len(para.split())
            
            if current_word_count + para_words > self.max_chunk_size and current_paragraphs:
                chunk_text = "\n\n".join(current_paragraphs)
                chunk = self._create_chunk_object(
                    chunk_text, doc_id, chunk_index, document, None
                )
                chunks.append(chunk)
                
                current_paragraphs = []
                current_word_count = 0
                chunk_index += 1
            
            current_paragraphs.append(para)
            current_word_count += para_words
        
        if current_paragraphs:
            chunk_text = "\n\n".join(current_paragraphs)
            chunk = self._create_chunk_object(
                chunk_text, doc_id, chunk_index, document, None
            )
            chunks.append(chunk)
        
        return chunks
    
    def _create_chunk_object(
        self,
        chunk_text: str,
        doc_id: str,
        chunk_index: int,
        document: Dict[str, Any],
        section_heading: str = None
    ) -> Chunk:
        """Create a Chunk object with metadata."""
        base_metadata = self._extract_metadata(document)
        base_metadata.update({
            "max_chunk_size": self.max_chunk_size,
            "word_count": len(chunk_text.split()),
            "char_count": len(chunk_text),
            "section_heading": section_heading,
        })
        
        return Chunk(
            content=chunk_text,
            chunk_id=self._create_chunk_id(doc_id, chunk_index),
            doc_id=doc_id,
            chunk_index=chunk_index,
            metadata=base_metadata,
        )

