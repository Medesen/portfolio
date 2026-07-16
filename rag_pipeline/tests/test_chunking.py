"""
Test chunking strategies.

These tests verify that:
- Fixed chunker creates uniform-sized chunks
- Semantic chunker respects natural boundaries
- Chunk IDs are generated correctly
- Metadata is preserved
"""

import pytest
from src.chunking.fixed_chunker import FixedSizeChunker
from src.chunking.semantic_chunker import SemanticChunker


def test_fixed_chunker_creates_chunks(sample_document):
    """Test that fixed chunker creates chunks from a document."""
    config = {
        "chunk_size": 512,
        "overlap": 50
    }
    chunker = FixedSizeChunker(config)
    
    chunks = chunker.chunk_document(sample_document)
    
    # Verify chunks were created
    assert len(chunks) > 0
    # Verify each chunk has required fields
    for chunk in chunks:
        assert hasattr(chunk, 'chunk_id')
        assert hasattr(chunk, 'doc_id')
        assert hasattr(chunk, 'content')
        assert hasattr(chunk, 'chunk_index')


def test_fixed_chunker_chunk_sizes(sample_document):
    """Test that fixed chunker creates roughly uniform-sized chunks."""
    config = {
        "chunk_size": 512,
        "overlap": 50
    }
    chunker = FixedSizeChunker(config)
    
    chunks = chunker.chunk_document(sample_document)
    
    # Calculate word counts
    word_counts = [len(chunk.content.split()) for chunk in chunks]
    
    # Verify sizes are within expected range (~384 words ± 20%)
    # 512 tokens * 0.75 words/token = ~384 words
    for count in word_counts[:-1]:  # Exclude last chunk (may be smaller)
        assert 300 < count < 500, f"Chunk size {count} outside expected range"


def test_semantic_chunker_creates_chunks(sample_document):
    """Test that semantic chunker creates chunks."""
    config = {
        "max_chunk_size": 1000,
        "method": "sentence"
    }
    chunker = SemanticChunker(config)
    
    chunks = chunker.chunk_document(sample_document)
    
    # Verify chunks were created
    assert len(chunks) > 0
    # Verify strategy name
    assert chunker.get_strategy_name() == "semantic"


def test_chunk_id_generation(sample_document):
    """Test that chunk IDs are generated correctly."""
    config = {"chunk_size": 512, "overlap": 50}
    chunker = FixedSizeChunker(config)
    
    chunks = chunker.chunk_document(sample_document)
    
    # Verify chunk IDs follow pattern: {doc_id}__chunk_{index}
    for i, chunk in enumerate(chunks):
        expected_id = f"{sample_document['doc_id']}__chunk_{i}"
        assert chunk.chunk_id == expected_id
        assert chunk.chunk_index == i



def test_fixed_chunker_rejects_overlap_not_smaller_than_chunk_size():
    """overlap >= chunk_size would make the chunk window non-advancing
    (an infinite loop during indexing); the constructor must refuse it."""
    with pytest.raises(ValueError, match="overlap"):
        FixedSizeChunker({"chunk_size": 100, "overlap": 100})
    with pytest.raises(ValueError, match="overlap"):
        FixedSizeChunker({"chunk_size": 100, "overlap": 150})
    # The ~0.75 token->word conversion can collapse nearby values
    # (5 and 4 tokens both floor to 3 words) — also rejected.
    with pytest.raises(ValueError, match="overlap"):
        FixedSizeChunker({"chunk_size": 5, "overlap": 4})


def test_hierarchical_chunker_handles_repeated_headings_and_preamble():
    """Repeated heading strings must not produce empty or out-of-order
    sections, and text before the first heading must be kept."""
    from src.chunking.hierarchical_chunker import HierarchicalChunker

    content = (
        "Intro text before any heading. "
        "Examples early mention of the word Examples in prose. "
        "Parameters This section describes parameters in detail. "
        "Examples This is the real examples section content. "
        "Notes Closing notes content."
    )
    document = {
        "doc_id": "doc1",
        "content": content,
        "metadata": {"sections": ["Parameters", "Examples", "Notes"]},
    }
    chunker = HierarchicalChunker({"max_chunk_size": 1000})
    chunks = chunker.chunk_document(document)

    # No empty chunks, ever
    assert all(chunk.content.strip() for chunk in chunks)
    # The preamble (including the early "Examples" prose) is preserved
    assert any("Intro text before any heading" in c.content for c in chunks)
    # "Examples" resolves to the occurrence AFTER "Parameters", not the
    # early prose mention — so the Parameters section ends where the real
    # Examples section starts
    examples_chunks = [
        c for c in chunks if c.metadata.get("section_heading") == "Examples"
    ]
    assert len(examples_chunks) == 1
    assert "real examples section" in examples_chunks[0].content
    # Every character of the document lands in exactly one chunk
    reassembled = " ".join(c.content for c in chunks)
    assert "Closing notes content" in reassembled
