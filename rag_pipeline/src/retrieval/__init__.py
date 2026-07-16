"""Retrieval components: embeddings, vector storage, and hybrid search.

Submodules are imported lazily (PEP 562): importing this package does not pull
in sentence-transformers, ChromaDB, or the cross-encoder until the class that
needs them is actually requested. This keeps ``import src.retrieval`` cheap and
lets modules with lighter dependencies be used (and tested) in environments
where the heavy optional dependencies are absent.
"""

from importlib import import_module

_LAZY_IMPORTS = {
    "Embedder": ".embedder",
    "VectorStore": ".vector_store",
    "Indexer": ".indexer",
    "QueryProcessor": ".query_processor",
    "BM25Index": ".bm25_index",
    "HybridSearcher": ".hybrid_searcher",
    "QueryRewriter": ".query_rewriter",
    "CrossEncoderReranker": ".reranker",
}

__all__ = list(_LAZY_IMPORTS)


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        return getattr(import_module(_LAZY_IMPORTS[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
