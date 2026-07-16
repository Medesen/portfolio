"""Evaluation framework for RAG system assessment.

Submodules are imported lazily (PEP 562): the metrics calculator, for example,
can be imported and tested without pulling in the evaluator's retrieval and
generation dependencies.
"""

from importlib import import_module

_LAZY_IMPORTS = {
    "RAGEvaluator": ".evaluator",
    "TestLoader": ".test_loader",
    "TestQuestion": ".test_loader",
    "RetrievalMetrics": ".metrics",
    "extract_doc_ids_from_chunks": ".metrics",
    "LLMJudge": ".llm_judge",
    "ResultsAnalyzer": ".results_analyzer",
}

__all__ = list(_LAZY_IMPORTS)


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        return getattr(import_module(_LAZY_IMPORTS[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
