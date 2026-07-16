"""Generation components: LLM integration and answer generation.

Submodules are imported lazily (PEP 562) so importing this package does not
pull in the HTTP client stack until a class is actually used.
"""

from importlib import import_module

_LAZY_IMPORTS = {
    "OllamaClient": ".llm_client",
    "PromptBuilder": ".prompt_builder",
    "AnswerGenerator": ".answer_generator",
}

__all__ = list(_LAZY_IMPORTS)


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        return getattr(import_module(_LAZY_IMPORTS[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
