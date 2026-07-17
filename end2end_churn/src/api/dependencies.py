"""FastAPI dependencies shared across route modules."""

from typing import cast

from fastapi import Request

from .types import ModelCache


def get_model_cache(request: Request) -> ModelCache:
    """
    Provide the model cache from app.state.

    The cache lives on app.state (not module state) so every worker process
    gets its own copy in multi-worker deployments. Endpoints that need the
    model take this as a dependency, which also makes it trivial to swap in
    a fake cache in tests.
    """
    return cast(ModelCache, request.app.state.model_cache)
