"""Minimal FastAPI app exposing the two-stage recommender with per-stage timing.

Deliberately minimal — this is a *measurement instrument*, not a product. No metrics
stack, no feature store, no auth: those are demonstrated in the sibling end2end_churn
project, and repeating them here would add volume, not evidence. What production would
change is documented in the README, not built.

The service is loaded from ``RECLAB_SERVICE_DIR`` (default ``outputs/serving``), which
``reclab build-service`` populates.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from reclab.serving.pipeline import RecommenderService

SERVICE_DIR = Path(os.environ.get("RECLAB_SERVICE_DIR", "outputs/serving"))

app = FastAPI(title="reclab two-stage recommender", version="1.0")
_service: RecommenderService | None = None


class RecommendRequest(BaseModel):
    history: list[int] = Field(..., description="Recent item ids for this visitor")
    k: int = Field(10, ge=1, le=200)
    ef_search: int = Field(64, ge=1, le=1000)


class RecommendResponse(BaseModel):
    items: list[int]
    scores: list[float]
    timings_ms: dict[str, float]
    n_candidates: int
    strategy: str
    requested_k: int


def get_service() -> RecommenderService:
    global _service
    if _service is None:
        if not (SERVICE_DIR / "config.json").exists():
            raise HTTPException(503, f"service artifacts not found in {SERVICE_DIR}; "
                                     "run `reclab build-service` first")
        _service = RecommenderService.load(SERVICE_DIR)
    return _service


@app.get("/health")
def health() -> dict:
    try:
        svc = get_service()
    except HTTPException:
        return {"status": "no_artifacts", "service_dir": str(SERVICE_DIR)}
    return {"status": "ok", "n_items": int(len(svc.item_ids)),
            "n_candidates": svc.n_candidates, "factors": svc.factors}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest) -> RecommendResponse:
    svc = get_service()
    rec = svc.recommend(req.history, k=req.k, ef_search=req.ef_search)
    return RecommendResponse(items=rec.items, scores=rec.scores,
                             timings_ms=rec.timings_ms, n_candidates=rec.n_candidates,
                             strategy=rec.strategy, requested_k=rec.requested_k)
