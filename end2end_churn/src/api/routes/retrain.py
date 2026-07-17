"""Retraining endpoint: manual trigger with an atomic rate-limit reservation."""

from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from ...utils.logger import get_logger
from .. import retraining
from ..auth import verify_token

logger = get_logger("churn_api")

router = APIRouter()


@router.post("/retrain", tags=["Model Management"])
async def trigger_retraining(
    background_tasks: BackgroundTasks, _token: Optional[str] = Depends(verify_token)
) -> dict[str, Any]:
    """
    Trigger model retraining in the background.

    Safeguards:
    - Rate limited to once per MIN_RETRAIN_INTERVAL_HOURS (default 24h)
    - Runs in background (doesn't block API)
    - New model saved but NOT auto-deployed
    - Requires manual review and deployment

    Returns:
        Retraining status and next steps

    Raises:
        429: Retraining triggered too recently (rate limit)
    """
    # Atomically check the rate limit and reserve the slot before scheduling, so
    # concurrent /retrain calls cannot both pass the check and launch duplicate runs.
    if not retraining.check_and_reserve_retraining():
        last_retrain = retraining.get_last_retrain_time()
        if last_retrain is not None:
            time_since_last = datetime.now() - last_retrain
            hours_remaining = retraining.MIN_RETRAIN_INTERVAL_HOURS - (
                time_since_last.total_seconds() / 3600
            )
            detail = (
                f"Retraining triggered too recently. Minimum "
                f"{retraining.MIN_RETRAIN_INTERVAL_HOURS}h between retrains. "
                f"Try again in {max(hours_remaining, 0.0):.1f} hours."
            )
        else:
            # The reservation failed without a readable prior timestamp — an
            # I/O problem with the state file, not an actual rate limit.
            # Don't tell the caller to wait 24 hours for a limit that isn't set.
            detail = (
                "Could not reserve the retraining slot (retrain state file "
                "unreadable or locked). Check server logs; this is not a rate limit."
            )
        raise HTTPException(status_code=429, detail=detail)

    # Trigger background retraining (slot already reserved above)
    background_tasks.add_task(retraining.retrain_model_task)

    logger.warning("Model retraining triggered via API")

    return {
        "status": "retraining_triggered",
        "message": "Model retraining started in background. Check logs for progress.",
        "note": "New model will be saved but NOT auto-deployed. Review metrics before deploying.",
        "next_steps": [
            "1. Monitor logs: docker compose logs -f api",
            "2. Review metrics: cat diagnostics/evaluation_report_*.txt",
            "3. Deploy if satisfied: make restart",
        ],
        "timestamp": datetime.now().isoformat(),
    }
