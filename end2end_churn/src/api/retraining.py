"""
Retraining coordination for the churn prediction service.

Retraining can be triggered via the /retrain endpoint or automatically on
drift. Both paths go through a single file-lock-based gate that makes the
rate-limit check and the slot reservation atomic across workers, then run
the training script in a background thread. A newly trained model is saved
but never auto-deployed.
"""

import asyncio
import fcntl
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from ..utils.logger import get_logger

logger = get_logger("churn_api")

# Automatic retraining configuration
AUTO_RETRAIN_ON_DRIFT = os.getenv("AUTO_RETRAIN_ON_DRIFT", "false").lower() == "true"
MIN_RETRAIN_INTERVAL_HOURS = int(os.getenv("MIN_RETRAIN_INTERVAL_HOURS", "24"))

# File path for storing last retrain timestamp (multi-worker safe)
LAST_RETRAIN_FILE = Path("models/.last_retrain")


def get_last_retrain_time() -> Optional[datetime]:
    """
    Read last retrain timestamp from file (multi-worker safe).

    Returns:
        datetime of last retrain, or None if never retrained
    """
    if not LAST_RETRAIN_FILE.exists():
        return None

    try:
        with open(LAST_RETRAIN_FILE, "r") as f:
            # Acquire shared lock for reading
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            try:
                timestamp_str = f.read().strip()
                if timestamp_str:
                    return datetime.fromisoformat(timestamp_str)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except Exception as e:
        logger.warning(f"Could not read last retrain time: {e}")

    return None


def check_and_reserve_retraining() -> bool:
    """
    Atomically check the retrain rate limit and reserve the retraining slot.

    Under a single exclusive lock on the timestamp file, read the last-retrain
    time and, if the minimum interval has elapsed (or no prior run exists),
    write the current time as the reservation BEFORE returning True. This closes
    the check-then-act race in the previous design (separate should_allow check
    and post-training timestamp write), where two concurrent triggers could both
    pass the check and launch duplicate 10-minute trainings.

    The reservation is written up front (not after training completes) and is
    intentionally NOT rolled back on failure: a failed run still consumes the
    interval, which prevents a crash-looping trainer from retraining
    continuously. Operators can force an earlier retry by deleting
    ``models/.last_retrain``.

    Returns:
        True if this caller reserved the slot and should proceed to retrain;
        False if rate-limited by an existing reservation (or on I/O error, in
        which case we fail closed and do not retrain).
    """
    try:
        LAST_RETRAIN_FILE.parent.mkdir(parents=True, exist_ok=True)
        # Open read+write without truncating (create if missing) so the read and
        # the reservation write happen under the same lock.
        with open(LAST_RETRAIN_FILE, "a+") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.seek(0)
                timestamp_str = f.read().strip()
                last_retrain = None
                if timestamp_str:
                    try:
                        last_retrain = datetime.fromisoformat(timestamp_str)
                    except ValueError:
                        logger.warning("Invalid last-retrain timestamp; treating as unset")

                now = datetime.now()
                if last_retrain is not None and (now - last_retrain) <= timedelta(
                    hours=MIN_RETRAIN_INTERVAL_HOURS
                ):
                    return False  # still within the interval -> rate limited

                # Reserve the slot: overwrite the file with the current time.
                f.seek(0)
                f.truncate()
                f.write(now.isoformat())
                f.flush()
                return True
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except Exception as e:
        logger.error(f"Could not check/reserve retraining: {e}")
        return False


async def retrain_model_task() -> None:
    """
    Background task to retrain the model.

    The rate-limit slot must already be reserved via check_and_reserve_retraining()
    before this task is scheduled. Training runs in a worker thread via
    asyncio.to_thread so the event loop stays responsive during the (up to
    10-minute) run. The reservation is intentionally not rolled back on failure,
    so a failed run still consumes the interval (crash-loop protection).
    """
    try:
        logger.info("=" * 60)
        logger.info("MODEL RETRAINING STARTED")
        logger.info("=" * 60)

        # Run training script in a worker thread so the blocking subprocess does
        # not stall the async event loop (MLflow tracking always enabled).
        result = await asyncio.to_thread(
            subprocess.run,
            [sys.executable, "train.py"],
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
            cwd=os.getcwd(),
        )

        if result.returncode == 0:
            logger.info("Model retraining completed successfully")
            logger.info("New model saved but NOT deployed")
            logger.info("→ Review metrics in diagnostics/ directory")
            logger.info("→ Run 'make restart' to deploy new model")
        else:
            logger.error(f"Model retraining failed (exit {result.returncode}): {result.stderr}")

    except subprocess.TimeoutExpired:
        logger.error("Model retraining timed out after 10 minutes")
    except Exception as e:
        logger.error(f"Model retraining error: {e}", exc_info=True)
