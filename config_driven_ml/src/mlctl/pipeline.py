"""Domain logic: train and evaluate a regressor on the diabetes dataset.

Deliberately small — the interesting part of this project is the config layer.
No Hydra imports here: these functions take validated Pydantic configs, which
keeps them trivially unit-testable.
"""

import json
import logging
from pathlib import Path

import joblib
from omegaconf import OmegaConf
from pydantic import BaseModel
from sklearn.datasets import load_diabetes
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

from mlctl.config_models import EvaluateConfig, GBMConfig, RidgeConfig, TrainConfig

logger = logging.getLogger(__name__)

SNAPSHOT_FILENAME = "config_snapshot.yaml"


def load_split(seed: int, test_size: float):
    X, y = load_diabetes(return_X_y=True)
    return train_test_split(X, y, test_size=test_size, random_state=seed)


def build_model(cfg: RidgeConfig | GBMConfig, seed: int):
    if isinstance(cfg, RidgeConfig):
        return Ridge(alpha=cfg.alpha, random_state=seed)
    return HistGradientBoostingRegressor(
        learning_rate=cfg.learning_rate,
        max_iter=cfg.max_iter,
        max_depth=cfg.max_depth,
        random_state=seed,
    )


def compute_metrics(model, X, y) -> dict[str, float]:
    pred = model.predict(X)
    return {
        "rmse": round(root_mean_squared_error(y, pred), 3),
        "mae": round(mean_absolute_error(y, pred), 3),
        "r2": round(r2_score(y, pred), 4),
    }


def save_config_snapshot(cfg: BaseModel, path: Path) -> None:
    """Persist the fully resolved config next to the run's artifacts, so any
    run can be reproduced (or re-evaluated) from its output directory alone."""
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(cfg.model_dump(mode="json")), path)


def run_training(cfg: TrainConfig) -> dict[str, float]:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = load_split(cfg.seed, cfg.test_size)
    logger.info(
        "Training %s on %d samples (experiment=%s, seed=%d)",
        cfg.model.kind, len(X_train), cfg.experiment_name, cfg.seed,
    )

    model = build_model(cfg.model, cfg.seed)
    model.fit(X_train, y_train)

    metrics = compute_metrics(model, X_test, y_test)
    logger.info("Test metrics: %s", metrics)

    joblib.dump(model, output_dir / "model.joblib")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    save_config_snapshot(cfg, output_dir / SNAPSHOT_FILENAME)
    logger.info("Artifacts written to %s", output_dir)

    return metrics


def run_evaluation(cfg: EvaluateConfig) -> dict[str, float]:
    run_dir = Path(cfg.run_dir)
    snapshot_path = run_dir / SNAPSHOT_FILENAME
    if not snapshot_path.exists():
        raise SystemExit(f"No {SNAPSHOT_FILENAME} found in {run_dir} — is this a training run directory?")

    train_cfg = TrainConfig.model_validate(
        OmegaConf.to_container(OmegaConf.load(snapshot_path))
    )
    logger.info(
        "Re-evaluating run %s (%s, seed=%d) on its held-out test split",
        run_dir, train_cfg.model.kind, train_cfg.seed,
    )

    # The snapshot pins seed and test_size, so this reconstructs the exact
    # same held-out split the run was scored on at training time.
    _, X_test, _, y_test = load_split(train_cfg.seed, train_cfg.test_size)
    model = joblib.load(run_dir / "model.joblib")

    metrics = compute_metrics(model, X_test, y_test)
    logger.info("Test metrics: %s", metrics)
    (run_dir / "eval_metrics.json").write_text(json.dumps(metrics, indent=2))

    return metrics
