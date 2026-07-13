"""Domain logic: train and evaluate a regressor on the diabetes dataset.

Deliberately small — the interesting part of this project is the config layer.
No Hydra imports here: these functions take validated Pydantic configs, which
keeps them trivially unit-testable.
"""

import json
import logging
from pathlib import Path

import joblib
import sklearn
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
METADATA_FILENAME = "run_metadata.json"


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
    """Persist the fully resolved config next to the run's artifacts, so a run
    can be re-evaluated from its output directory *within the same environment*
    (dependencies are pinned via requirements.lock). The snapshot pins the config
    only; the dataset and model are reloaded from the installed packages, so a
    run_metadata.json fingerprint (see run_training) lets re-eval flag drift."""
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(cfg.model_dump(mode="json")), path)


def _warn_on_environment_drift(run_dir: Path) -> None:
    """Warn (not fail) when the current environment differs from the one recorded
    at training time. Re-evaluation reloads the dataset and the pickled model from
    the installed packages rather than from the run directory, so a changed
    scikit-learn or dataset shape can move the metrics or break unpickling."""
    metadata_path = run_dir / METADATA_FILENAME
    if not metadata_path.exists():
        logger.warning(
            "No %s in %s; cannot verify the training environment matches this one.",
            METADATA_FILENAME, run_dir,
        )
        return

    recorded = json.loads(metadata_path.read_text())

    current_sklearn = sklearn.__version__
    if recorded.get("sklearn_version") not in (None, current_sklearn):
        logger.warning(
            "scikit-learn changed since training (%s -> %s); re-evaluated metrics "
            "may differ and the pickled model may not load cleanly.",
            recorded["sklearn_version"], current_sklearn,
        )

    X, _ = load_diabetes(return_X_y=True)
    current_shape = [int(X.shape[0]), int(X.shape[1])]
    recorded_shape = [recorded.get("n_samples"), recorded.get("n_features")]
    if None not in recorded_shape and recorded_shape != current_shape:
        logger.warning(
            "diabetes dataset shape changed since training (%s -> %s); the "
            "reconstructed held-out split may not match the original.",
            recorded_shape, current_shape,
        )


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

    # Environment fingerprint (kept out of the config snapshot, which is
    # re-validated as a strict TrainConfig) so a later re-eval can detect drift.
    metadata = {
        "sklearn_version": sklearn.__version__,
        "dataset": "sklearn.datasets.load_diabetes",
        "n_samples": len(X_train) + len(X_test),
        "n_features": int(X_train.shape[1]),
    }
    (output_dir / METADATA_FILENAME).write_text(json.dumps(metadata, indent=2))
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

    # Warn if the environment (scikit-learn / dataset shape) drifted since
    # training — the run directory pins the config, not the packages.
    _warn_on_environment_drift(run_dir)

    # The snapshot pins seed and test_size, so this reconstructs the exact
    # same held-out split the run was scored on at training time.
    _, X_test, _, y_test = load_split(train_cfg.seed, train_cfg.test_size)
    model = joblib.load(run_dir / "model.joblib")

    metrics = compute_metrics(model, X_test, y_test)
    logger.info("Test metrics: %s", metrics)
    (run_dir / "eval_metrics.json").write_text(json.dumps(metrics, indent=2))

    return metrics
