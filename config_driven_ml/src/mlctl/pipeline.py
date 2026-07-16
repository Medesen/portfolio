"""Domain logic: train and evaluate a regressor on the diabetes dataset.

Deliberately small — the interesting part of this project is the config layer.
No Hydra composition happens here: these functions take validated Pydantic
configs, which keeps them trivially unit-testable. (OmegaConf — Hydra's config
engine, declared as a direct dependency — is used only to serialize/load the
per-run config snapshot in YAML.)
"""

import hashlib
import json
import logging
from pathlib import Path

import joblib
import numpy as np
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
SPLIT_FILENAME = "split.json"
MODEL_FILENAME = "model.joblib"
MODEL_CHECKSUM_FILENAME = MODEL_FILENAME + ".sha256"


def _dataset_checksum(X: np.ndarray, y: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(X).tobytes())
    h.update(np.ascontiguousarray(y).tobytes())
    return h.hexdigest()


def load_split(seed: int, test_size: float):
    """Split by row index (same permutation as splitting the arrays directly)
    so the test-row identities can be persisted alongside the run."""
    X, y = load_diabetes(return_X_y=True)
    idx = np.arange(len(X))
    train_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=seed)
    return (
        X[train_idx], X[test_idx], y[train_idx], y[test_idx],
        test_idx, _dataset_checksum(X, y),
    )


def build_model(cfg: RidgeConfig | GBMConfig, seed: int):
    if isinstance(cfg, RidgeConfig):
        # No random_state: Ridge only uses it with solver="sag"/"saga", and the
        # default auto solver is deterministic — passing the seed would be noise.
        return Ridge(alpha=cfg.alpha)
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

    X_train, X_test, y_train, y_test, test_idx, dataset_sha = load_split(
        cfg.seed, cfg.test_size
    )
    logger.info(
        "Training %s on %d samples (experiment=%s, seed=%d)",
        cfg.model.kind, len(X_train), cfg.experiment_name, cfg.seed,
    )

    model = build_model(cfg.model, cfg.seed)
    model.fit(X_train, y_train)

    metrics = compute_metrics(model, X_test, y_test)
    logger.info("Test metrics: %s", metrics)

    model_path = output_dir / MODEL_FILENAME
    joblib.dump(model, model_path)
    # Integrity sidecar: proves the artifact is byte-identical to what training
    # wrote (it does NOT make an artifact from an untrusted source safe to load).
    (output_dir / MODEL_CHECKSUM_FILENAME).write_text(
        hashlib.sha256(model_path.read_bytes()).hexdigest()
    )
    # Persist the evaluation split itself — row indices plus a dataset
    # checksum — so re-evaluation does not depend on the dataset and splitter
    # implementation staying identical forever.
    (output_dir / SPLIT_FILENAME).write_text(
        json.dumps({"dataset_sha256": dataset_sha, "test_indices": test_idx.tolist()})
    )
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

    # Preferred path: the run persisted its exact test rows (indices + dataset
    # checksum), so the split is recovered rather than reconstructed.
    split_path = run_dir / SPLIT_FILENAME
    if split_path.exists():
        record = json.loads(split_path.read_text())
        X, y = load_diabetes(return_X_y=True)
        current_sha = _dataset_checksum(X, y)
        if record["dataset_sha256"] != current_sha:
            raise SystemExit(
                "Dataset checksum differs from the one recorded at training time; "
                "the persisted test indices no longer identify the same rows. "
                "Re-train, or evaluate in the original environment."
            )
        test_idx = np.asarray(record["test_indices"], dtype=int)
        X_test, y_test = X[test_idx], y[test_idx]
    else:
        # Older runs predate split persistence: fall back to reconstructing
        # from the pinned seed/test_size, which is only exact while the
        # dataset and splitter implementation are unchanged.
        logger.warning(
            "No %s in %s (pre-split-persistence run); reconstructing the "
            "held-out split from the snapshot's seed and test_size.",
            SPLIT_FILENAME, run_dir,
        )
        _, X_test, _, y_test, _, _ = load_split(train_cfg.seed, train_cfg.test_size)

    model_path = run_dir / MODEL_FILENAME
    checksum_path = run_dir / MODEL_CHECKSUM_FILENAME
    if checksum_path.exists():
        expected = checksum_path.read_text().strip()
        actual = hashlib.sha256(model_path.read_bytes()).hexdigest()
        if actual != expected:
            raise SystemExit(
                f"{MODEL_FILENAME} does not match its recorded SHA-256; "
                "refusing to deserialize a modified artifact."
            )
    else:
        logger.warning(
            "No %s alongside the artifact (pre-checksum run); loading without "
            "integrity verification.", MODEL_CHECKSUM_FILENAME,
        )
    model = joblib.load(model_path)

    metrics = compute_metrics(model, X_test, y_test)
    logger.info("Test metrics: %s", metrics)
    (run_dir / "eval_metrics.json").write_text(json.dumps(metrics, indent=2))

    return metrics
