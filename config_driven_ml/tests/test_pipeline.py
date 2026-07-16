"""End-to-end test: train -> artifacts -> evaluate reproduces metrics."""

import json

import pytest

from mlctl.config_models import EvaluateConfig, GBMConfig, TrainConfig
from mlctl.pipeline import run_evaluation, run_training


def _train(tmp_path, **overrides):
    train_cfg = TrainConfig(
        experiment_name="test",
        seed=7,
        model=GBMConfig(max_iter=20),
        output_dir=str(tmp_path / "run"),
        **overrides,
    )
    return run_training(train_cfg), tmp_path / "run"


def test_train_then_evaluate_roundtrip(tmp_path):
    train_metrics, run_dir = _train(tmp_path)

    assert (run_dir / "model.joblib").exists()
    assert (run_dir / "model.joblib.sha256").exists()
    assert (run_dir / "config_snapshot.yaml").exists()
    assert json.loads((run_dir / "metrics.json").read_text()) == train_metrics
    assert train_metrics["rmse"] > 0

    # Evaluation recovers the persisted test rows (split.json), so its
    # metrics must match what training reported exactly.
    eval_metrics = run_evaluation(EvaluateConfig(run_dir=str(run_dir)))
    assert eval_metrics == train_metrics


def test_split_persisted_with_dataset_checksum(tmp_path):
    _, run_dir = _train(tmp_path)
    record = json.loads((run_dir / "split.json").read_text())
    assert len(record["dataset_sha256"]) == 64
    # default test_size=0.2 of the 442-row diabetes dataset
    assert len(record["test_indices"]) == 89
    assert len(set(record["test_indices"])) == 89


def test_evaluation_rejects_tampered_model(tmp_path):
    _, run_dir = _train(tmp_path)
    (run_dir / "model.joblib").write_bytes(b"not the trained model")
    with pytest.raises(SystemExit, match="SHA-256"):
        run_evaluation(EvaluateConfig(run_dir=str(run_dir)))


def test_evaluation_rejects_foreign_split_record(tmp_path):
    _, run_dir = _train(tmp_path)
    record = json.loads((run_dir / "split.json").read_text())
    record["dataset_sha256"] = "0" * 64
    (run_dir / "split.json").write_text(json.dumps(record))
    with pytest.raises(SystemExit, match="checksum differs"):
        run_evaluation(EvaluateConfig(run_dir=str(run_dir)))


def test_run_metadata_fingerprint_written(tmp_path):
    """Training writes an environment fingerprint (scikit-learn version, dataset
    shape) so a later re-eval can detect drift."""
    train_cfg = TrainConfig(
        experiment_name="test",
        seed=7,
        model=GBMConfig(max_iter=20),
        output_dir=str(tmp_path / "run"),
    )
    run_training(train_cfg)

    meta_path = tmp_path / "run" / "run_metadata.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert meta["sklearn_version"]  # non-empty
    assert meta["n_samples"] == 442  # diabetes dataset
    assert meta["n_features"] == 10
