"""End-to-end test: train -> artifacts -> evaluate reproduces metrics."""

import json

from mlctl.config_models import EvaluateConfig, GBMConfig, TrainConfig
from mlctl.pipeline import run_evaluation, run_training


def test_train_then_evaluate_roundtrip(tmp_path):
    train_cfg = TrainConfig(
        experiment_name="test",
        seed=7,
        model=GBMConfig(max_iter=20),
        output_dir=str(tmp_path / "run"),
    )
    train_metrics = run_training(train_cfg)

    run_dir = tmp_path / "run"
    assert (run_dir / "model.joblib").exists()
    assert (run_dir / "config_snapshot.yaml").exists()
    assert json.loads((run_dir / "metrics.json").read_text()) == train_metrics
    assert train_metrics["rmse"] > 0

    # Evaluation reconstructs the split from the snapshot alone, so its
    # metrics must match what training reported.
    eval_metrics = run_evaluation(EvaluateConfig(run_dir=str(run_dir)))
    assert eval_metrics == train_metrics


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
