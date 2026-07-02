"""Evaluate command - thin CLI entry over pipeline.run_evaluation."""

from mlctl.config_layer import config_command
from mlctl.config_models import EvaluateConfig
from mlctl.pipeline import run_evaluation


@config_command(EvaluateConfig, config_path="configs", config_name="evaluate")
def main(cfg: EvaluateConfig) -> None:
    run_evaluation(cfg)
