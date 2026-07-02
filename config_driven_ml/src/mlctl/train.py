"""Train command - thin CLI entry over pipeline.run_training."""

from mlctl.config_layer import config_command
from mlctl.config_models import TrainConfig
from mlctl.pipeline import run_training


@config_command(TrainConfig, config_path="configs", config_name="train")
def main(cfg: TrainConfig) -> None:
    run_training(cfg)
