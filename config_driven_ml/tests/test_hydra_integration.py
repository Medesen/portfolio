"""Integration tests: Hydra composition -> Pydantic validation boundary."""

import pytest
from hydra import compose, initialize

from mlctl.config_layer import validate_config
from mlctl.config_models import EvaluateConfig, RidgeConfig, TrainConfig


@pytest.fixture
def hydra_ctx():
    with initialize(config_path="../src/mlctl/configs", version_base=None):
        yield


def test_default_composition_validates(hydra_ctx):
    cfg = compose(config_name="train")
    validated = validate_config(TrainConfig, cfg)
    assert validated.model.kind == "gbm"
    assert validated.output_dir == "outputs/baseline/gbm/seed_42"


def test_model_group_override(hydra_ctx):
    cfg = compose(config_name="train", overrides=["model=ridge", "model.alpha=3.0"])
    validated = validate_config(TrainConfig, cfg)
    assert isinstance(validated.model, RidgeConfig)
    assert validated.model.alpha == 3.0


def test_named_experiment_config(hydra_ctx):
    cfg = compose(config_name="ridge_strong")
    validated = validate_config(TrainConfig, cfg)
    assert isinstance(validated.model, RidgeConfig)
    assert validated.model.alpha == 10.0
    assert validated.output_dir == "outputs/ridge_strong/ridge/seed_42"


def test_invalid_override_exits_cleanly(hydra_ctx):
    cfg = compose(config_name="train", overrides=["model.max_iter=-5"])
    with pytest.raises(SystemExit, match="Invalid configuration"):
        validate_config(TrainConfig, cfg)


def test_missing_required_value_exits_cleanly(hydra_ctx):
    cfg = compose(config_name="evaluate")
    with pytest.raises(SystemExit, match="Required config value was never set"):
        validate_config(EvaluateConfig, cfg)
