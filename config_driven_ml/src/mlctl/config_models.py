"""Config schemas. These Pydantic models are the single source of truth:
registering them below is what makes them available to Hydra composition."""

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

from mlctl.config_layer import register_config

# extra="forbid": an unknown or misspelled config key is a hard error rather than
# a silently ignored field. This is the whole point of a config-validation layer:
# `--model.alpah=10` or a stray YAML key fails fast instead of using the default.
_STRICT = ConfigDict(extra="forbid")


class RidgeConfig(BaseModel):
    model_config = _STRICT

    kind: Literal["ridge"] = "ridge"
    alpha: float = Field(default=1.0, gt=0, description="L2 regularization strength")


class GBMConfig(BaseModel):
    model_config = _STRICT

    kind: Literal["gbm"] = "gbm"
    learning_rate: float = Field(default=0.1, gt=0, le=1)
    max_iter: int = Field(default=200, gt=0, description="Number of boosting iterations")
    max_depth: int | None = Field(default=None, ge=1, description="None = unlimited")


ModelConfig = Annotated[Union[RidgeConfig, GBMConfig], Field(discriminator="kind")]


class TrainConfig(BaseModel):
    model_config = _STRICT

    experiment_name: str = "baseline"
    seed: int = Field(default=42, ge=0)
    test_size: float = Field(default=0.2, gt=0, lt=1)
    model: ModelConfig
    output_dir: str = "outputs/${experiment_name}/${model.kind}/seed_${seed}"


class EvaluateConfig(BaseModel):
    model_config = _STRICT

    run_dir: str = Field(description="Directory of a finished training run")


register_config("base_train", TrainConfig)
register_config("base_evaluate", EvaluateConfig)
register_config("ridge", RidgeConfig, group="model")
register_config("gbm", GBMConfig, group="model")
