"""The config layer: Hydra handles composition, Pydantic handles validation.

Division of labor:
- Pydantic models are the single source of truth for config schemas and defaults.
- Hydra owns composition: defaults lists, config groups, CLI overrides, multirun.
- At the boundary, the composed config is validated exactly once and commands
  receive a typed Pydantic model, never a raw DictConfig.
"""

from collections.abc import Callable
from functools import wraps
from typing import Any

from hydra import main as hydra_main
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import MissingMandatoryValue
from pydantic import BaseModel, ValidationError


def pydantic_to_node(cls: type[BaseModel]) -> dict[str, Any]:
    """Turn a Pydantic model class into a node for Hydra's ConfigStore.

    Fields without defaults are stored as Hydra's `???` sentinel, which makes
    composition fail loudly unless a config group, YAML file, or CLI override
    supplies a value. Fields with defaults are serialized as plain data.

    Deliberately minimal: a required nested model is also stored as `???` and
    is expected to be filled in via a config group selection.
    """
    node: dict[str, Any] = {}
    for name, field in cls.model_fields.items():
        if field.is_required():
            node[name] = "???"
        else:
            default = field.get_default(call_default_factory=True)
            if isinstance(default, BaseModel):
                default = default.model_dump(mode="json")
            node[name] = default
    return node


def register_config(name: str, cls: type[BaseModel], group: str | None = None) -> None:
    """Register a Pydantic model with Hydra's ConfigStore."""
    ConfigStore.instance().store(name=name, node=pydantic_to_node(cls), group=group)


def validate_config(model_cls: type[BaseModel], cfg: DictConfig) -> BaseModel:
    """Resolve a composed Hydra config and check it against a Pydantic model.

    Both failure modes exit with a short, readable message and status 1 — a
    config CLI should never greet the user with a stack trace.
    """
    try:
        data = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    except MissingMandatoryValue as e:
        raise SystemExit(f"Required config value was never set:\n{e}") from e
    try:
        return model_cls.model_validate(data)
    except ValidationError as e:
        raise SystemExit(f"Invalid configuration:\n{e}") from e


def config_command(
    model_cls: type[BaseModel], *, config_path: str | None, config_name: str
) -> Callable:
    """Decorator that turns a function into a Hydra CLI command with a validated config.

    Hydra composes the config (defaults list, config groups, CLI overrides),
    the result is validated against `model_cls`, and the wrapped function
    receives the typed model instead of a raw DictConfig.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(cfg: DictConfig) -> Any:
            return func(validate_config(model_cls, cfg))

        return hydra_main(
            config_path=config_path, config_name=config_name, version_base=None
        )(wrapper)

    return decorator
