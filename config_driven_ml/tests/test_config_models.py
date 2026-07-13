"""Schema tests: the discriminated union and field constraints."""

import pytest
from pydantic import ValidationError

from mlctl.config_models import GBMConfig, RidgeConfig, TrainConfig


def make_train_config(model: dict) -> TrainConfig:
    return TrainConfig.model_validate({"model": model})


def test_discriminator_selects_ridge():
    cfg = make_train_config({"kind": "ridge", "alpha": 2.5})
    assert isinstance(cfg.model, RidgeConfig)
    assert cfg.model.alpha == 2.5


def test_discriminator_selects_gbm():
    cfg = make_train_config({"kind": "gbm", "max_iter": 50})
    assert isinstance(cfg.model, GBMConfig)
    assert cfg.model.max_iter == 50


def test_unknown_kind_rejected():
    with pytest.raises(ValidationError):
        make_train_config({"kind": "xgboost"})


def test_negative_alpha_rejected():
    with pytest.raises(ValidationError):
        make_train_config({"kind": "ridge", "alpha": -1.0})


def test_test_size_bounds_enforced():
    with pytest.raises(ValidationError):
        TrainConfig.model_validate(
            {"model": {"kind": "gbm"}, "test_size": 1.5}
        )


def test_model_is_required():
    with pytest.raises(ValidationError):
        TrainConfig.model_validate({})


def test_unknown_field_rejected_on_model():
    # A misspelled hyperparameter (alpah) must fail, not be silently ignored.
    with pytest.raises(ValidationError):
        RidgeConfig.model_validate({"kind": "ridge", "alpha": 1.0, "alpah": 10.0})


def test_unknown_field_rejected_on_train_config():
    with pytest.raises(ValidationError):
        TrainConfig.model_validate({"model": {"kind": "gbm"}, "seeed": 7})
