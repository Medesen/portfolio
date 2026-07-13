"""
Model Factory Pattern for Multiple ML Algorithms

This module provides a factory pattern for instantiating different ML models
(Random Forest, XGBoost, Logistic Regression) with appropriate hyperparameter grids.

The factory pattern makes it easy to:
- Add new models without modifying training code
- Compare multiple algorithms systematically
- Swap models via configuration
"""

from typing import Optional

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def get_model(model_type: str, random_state: int = 42):
    """
    Get an instantiated model based on type.

    Args:
        model_type: One of 'random_forest', 'xgboost', 'logistic_regression'
        random_state: Random seed for reproducibility

    Returns:
        Instantiated model with default parameters

    Raises:
        ValueError: If model_type is not recognized
    """
    # Set n_jobs=1 for all models to avoid nested parallelism
    # GridSearchCV handles parallelization at the top level
    models = {
        "random_forest": RandomForestClassifier(
            random_state=random_state, n_jobs=1  # Avoid nested parallelism with GridSearchCV
        ),
        "xgboost": xgb.XGBClassifier(
            random_state=random_state,
            eval_metric="logloss",
            use_label_encoder=False,  # Suppress label encoder warnings
            objective="binary:logistic",  # Explicit binary classification
            verbosity=0,  # Suppress training output warnings
            n_jobs=1,  # Avoid nested parallelism with GridSearchCV
        ),
        "logistic_regression": LogisticRegression(
            random_state=random_state,
            max_iter=1000,  # Increase for convergence
            solver="lbfgs",  # Good for small-to-medium datasets
            n_jobs=1,  # Avoid nested parallelism with GridSearchCV
        ),
    }

    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. " f"Available: {list(models.keys())}")

    return models[model_type]


def get_param_grid(model_type: str) -> dict[str, list]:
    """
    Get hyperparameter grid for a specific model type.

    These grids are designed for:
    - Random Forest: Standard ensemble parameters
    - XGBoost: Learning rate, tree depth, regularization
    - Logistic Regression: Regularization strength and penalty type

    Args:
        model_type: One of 'random_forest', 'xgboost', 'logistic_regression'

    Returns:
        Dictionary mapping parameter names to lists of values

    Raises:
        ValueError: If model_type is not recognized
    """
    param_grids = {
        "random_forest": {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__max_depth": [10, 20, None],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__min_samples_leaf": [1, 2, 4],
        },
        "xgboost": {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [3, 6, 10],
            "classifier__learning_rate": [0.01, 0.1, 0.3],
            "classifier__subsample": [0.8, 1.0],
            "classifier__colsample_bytree": [0.8, 1.0],
        },
        "logistic_regression": {
            # Regularization strength (smaller = stronger regularization)
            "classifier__C": [0.01, 0.1, 1.0, 10.0, 100.0],
            # Regularization type
            "classifier__penalty": ["l1", "l2"],
            # Solver must match penalty (saga supports both l1 and l2)
            "classifier__solver": ["saga"],
        },
    }

    if model_type not in param_grids:
        raise ValueError(
            f"Unknown model type: {model_type}. " f"Available: {list(param_grids.keys())}"
        )

    return param_grids[model_type]


def get_quick_param_grid(model_type: str) -> dict[str, list]:
    """
    Get minimal hyperparameter grid for fast experimentation.

    These grids have single values (no search) for rapid iteration.

    Args:
        model_type: One of 'random_forest', 'xgboost', 'logistic_regression'

    Returns:
        Dictionary with single-value parameter lists (no grid search)

    Raises:
        ValueError: If model_type is not recognized
    """
    quick_grids = {
        "random_forest": {
            "classifier__n_estimators": [100],
            "classifier__max_depth": [10],
            "classifier__min_samples_split": [2],
            "classifier__min_samples_leaf": [1],
        },
        "xgboost": {
            "classifier__n_estimators": [100],
            "classifier__max_depth": [6],
            "classifier__learning_rate": [0.1],
            "classifier__subsample": [1.0],
            "classifier__colsample_bytree": [1.0],
        },
        "logistic_regression": {
            "classifier__C": [1.0],
            "classifier__penalty": ["l2"],
            "classifier__solver": ["lbfgs"],  # Faster than saga for single config
        },
    }

    if model_type not in quick_grids:
        raise ValueError(
            f"Unknown model type: {model_type}. " f"Available: {list(quick_grids.keys())}"
        )

    return quick_grids[model_type]


def get_model_display_name(model_type: str) -> str:
    """
    Get human-readable display name for model type.

    Args:
        model_type: Internal model type identifier

    Returns:
        Display name for reports and logs
    """
    display_names = {
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
        "logistic_regression": "Logistic Regression",
    }
    return display_names.get(model_type, model_type)


def get_all_model_types() -> list[str]:
    """
    Get list of all available model types.

    Returns:
        List of model type identifiers
    """
    return ["random_forest", "xgboost", "logistic_regression"]
