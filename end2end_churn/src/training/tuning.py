"""Hyperparameter tuning utilities."""

import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import parallel_backend
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from ..utils.logger import get_logger

logger = get_logger("churn_training")


def perform_grid_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: ColumnTransformer,
    model: Optional[Any] = None,
    param_grid: Optional[Dict] = None,
    cv_folds: int = 5,
    scoring: str = "roc_auc",
    random_state: int = 42,
    n_jobs: int = -1,
) -> Tuple[GridSearchCV, Dict, float]:
    """
    Perform grid search with cross-validation to find best hyperparameters.

    Args:
        X_train: Training features
        y_train: Training target
        preprocessor: Preprocessing pipeline
        model: Base model to tune (default: None, uses RandomForestClassifier for backward compatibility)
        param_grid: Parameter grid for hyperparameter tuning (default: None, uses standard grid)
        cv_folds: Number of cross-validation folds (default: 5)
        scoring: Scoring metric for cross-validation (default: 'roc_auc')
        random_state: Random seed for reproducibility (default: 42)
        n_jobs: Number of parallel jobs (default: -1, use all cores)

    Returns:
        Tuple of (best_model, best_params, search_time_seconds)
    """
    logger.info("Performing hyperparameter tuning")
    logger.info(f"Grid search with {cv_folds}-fold cross-validation")

    # Use default model if none provided (backward compatibility)
    # Note: Set model's n_jobs=1 to avoid nested parallelism (GridSearchCV handles parallelization)
    if model is None:
        model = RandomForestClassifier(random_state=random_state, n_jobs=1)

    # Use default parameter grid if none provided
    if param_grid is None:
        param_grid = {
            "classifier__n_estimators": [50, 100, 200],
            "classifier__max_depth": [10, 20, None],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__min_samples_leaf": [1, 2, 4],
        }

    logger.info(f"Testing {np.prod([len(v) for v in param_grid.values()])} combinations")

    # Create pipeline with classifier
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])

    # Grid search
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv_folds,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=1,
        return_train_score=True,
    )

    # Fit with explicit backend context for clean worker shutdown
    # This prevents "ChildProcessError: No child processes" warnings in Docker
    start_time = time.time()
    with parallel_backend("loky", n_jobs=n_jobs):
        grid_search.fit(X_train, y_train)
    search_time = time.time() - start_time

    logger.info(f"✓ Grid search complete in {search_time:.1f} seconds")
    logger.info(f"✓ Best CV score (ROC AUC): {grid_search.best_score_:.4f}")
    logger.info("✓ Best parameters:")
    for param, value in grid_search.best_params_.items():
        logger.info(f"    {param}: {value}")

    return grid_search, grid_search.best_params_, search_time
