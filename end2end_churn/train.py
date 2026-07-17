"""
CLI entry point for training the churn prediction model.

The pipeline itself lives in ``src/training``: ``pipeline.py`` orchestrates
data preparation and model fitting, delegating evaluation and threshold
tuning to ``reporting.py``, artifact publication to ``artifacts.py``, and
MLflow model logging to ``mlflow_logging.py``. This shim parses arguments
and calls ``main()`` so existing surfaces keep working unchanged.

Usage (via Docker/Make - recommended):
    make train                  # default config (Random Forest)
    make train-quick            # quick config for smoke runs
    make train-prod             # production config
    make train-xgboost / train-logreg / train-register   # variants

Direct usage (development only):
    python train.py [--config CONFIG_PATH] [--model-type MODEL_TYPE]
"""

import argparse

from src.training.pipeline import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train churn prediction model with configuration management"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config file (YAML or JSON). If not provided, uses defaults.",
    )
    parser.add_argument(
        "--env", action="store_true", help="Load configuration from environment variables"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["random_forest", "xgboost", "logistic_regression"],
        help="Model type to train (overrides config file). Options: random_forest, xgboost, logistic_regression",
    )

    args = parser.parse_args()

    main(config_path=args.config, use_env=args.env, model_type=args.model_type)
