"""
Configuration management using Pydantic for type-safe, validated configs.

This module provides configuration classes for all components of the churn prediction
system, supporting loading from files (YAML/JSON), environment variables, or defaults.
"""

import json
import os
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field, field_validator


def load_secret(env_var_name: str) -> Optional[str]:
    """
    Load secret from file or environment variable.

    Priority order:
    1. File path from {ENV_VAR_NAME}_FILE environment variable
    2. Direct value from {ENV_VAR_NAME} environment variable
    3. None

    This pattern supports both Docker secrets and K8s secrets.

    Args:
        env_var_name: Name of environment variable (e.g., "SERVICE_TOKEN")

    Returns:
        Secret value or None

    Example:
        # Docker: SERVICE_TOKEN_FILE=/run/secrets/service_token
        # K8s: SERVICE_TOKEN_FILE=/var/secrets/service_token
        # Dev: SERVICE_TOKEN=dev-token-123

        token = load_secret("SERVICE_TOKEN")
    """
    # Try file-based secret first (Docker/K8s secrets)
    secret_file = os.getenv(f"{env_var_name}_FILE")
    if secret_file and Path(secret_file).exists():
        try:
            with open(secret_file, "r") as f:
                return f.read().strip()
        except Exception as e:
            # Log warning but don't expose secret path details
            print(f"Warning: Failed to read secret from file: {e}")

    # Fall back to environment variable (for development)
    return os.getenv(env_var_name)


class DataConfig(BaseModel):
    """Configuration for data loading and processing."""

    data_path: str = Field("data/dataset.arff", description="Path to training data")
    random_state: int = Field(42, description="Random seed for reproducibility")
    test_size: float = Field(0.2, ge=0.0, le=1.0, description="Test set proportion")
    val_size: float = Field(
        0.25, ge=0.0, le=1.0, description="Validation set proportion (of train+val)"
    )
    stratify: bool = Field(True, description="Stratify splits by target")

    @field_validator("test_size", "val_size")
    @classmethod
    def check_split_size(cls, v):
        """Validate that split sizes are between 0 and 1 (exclusive)."""
        if not 0 < v < 1:
            raise ValueError("Split size must be between 0 and 1 (exclusive)")
        return v


class ModelConfig(BaseModel):
    """Configuration for model training and hyperparameter tuning."""

    model_type: str = Field(
        "random_forest", description="Model type (random_forest, xgboost, logistic_regression)"
    )
    n_estimators_grid: list[int] = Field([100, 200], description="Number of trees to try")
    max_depth_grid: list[Optional[int]] = Field([10, 20, None], description="Max depth to try")
    min_samples_split_grid: list[int] = Field([2, 5], description="Min samples to split")
    min_samples_leaf_grid: list[int] = Field([1, 2], description="Min samples per leaf")
    cv_folds: int = Field(5, ge=2, description="Cross-validation folds")
    scoring: str = Field("roc_auc", description="Scoring metric for CV")
    n_jobs: int = Field(-1, description="Parallel jobs (-1 = all cores)")

    @field_validator("model_type")
    @classmethod
    def check_model_type(cls, v):
        """Validate that model_type is one of the supported types."""
        valid_types = ["random_forest", "xgboost", "logistic_regression"]
        if v not in valid_types:
            raise ValueError(f"model_type must be one of {valid_types}, got {v}")
        return v

    @field_validator("n_estimators_grid", "min_samples_split_grid", "min_samples_leaf_grid")
    @classmethod
    def check_positive_values(cls, v):
        """Validate that all grid values are positive."""
        if any(val <= 0 for val in v):
            raise ValueError("All grid values must be positive")
        return v

    @field_validator("max_depth_grid")
    @classmethod
    def check_max_depth(cls, v):
        """Validate that max_depth values are positive or None."""
        for val in v:
            if val is not None and val <= 0:
                raise ValueError("max_depth must be positive or None")
        return v

    def to_param_grid(self) -> dict[str, list]:
        """
        Convert configuration to scikit-learn param_grid format.

        This is the hyperparameter grid source for random_forest training: the
        config's grid fields map directly onto Random Forest parameters, so the
        --quick config and any custom config are honored during grid search.
        XGBoost and Logistic Regression have no config grid fields and use the
        model-specific grids from model_factory.get_param_grid() instead.

        Returns:
            Dictionary mapping parameter names to lists of values to try
        """
        return {
            "classifier__n_estimators": self.n_estimators_grid,
            "classifier__max_depth": self.max_depth_grid,
            "classifier__min_samples_split": self.min_samples_split_grid,
            "classifier__min_samples_leaf": self.min_samples_leaf_grid,
        }


class MLflowConfig(BaseModel):
    """
    Configuration for MLflow experiment tracking.

    Note: MLflow is always enabled for tracking and reproducibility.
    This config controls MLflow settings like tracking URI and experiment name.
    """

    tracking_uri: str = Field("./mlruns", description="MLflow tracking URI")
    experiment_name: str = Field("churn_prediction", description="Experiment name")
    log_models: bool = Field(True, description="Log models to MLflow")
    log_artifacts: bool = Field(True, description="Log artifacts to MLflow")


class ServiceConfig(BaseModel):
    """Configuration for FastAPI service."""

    host: str = Field("0.0.0.0", description="Service host")
    port: int = Field(8000, ge=1, le=65535, description="Service port")
    log_level: str = Field("INFO", description="Logging level")
    max_batch_size: int = Field(1000, ge=1, description="Maximum batch size for predictions")
    max_request_size_mb: int = Field(10, ge=1, description="Maximum request size in MB")
    service_token: Optional[str] = Field(None, description="Bearer token for authentication")

    @field_validator("log_level")
    @classmethod
    def check_log_level(cls, v):
        """Validate log level is one of the standard Python logging levels."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()

    @classmethod
    def from_env(cls) -> "ServiceConfig":
        """
        Load service config from environment with secure secret support.

        Secrets are loaded using load_secret() which supports:
        - File-based secrets (Docker/K8s): SERVICE_TOKEN_FILE
        - Environment variables (dev): SERVICE_TOKEN

        Returns:
            ServiceConfig instance with values from environment
        """
        return cls(
            host=os.getenv("SERVICE_HOST", "0.0.0.0"),
            port=int(os.getenv("SERVICE_PORT", "8000")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            max_batch_size=int(os.getenv("MAX_BATCH_SIZE", "1000")),
            max_request_size_mb=int(os.getenv("MAX_REQUEST_SIZE_MB", "10")),
            service_token=load_secret("SERVICE_TOKEN"),  # Use secure secret loader
        )


class DriftConfig(BaseModel):
    """Configuration for drift detection."""

    numeric_threshold: float = Field(
        0.2, ge=0.0, le=1.0, description="Numeric drift threshold (Kolmogorov-Smirnov)"
    )
    categorical_threshold: float = Field(
        0.25, ge=0.0, le=1.0, description="PSI threshold for categorical features"
    )
    prediction_threshold: float = Field(
        0.1, ge=0.0, le=1.0, description="Prediction drift threshold (PSI)"
    )
    min_sample_size: int = Field(100, ge=10, description="Minimum sample size for drift detection")
    max_drift_batch_size: int = Field(
        1000, ge=1, description="Maximum batch size for drift analysis"
    )


class TrainingConfig(BaseModel):
    """
    Complete training configuration combining all component configs.

    This is the top-level configuration class that can be loaded from files
    or environment variables, and passed to the training script.
    """

    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    drift: DriftConfig = Field(default_factory=DriftConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "TrainingConfig":
        """
        Load configuration from YAML file.

        Args:
            path: Path to YAML file

        Returns:
            TrainingConfig instance

        Raises:
            FileNotFoundError: If file doesn't exist
            yaml.YAMLError: If YAML is invalid
            pydantic.ValidationError: If config values are invalid
        """
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    @classmethod
    def from_json(cls, path: str) -> "TrainingConfig":
        """
        Load configuration from JSON file.

        Args:
            path: Path to JSON file

        Returns:
            TrainingConfig instance

        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If JSON is invalid
            pydantic.ValidationError: If config values are invalid
        """
        with open(path, "r") as f:
            config_dict = json.load(f)
        return cls(**config_dict)

    @classmethod
    def from_env(cls) -> "TrainingConfig":
        """
        Load configuration from environment variables.

        This is useful for containerized deployments (e.g., Kubernetes)
        where configuration is injected via environment variables.

        Environment variable names:
        - DATA_PATH, RANDOM_STATE, TEST_SIZE, VAL_SIZE
        - CV_FOLDS, SCORING, N_JOBS
        - MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME
        - DRIFT_THRESHOLD_NUMERIC, DRIFT_THRESHOLD_CATEGORICAL, etc.

        Returns:
            TrainingConfig instance with values from environment
        """
        return cls(
            data=DataConfig(
                data_path=os.getenv("DATA_PATH", "data/dataset.arff"),
                random_state=int(os.getenv("RANDOM_STATE", "42")),
                test_size=float(os.getenv("TEST_SIZE", "0.2")),
                val_size=float(os.getenv("VAL_SIZE", "0.25")),
                stratify=os.getenv("STRATIFY", "true").lower() == "true",
            ),
            model=ModelConfig(
                cv_folds=int(os.getenv("CV_FOLDS", "5")),
                scoring=os.getenv("SCORING", "roc_auc"),
                n_jobs=int(os.getenv("N_JOBS", "-1")),
            ),
            mlflow=MLflowConfig(
                tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "./mlruns"),
                experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME", "churn_prediction"),
                log_models=os.getenv("MLFLOW_LOG_MODELS", "true").lower() == "true",
                log_artifacts=os.getenv("MLFLOW_LOG_ARTIFACTS", "true").lower() == "true",
            ),
            drift=DriftConfig(
                numeric_threshold=float(os.getenv("DRIFT_THRESHOLD_NUMERIC", "0.2")),
                categorical_threshold=float(os.getenv("DRIFT_THRESHOLD_CATEGORICAL", "0.25")),
                prediction_threshold=float(os.getenv("DRIFT_THRESHOLD_PREDICTION", "0.1")),
                min_sample_size=int(os.getenv("DRIFT_MIN_SAMPLE_SIZE", "100")),
                max_drift_batch_size=int(os.getenv("MAX_DRIFT_BATCH_SIZE", "1000")),
            ),
        )

    def save(self, path: str):
        """
        Save configuration to file.

        File format is determined by extension (.yaml, .yml, or .json).

        Args:
            path: Output path

        Raises:
            ValueError: If extension is not .yaml, .yml, or .json
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        if path.endswith(".yaml") or path.endswith(".yml"):
            with open(path, "w") as f:
                yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)
        elif path.endswith(".json"):
            with open(path, "w") as f:
                json.dump(self.model_dump(), f, indent=2)
        else:
            raise ValueError("Config file must have extension .yaml, .yml, or .json")

    def to_dict(self) -> dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            Dictionary representation of config
        """
        return self.model_dump()
