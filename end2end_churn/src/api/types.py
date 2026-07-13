"""Type definitions for API layer.

This module provides TypedDict and other type definitions to ensure type safety
across the API layer, replacing raw dicts with properly typed structures.

Type Safety Implementation
"""

from typing import Optional, TypedDict

from sklearn.pipeline import Pipeline


class ModelCache(TypedDict):
    """Type-safe model cache structure.

    Stored in app.state.model_cache for multi-worker safety.
    Each worker process maintains its own cache in memory.

    The metadata dict is cached at model-load time so every consumer (drift
    analysis, health, drift info) sees the metadata belonging to the model that
    is actually serving. Re-reading the latest metadata from disk per request
    would go stale the moment a retrain overwrites the files while the old
    model remains cached until restart.

    Attributes:
        model: Loaded scikit-learn pipeline (None if not yet loaded)
        version: Model version identifier (e.g., "20251031_165906")
        features: Expected feature names in correct order
        threshold: Tuned classification threshold from training
        metadata: Full metadata dict loaded together with the model (None if
            no metadata file was found at load time)
    """

    model: Optional[Pipeline]
    version: Optional[str]
    features: Optional[list[str]]
    threshold: float
    metadata: Optional[dict]


class DriftMetrics(TypedDict, total=False):
    """Drift detection metrics for numeric features.

    Using total=False allows optional fields while maintaining type safety.
    """

    reference_mean: float
    current_mean: float
    mean_change: float
    reference_std: float
    current_std: float
    std_change: float
    threshold: float
    relative_drift_detected: bool
    ks_statistic: Optional[float]
    ks_p_value: Optional[float]
    ks_drift_detected: bool
    ks_p_value_threshold: Optional[float]


class CategoricalDriftMetrics(TypedDict, total=False):
    """Drift detection metrics for categorical features."""

    psi: float
    threshold: float
    reference_distribution: dict[str, float]
    current_distribution: dict[str, float]


class PredictionDriftMetrics(TypedDict, total=False):
    """Drift detection metrics for predictions."""

    reference_positive_rate: float
    current_positive_rate: float
    absolute_change: float
    threshold: float
