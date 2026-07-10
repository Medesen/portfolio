"""Data loading, schema, and a synthetic RCT generator for tests."""

from upliftlab.data.load import (
    ARMS,
    CATEGORICAL_COVARIATES,
    CONTROL,
    NUMERIC_COVARIATES,
    OUTCOMES,
    TREATMENTS,
    default_raw_path,
    design_matrix,
    load,
    two_arm,
)
from upliftlab.data.synthetic import make_synthetic_rct

__all__ = [
    "ARMS",
    "CATEGORICAL_COVARIATES",
    "CONTROL",
    "NUMERIC_COVARIATES",
    "OUTCOMES",
    "TREATMENTS",
    "default_raw_path",
    "design_matrix",
    "load",
    "two_arm",
    "make_synthetic_rct",
]
