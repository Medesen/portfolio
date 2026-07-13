"""Schema alignment and data validation utilities."""

import warnings

import numpy as np
import pandas as pd
import pandera as pa
from pandera import Check, Column


def align_schema(df: pd.DataFrame, expected_columns: list[str]) -> tuple[pd.DataFrame, dict]:
    """
    Align DataFrame to expected schema.

    Handles:
    - Missing columns (filled with NaN - imputers will handle them)
    - Extra columns (dropped - model doesn't need them)
    - Wrong order (reordered to match training schema)

    Args:
        df: Input DataFrame
        expected_columns: List of expected column names in correct order

    Returns:
        Tuple of (aligned DataFrame, alignment_info dict)
    """
    alignment_info = {"missing_columns": [], "extra_columns": [], "reordered": False}

    incoming_columns = set(df.columns)
    expected_columns_set = set(expected_columns)

    # Find missing columns
    missing = expected_columns_set - incoming_columns
    if missing:
        alignment_info["missing_columns"] = sorted(list(missing))
        warnings.warn(f"Missing columns (will be filled with NaN): {missing}")
        # Add missing columns with NaN
        # The preprocessing pipeline's imputers will handle NaN values
        for col in missing:
            df[col] = np.nan

    # Find extra columns
    extra = incoming_columns - expected_columns_set
    if extra:
        alignment_info["extra_columns"] = sorted(list(extra))
        warnings.warn(f"Extra columns (will be dropped): {extra}")
        # Drop extra columns - model doesn't need them
        df = df.drop(columns=list(extra))

    # Reorder columns to match expected order
    # This is critical - scikit-learn requires same column order as training
    if list(df.columns) != expected_columns:
        alignment_info["reordered"] = True
        df = df[expected_columns]

    return df, alignment_info


# Define expected schema with constraints for runtime validation
customer_schema = pa.DataFrameSchema(
    {
        # Demographics
        "gender": Column(str, Check.isin(["Male", "Female"]), nullable=True),
        "SeniorCitizen": Column(float, Check.isin([0, 1]), nullable=True),
        "Partner": Column(str, Check.isin(["Yes", "No"]), nullable=True),
        "Dependents": Column(str, Check.isin(["Yes", "No"]), nullable=True),
        # Account info (numeric constraints)
        "tenure": Column(float, Check.greater_than_or_equal_to(0), nullable=True),
        "MonthlyCharges": Column(float, Check.greater_than(0), nullable=True),
        "TotalCharges": Column(float, Check.greater_than_or_equal_to(0), nullable=True),
        # Contract info
        "Contract": Column(
            str, Check.isin(["Month-to-month", "One year", "Two year"]), nullable=True
        ),
        "PaperlessBilling": Column(str, Check.isin(["Yes", "No"]), nullable=True),
        "PaymentMethod": Column(str, nullable=True),  # Many possible values, skip strict validation
        # Phone services
        "PhoneService": Column(str, Check.isin(["Yes", "No"]), nullable=True),
        "MultipleLines": Column(str, Check.isin(["Yes", "No", "No phone service"]), nullable=True),
        # Internet services
        "InternetService": Column(str, Check.isin(["DSL", "Fiber optic", "No"]), nullable=True),
        "OnlineSecurity": Column(
            str, Check.isin(["Yes", "No", "No internet service"]), nullable=True
        ),
        "OnlineBackup": Column(
            str, Check.isin(["Yes", "No", "No internet service"]), nullable=True
        ),
        "DeviceProtection": Column(
            str, Check.isin(["Yes", "No", "No internet service"]), nullable=True
        ),
        "TechSupport": Column(str, Check.isin(["Yes", "No", "No internet service"]), nullable=True),
        "StreamingTV": Column(str, Check.isin(["Yes", "No", "No internet service"]), nullable=True),
        "StreamingMovies": Column(
            str, Check.isin(["Yes", "No", "No internet service"]), nullable=True
        ),
    },
    strict=False,
    coerce=True,
)  # strict=False allows extra columns, coerce=True tries type conversion


def validate_data(df: pd.DataFrame, enable_validation: bool = True) -> tuple[bool, list[str]]:
    """
    Validate DataFrame against expected schema using Pandera.

    Args:
        df: DataFrame to validate
        enable_validation: Whether to perform validation (can disable for performance)

    Returns:
        Tuple of (is_valid, list of error messages)
    """
    if not enable_validation:
        return True, []

    try:
        customer_schema.validate(df, lazy=True)
        return True, []
    except pa.errors.SchemaErrors as e:
        # Extract unique error messages
        errors = []
        for _, row in e.failure_cases.iterrows():
            error_msg = f"{row['column']}: {row['check']}"
            if error_msg not in errors:
                errors.append(error_msg)
        return False, errors
    except Exception as e:
        # Catch other validation errors
        return False, [f"Validation error: {str(e)}"]


def generate_alignment_warnings(alignment_info: dict) -> list[str]:
    """
    Generate user-friendly warning messages from alignment info.

    Args:
        alignment_info: Dictionary from align_schema function

    Returns:
        List of warning messages
    """
    warnings_list = []

    if alignment_info["missing_columns"]:
        warnings_list.append(
            f"Missing columns: {', '.join(alignment_info['missing_columns'])}. "
            f"These will be imputed using model defaults."
        )

    if alignment_info["extra_columns"]:
        warnings_list.append(f"Extra columns ignored: {', '.join(alignment_info['extra_columns'])}")

    if alignment_info["reordered"]:
        warnings_list.append("Column order adjusted to match model expectations")

    # Warn if too many columns are missing (model may not perform well)
    if len(alignment_info["missing_columns"]) > 5:
        warnings_list.append(
            f"WARNING: {len(alignment_info['missing_columns'])} columns missing. "
            f"Prediction quality may be significantly degraded."
        )

    return warnings_list
