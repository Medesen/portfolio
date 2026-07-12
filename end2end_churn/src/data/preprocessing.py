"""Data preprocessing and splitting utilities."""

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from ..utils.logger import get_logger

logger = get_logger("churn_training")


def preprocess_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Clean and prepare data for modeling.

    Args:
        df: Raw DataFrame with all columns

    Returns:
        Tuple of (features, target)
    """
    logger.info("Preprocessing data")

    # Drop identifier column
    df = df.drop("customerID", axis=1)

    # Convert TotalCharges from string to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Convert integer columns to float to handle missing values (NaN)
    # This prevents MLflow schema mismatch warnings and allows proper NaN handling
    # Only convert if columns exist (for compatibility with minimal test fixtures)
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].astype(float)
    if "tenure" in df.columns:
        df["tenure"] = df["tenure"].astype(float)

    # Separate features and target
    X = df.drop("Churn", axis=1)
    y = df["Churn"].map({"Yes": 1, "No": 0})

    logger.info(f"✓ Features shape: {X.shape}")
    logger.info(f"✓ Churn rate: {y.mean():.2%}")

    return X, y


def create_three_way_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    val_size: float = 0.25,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Create train/validation/test splits.

    The validation set is used for model selection and hyperparameter tuning.
    The test set is ONLY touched at the very end for final evaluation.

    Args:
        X: Features
        y: Target
        test_size: Proportion of data to use for test set (default: 0.2)
        val_size: Proportion of train+val to use for validation (default: 0.25)
        random_state: Random seed for reproducibility
        stratify: Whether to stratify splits by target

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    logger.info("Creating 3-way split")

    # First split: (1-test_size) train+val, test_size test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if stratify else None
    )

    # Second split: (1-val_size) train, val_size val (of remaining data)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_size,
        random_state=random_state,
        stratify=y_temp if stratify else None,
    )

    logger.info(f"✓ Train:      {len(X_train):4d} samples ({len(X_train)/len(X)*100:.1f}%)")
    logger.info(f"✓ Validation: {len(X_val):4d} samples ({len(X_val)/len(X)*100:.1f}%)")
    logger.info(f"✓ Test:       {len(X_test):4d} samples ({len(X_test)/len(X)*100:.1f}%)")

    return X_train, X_val, X_test, y_train, y_val, y_test
