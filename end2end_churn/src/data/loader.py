"""Data loading utilities for ARFF format."""

import arff
import pandas as pd

from ..utils.logger import get_logger

logger = get_logger("churn_training")


def load_data(data_path: str) -> pd.DataFrame:
    """
    Load ARFF file and convert to pandas DataFrame.

    Args:
        data_path: Path to ARFF file

    Returns:
        DataFrame with all features and target
    """
    logger.info(f"Loading data from {data_path}")
    with open(data_path, "r") as f:
        dataset = arff.load(f)

    df = pd.DataFrame(dataset["data"], columns=[attr[0] for attr in dataset["attributes"]])
    logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
    return df
