"""Model training and evaluation utilities."""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from ..evaluation.metrics import compute_metrics


def evaluate_model(
    model: Pipeline, X_val: pd.DataFrame, y_val: pd.Series
) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """
    Evaluate model on validation set.

    Args:
        model: Trained model pipeline
        X_val: Validation features
        y_val: Validation target

    Returns:
        Tuple of (metrics_dict, predictions, probabilities)
    """
    from ..utils.logger import get_logger

    logger = get_logger("churn_training")

    logger.info("Evaluating on validation set")
    y_val_pred = model.predict(X_val)
    y_val_proba = model.predict_proba(X_val)[:, 1]

    metrics = compute_metrics(y_val, y_val_pred, y_val_proba)

    logger.info(f"  Accuracy:           {metrics['accuracy']:.4f}")
    logger.info(f"  Precision:          {metrics['precision']:.4f}")
    logger.info(f"  Recall:             {metrics['recall']:.4f}")
    logger.info(f"  F1 Score:           {metrics['f1']:.4f}")
    logger.info(f"  ROC AUC:            {metrics['roc_auc']:.4f}")
    logger.info(f"  Average Precision:  {metrics['avg_precision']:.4f}")

    return metrics, y_val_pred, y_val_proba


def extract_feature_importances(
    model: Pipeline,
    numeric_features: List[str],
    categorical_features: List[str],
    X_train: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extract feature importances from the trained model.

    Works with:
    - Tree-based models (Random Forest, XGBoost): Uses feature_importances_
    - Linear models (Logistic Regression): Uses absolute value of coefficients

    Note: After one-hot encoding, categorical features become multiple columns.
    We need to get the feature names from the preprocessor.

    Args:
        model: Trained model pipeline
        numeric_features: List of numeric feature names
        categorical_features: List of categorical feature names
        X_train: Training features (not used, kept for API consistency)

    Returns:
        DataFrame with features and their importance scores, sorted by importance
    """
    from ..utils.logger import get_logger

    logger = get_logger("churn_training")

    logger.info("Extracting feature importances")

    # Get the classifier
    classifier = model.named_steps["classifier"]

    # Extract importances based on model type
    if hasattr(classifier, "feature_importances_"):
        # Tree-based models (Random Forest, XGBoost)
        importances = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        # Linear models (Logistic Regression)
        # Use absolute value of coefficients as importance
        importances = np.abs(classifier.coef_[0])
    else:
        logger.warning(
            f"Model type {type(classifier).__name__} doesn't support feature importances"
        )
        return pd.DataFrame({"feature": [], "importance": []})

    # Get feature names after preprocessing
    preprocessor = model.named_steps["preprocessor"]

    # Get categorical feature names after one-hot encoding
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_feature_names = cat_encoder.get_feature_names_out(categorical_features)

    # Combine numeric and categorical feature names
    all_feature_names = numeric_features + list(cat_feature_names)

    # Create DataFrame
    feature_importance_df = pd.DataFrame(
        {"feature": all_feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)

    logger.info(f"✓ Extracted {len(feature_importance_df)} features")
    logger.info("Top 10 most important features:")
    for idx, row in feature_importance_df.head(10).iterrows():
        logger.info(f"  {row['feature']:40s} {row['importance']:.4f}")

    return feature_importance_df
