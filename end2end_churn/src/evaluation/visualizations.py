"""Visualization utilities for model evaluation."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def plot_confusion_matrix(
    y_true: pd.Series, y_pred: np.ndarray, output_path: str, title: str = "Confusion Matrix"
):
    """
    Create and save confusion matrix heatmap.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        output_path: Where to save the plot
        title: Plot title
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
        cbar_kws={"label": "Count"},
    )
    plt.title(title, fontsize=16, pad=20)
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_roc_curve(
    y_true: pd.Series, y_proba: np.ndarray, output_path: str, title: str = "ROC Curve"
) -> None:
    """
    Create and save ROC curve.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities for positive class
        output_path: Where to save the plot
        title: Plot title
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot(
        [0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random classifier (AUC = 0.500)"
    )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate (Recall)", fontsize=12)
    plt.title(title, fontsize=16, pad=20)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_precision_recall_curve(
    y_true: pd.Series, y_proba: np.ndarray, output_path: str, title: str = "Precision-Recall Curve"
) -> None:
    """
    Create and save precision-recall curve.

    For imbalanced datasets, PR curves are often more informative than ROC curves.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities for positive class
        output_path: Where to save the plot
        title: Plot title
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    avg_prec = average_precision_score(y_true, y_proba)
    baseline = y_true.mean()

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color="darkorange", lw=2, label=f"PR curve (AP = {avg_prec:.3f})")
    plt.plot(
        [0, 1],
        [baseline, baseline],
        color="navy",
        lw=2,
        linestyle="--",
        label=f"Random (AP = {baseline:.3f})",
    )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title(title, fontsize=16, pad=20)
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_feature_importances(
    feature_importance_df: pd.DataFrame, output_path: str, top_n: int = 20
) -> None:
    """
    Create and save feature importance bar chart.

    Args:
        feature_importance_df: DataFrame with 'feature' and 'importance' columns
        output_path: Where to save the plot
        top_n: Number of top features to show
    """
    top_features = feature_importance_df.head(top_n)

    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_features)), top_features["importance"], color="steelblue")
    plt.yticks(range(len(top_features)), top_features["feature"])
    plt.xlabel("Importance", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.title(f"Top {top_n} Most Important Features", fontsize=16, pad=20)
    plt.gca().invert_yaxis()  # Most important at top
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
