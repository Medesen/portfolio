"""
Drift detection utilities for monitoring input data and prediction distributions.

This module implements:
- Population Stability Index (PSI) for categorical features
- Statistical drift detection for numeric features (relative change + KS test)
- Prediction drift detection
- Comprehensive drift analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from scipy import stats


def calculate_psi(
    expected: Dict[str, float],
    actual: Dict[str, float],
    threshold: float = 0.1
) -> Tuple[float, str]:
    """
    Calculate Population Stability Index (PSI) for categorical features.
    
    PSI measures how much a distribution has shifted.
    
    PSI < 0.1: No significant change
    0.1 <= PSI < 0.25: Moderate change
    PSI >= 0.25: Significant change (likely drift)
    
    Args:
        expected: Distribution from training (category -> proportion)
        actual: Distribution from current data (category -> proportion)
        threshold: PSI threshold for drift alert
        
    Returns:
        Tuple of (PSI value, interpretation)
    """
    # Combine categories from both distributions
    all_categories = set(expected.keys()) | set(actual.keys())
    
    psi = 0.0
    for category in all_categories:
        # Get proportions (default to small value to avoid log(0))
        exp_prop = expected.get(category, 0.0001)
        act_prop = actual.get(category, 0.0001)
        
        # PSI formula: (actual - expected) * ln(actual / expected)
        psi += (act_prop - exp_prop) * np.log(act_prop / exp_prop)
    
    # Interpret
    if psi < 0.1:
        interpretation = "stable"
    elif psi < 0.25:
        interpretation = "moderate_drift"
    else:
        interpretation = "significant_drift"
    
    return float(psi), interpretation


def detect_numeric_drift(
    reference_stats: Dict[str, float],
    current_data: pd.Series,
    threshold: float = 0.2,
    ks_p_value_threshold: float = 0.05
) -> Dict[str, Any]:
    """
    Detect drift in numeric features using multiple methods:
    1. Relative change in mean/std (fast, interpretable)
    2. Kolmogorov-Smirnov test (robust, distribution-based)
    
    Args:
        reference_stats: Training statistics (mean, std, samples, etc.)
        current_data: Current feature values
        threshold: Relative change threshold (default 20%)
        ks_p_value_threshold: KS test significance level (default 0.05)
        
    Returns:
        Dictionary with drift metrics and status
        
    Note:
        Drift is detected if EITHER relative change OR KS test indicates drift.
        This provides both interpretable metrics (mean/std change) and
        statistical rigor (KS test).
    """
    current_values = current_data.dropna()
    
    if len(current_values) == 0:
        return {
            'drift_detected': False,
            'reason': 'insufficient_data',
            'metrics': {}
        }
    
    # Compute current statistics
    current_mean = float(current_values.mean())
    current_std = float(current_values.std())
    
    # Method 1: Relative changes (fast, interpretable). The denominator must be
    # the ABSOLUTE reference value: a signed denominator makes the change
    # negative for negative-mean features, so drift could never trigger there.
    mean_change = abs(current_mean - reference_stats['mean']) / (abs(reference_stats['mean']) + 1e-10)
    std_change = abs(current_std - reference_stats['std']) / (abs(reference_stats['std']) + 1e-10)
    
    relative_drift = (mean_change > threshold) or (std_change > threshold)
    
    # Method 2: Kolmogorov-Smirnov test (statistical rigor)
    # KS test compares two distributions to detect if they differ significantly
    ks_statistic = None
    ks_p_value = None
    ks_drift = False
    
    if 'samples' in reference_stats and reference_stats['samples'] is not None:
        try:
            # Perform two-sample KS test
            ks_result = stats.ks_2samp(reference_stats['samples'], current_values)
            ks_statistic = float(ks_result.statistic)
            ks_p_value = float(ks_result.pvalue)
            
            # Drift detected if p-value < threshold (reject null hypothesis of same distribution)
            ks_drift = ks_p_value < ks_p_value_threshold
        except Exception as e:
            # KS test failed (e.g., not enough samples), fall back to relative change only
            ks_statistic = None
            ks_p_value = None
            ks_drift = False
    
    # Combined drift detection: drift if EITHER method detects it
    drift_detected = relative_drift or ks_drift
    
    metrics = {
        # Relative change metrics
        'reference_mean': reference_stats['mean'],
        'current_mean': current_mean,
        'mean_change': float(mean_change),
        'reference_std': reference_stats['std'],
        'current_std': current_std,
        'std_change': float(std_change),
        'threshold': threshold,
        'relative_drift_detected': relative_drift,
        # KS test metrics (if available)
        'ks_statistic': ks_statistic,
        'ks_p_value': ks_p_value,
        'ks_drift_detected': ks_drift,
        'ks_p_value_threshold': ks_p_value_threshold if ks_statistic is not None else None
    }
    
    # Determine reason
    if not drift_detected:
        reason = 'stable'
    elif relative_drift and ks_drift:
        reason = 'drift_detected_by_both_methods'
    elif relative_drift:
        reason = 'drift_detected_by_relative_change'
    elif ks_drift:
        reason = 'drift_detected_by_ks_test'
    else:
        reason = 'stable'
    
    return {
        'drift_detected': drift_detected,
        'reason': reason,
        'metrics': metrics
    }


def detect_categorical_drift(
    reference_stats: Dict[str, Any],
    current_data: pd.Series,
    threshold: float = 0.25
) -> Dict[str, Any]:
    """
    Detect drift in categorical features using PSI.
    
    Args:
        reference_stats: Training distribution
        current_data: Current feature values
        threshold: PSI threshold for drift
        
    Returns:
        Dictionary with drift metrics and status
    """
    # Compute current distribution
    current_dist = current_data.value_counts(normalize=True, dropna=False).to_dict()
    
    # Calculate PSI
    reference_dist = reference_stats['distribution']
    psi, interpretation = calculate_psi(reference_dist, current_dist, threshold)
    
    drift_detected = (psi >= threshold)
    
    return {
        'drift_detected': drift_detected,
        'reason': interpretation,
        'metrics': {
            'psi': psi,
            'threshold': threshold,
            'reference_distribution': reference_dist,
            'current_distribution': current_dist
        }
    }


def detect_prediction_drift(
    reference_positive_rate: float,
    current_predictions: np.ndarray,
    threshold: float = 0.1
) -> Dict[str, Any]:
    """
    Detect drift in prediction distribution.
    
    Args:
        reference_positive_rate: Training positive rate (e.g., 0.27)
        current_predictions: Current model predictions
        threshold: Absolute change threshold
        
    Returns:
        Dictionary with drift metrics and status
    """
    current_positive_rate = float(current_predictions.mean())
    absolute_change = abs(current_positive_rate - reference_positive_rate)
    
    drift_detected = (absolute_change > threshold)
    
    return {
        'drift_detected': drift_detected,
        'reason': 'prediction_shift' if drift_detected else 'stable',
        'metrics': {
            'reference_positive_rate': reference_positive_rate,
            'current_positive_rate': current_positive_rate,
            'absolute_change': absolute_change,
            'threshold': threshold
        }
    }


def analyze_drift(
    reference_stats: Dict[str, Any],
    current_data: pd.DataFrame,
    predictions: np.ndarray,
    numeric_threshold: float = 0.2,
    categorical_threshold: float = 0.25,
    prediction_threshold: float = 0.1
) -> Dict[str, Any]:
    """
    Comprehensive drift analysis across all features and predictions.
    
    Args:
        reference_stats: Reference statistics from training
        current_data: Current data to analyze
        predictions: Current predictions
        numeric_threshold: Threshold for numeric drift
        categorical_threshold: Threshold for categorical drift (PSI)
        prediction_threshold: Threshold for prediction drift
        
    Returns:
        Dictionary with comprehensive drift report
    """
    report = {
        'overall_drift_detected': False,
        'numeric_features': {},
        'categorical_features': {},
        'prediction_drift': {},
        'summary': {
            'n_features_drifted': 0,
            'drifted_features': []
        }
    }
    
    # Check numeric features
    for feature, ref_stats in reference_stats['numeric'].items():
        if feature in current_data.columns:
            drift_result = detect_numeric_drift(
                ref_stats,
                current_data[feature],
                numeric_threshold
            )
            report['numeric_features'][feature] = drift_result
            
            if drift_result['drift_detected']:
                report['overall_drift_detected'] = True
                report['summary']['n_features_drifted'] += 1
                report['summary']['drifted_features'].append(feature)
    
    # Check categorical features
    for feature, ref_stats in reference_stats['categorical'].items():
        if feature in current_data.columns:
            drift_result = detect_categorical_drift(
                ref_stats,
                current_data[feature],
                categorical_threshold
            )
            report['categorical_features'][feature] = drift_result
            
            if drift_result['drift_detected']:
                report['overall_drift_detected'] = True
                report['summary']['n_features_drifted'] += 1
                report['summary']['drifted_features'].append(feature)
    
    # Check prediction drift
    report['prediction_drift'] = detect_prediction_drift(
        reference_stats['target']['positive_rate'],
        predictions,
        prediction_threshold
    )
    
    if report['prediction_drift']['drift_detected']:
        report['overall_drift_detected'] = True
    
    return report

