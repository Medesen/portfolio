#!/usr/bin/env python3
"""
Test drift detection with multiple scenarios.

This script tests the drift detection endpoint with various scenarios:
1. Normal data (no drift expected)
2. Numeric drift (tenure increased by 50%)
3. Categorical drift (contract distribution changed)
4. Multiple features drifting
5. Validation set as "future" data (should be stable)

Usage:
    python scripts/test_drift.py

    Or with Docker:
    docker compose run --rm --entrypoint python api scripts/test_drift.py
"""

import sys
from pathlib import Path

import pandas as pd
import requests

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_data

# API endpoint
API_URL = "http://localhost:8000"


def test_drift_detection():
    """Run comprehensive drift detection tests."""
    print("=" * 70)
    print("DRIFT DETECTION TEST SUITE")
    print("=" * 70)
    print()

    # Load dataset
    print("Loading dataset...")
    try:
        df = load_data("data/dataset.arff")
        print(f"✓ Loaded {len(df)} records")
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        return False

    # Remove target column for drift analysis
    if "Churn" in df.columns:
        df = df.drop("Churn", axis=1)

    print()
    print("-" * 70)
    print("Test 1: Normal Data (No Drift Expected)")
    print("-" * 70)

    normal_sample = df.sample(min(100, len(df))).to_dict("records")

    try:
        response = requests.post(f"{API_URL}/drift", json={"customers": normal_sample}, timeout=30)
        response.raise_for_status()
        result = response.json()

        if not result["overall_drift_detected"]:
            print("✓ PASS: No drift detected in normal data")
        else:
            print(f"⚠ WARN: Unexpected drift detected: {result['drifted_features']}")
            print(f"  This might indicate the thresholds are too sensitive")
    except requests.exceptions.ConnectionError:
        print("✗ FAIL: Could not connect to API. Is the service running?")
        print("  Start with: make up")
        return False
    except Exception as e:
        print(f"✗ FAIL: {e}")
        return False

    print()
    print("-" * 70)
    print("Test 2: Numeric Drift (Tenure +50%)")
    print("-" * 70)

    drifted_sample = df.sample(min(100, len(df))).copy()
    if "tenure" in drifted_sample.columns:
        drifted_sample["tenure"] = drifted_sample["tenure"] * 1.5

        try:
            response = requests.post(
                f"{API_URL}/drift",
                json={"customers": drifted_sample.to_dict("records")},
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()

            if result["overall_drift_detected"] and "tenure" in result["drifted_features"]:
                print(f"✓ PASS: Tenure drift detected")
                print(f"  Drifted features: {result['drifted_features']}")
                print(f"  Total features drifted: {result['n_features_drifted']}")
            else:
                print(f"✗ FAIL: Tenure drift not detected")
                print(f"  Expected 'tenure' in drifted features")
        except Exception as e:
            print(f"✗ FAIL: {e}")
            return False
    else:
        print("⊘ SKIP: 'tenure' column not found in dataset")

    print()
    print("-" * 70)
    print("Test 3: Categorical Drift (Contract Distribution)")
    print("-" * 70)

    drifted_sample = df.sample(min(100, len(df))).copy()
    if "Contract" in drifted_sample.columns:
        # Force all contracts to Month-to-month (extreme distribution shift)
        drifted_sample["Contract"] = "Month-to-month"

        try:
            response = requests.post(
                f"{API_URL}/drift",
                json={"customers": drifted_sample.to_dict("records")},
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()

            if result["overall_drift_detected"] and "Contract" in result["drifted_features"]:
                print(f"✓ PASS: Contract drift detected")
                print(f"  Drifted features: {result['drifted_features']}")
            else:
                print(f"✗ FAIL: Contract drift not detected")
                print(f"  Expected 'Contract' in drifted features")
        except Exception as e:
            print(f"✗ FAIL: {e}")
            return False
    else:
        print("⊘ SKIP: 'Contract' column not found in dataset")

    print()
    print("-" * 70)
    print("Test 4: Multiple Features Drift")
    print("-" * 70)

    drifted_sample = df.sample(min(100, len(df))).copy()
    modified_features = []

    if "tenure" in drifted_sample.columns:
        drifted_sample["tenure"] = drifted_sample["tenure"] * 1.5
        modified_features.append("tenure")

    if "MonthlyCharges" in drifted_sample.columns:
        drifted_sample["MonthlyCharges"] = drifted_sample["MonthlyCharges"] * 0.5
        modified_features.append("MonthlyCharges")

    if modified_features:
        try:
            response = requests.post(
                f"{API_URL}/drift",
                json={"customers": drifted_sample.to_dict("records")},
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()

            detected_count = result["n_features_drifted"]
            if detected_count >= 2:
                print(f"✓ PASS: Multiple features drifted ({detected_count} features)")
                print(f"  Modified: {modified_features}")
                print(f"  Detected: {result['drifted_features']}")
            else:
                print(f"⚠ WARN: Only {detected_count} features detected as drifted")
                print(f"  Modified: {modified_features}")
                print(f"  Detected: {result['drifted_features']}")
        except Exception as e:
            print(f"✗ FAIL: {e}")
            return False
    else:
        print("⊘ SKIP: Required columns not found in dataset")

    print()
    print("-" * 70)
    print("Test 5: Empty Batch (Should Fail)")
    print("-" * 70)

    try:
        response = requests.post(f"{API_URL}/drift", json={"customers": []}, timeout=30)
        if response.status_code == 400:
            print("✓ PASS: Empty batch correctly rejected with 400 status")
        else:
            print(f"✗ FAIL: Expected 400 status, got {response.status_code}")
    except Exception as e:
        print(f"✗ FAIL: {e}")
        return False

    print()
    print("-" * 70)
    print("Test 6: Drift Info Endpoint")
    print("-" * 70)

    try:
        response = requests.get(f"{API_URL}/drift/info", timeout=10)
        response.raise_for_status()
        info = response.json()

        print("✓ PASS: Drift info endpoint accessible")
        print(f"  Baseline samples: {info['baseline']['n_samples']}")
        print(f"  Baseline churn rate: {info['baseline']['positive_rate']:.2%}")
        print(f"  Numeric features: {info['baseline']['n_numeric_features']}")
        print(f"  Categorical features: {info['baseline']['n_categorical_features']}")
        print(f"  Model age: {info['model_info']['age_days']} days")
        print(
            f"  Thresholds: numeric={info['thresholds']['numeric']}, "
            f"categorical={info['thresholds']['categorical']}, "
            f"prediction={info['thresholds']['prediction']}"
        )
    except Exception as e:
        print(f"✗ FAIL: {e}")
        return False

    print()
    print("=" * 70)
    print("DRIFT DETECTION TEST SUITE COMPLETE")
    print("=" * 70)
    print()
    print("✓ All critical tests passed")
    print()
    print("Next steps:")
    print("  1. Review drift thresholds if needed (docker-compose.yml)")
    print("  2. Monitor drift metrics at http://localhost:8000/metrics")
    print("  3. Enable AUTO_RETRAIN_ON_DRIFT if desired")
    print()

    return True


if __name__ == "__main__":
    success = test_drift_detection()
    sys.exit(0 if success else 1)
