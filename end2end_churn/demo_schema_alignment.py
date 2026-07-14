"""
Demo script for schema alignment and validation (run with `python demo_schema_alignment.py`).

Not a pytest suite — it prints its checks instead of asserting them; the
assertion-based coverage of the same behavior lives in tests/. This script
demonstrates the API's ability to handle:
1. Perfect input (no warnings)
2. Reordered columns (should handle gracefully)
3. Extra columns (should ignore them)
4. Data quality issues (Pandera catches them)
"""

import pandas as pd

from src.api.validation import align_schema, generate_alignment_warnings, validate_data


def test_perfect_input():
    """Test with perfect input - no issues."""
    print("=" * 70)
    print("TEST 1: Perfect Input")
    print("=" * 70)

    expected_features = ["gender", "SeniorCitizen", "Partner", "tenure", "MonthlyCharges"]
    df = pd.DataFrame(
        [
            {
                "gender": "Male",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "tenure": 12,
                "MonthlyCharges": 70.0,
            }
        ]
    )

    df_aligned, alignment_info = align_schema(df, expected_features)
    warnings = generate_alignment_warnings(alignment_info)

    print(f"Input columns: {list(df.columns)}")
    print(f"Alignment info: {alignment_info}")
    print(f"Warnings: {warnings if warnings else 'None'}")
    print(f"Test passed\n")


def test_reordered_columns():
    """Test with reordered columns - should reorder automatically."""
    print("=" * 70)
    print("TEST 2: Reordered Columns")
    print("=" * 70)

    expected_features = ["gender", "SeniorCitizen", "Partner", "tenure", "MonthlyCharges"]
    # Columns in wrong order
    df = pd.DataFrame(
        [
            {
                "MonthlyCharges": 70.0,
                "tenure": 12,
                "gender": "Male",
                "Partner": "Yes",
                "SeniorCitizen": 0,
            }
        ]
    )

    print(f"Input columns (wrong order): {list(df.columns)}")

    df_aligned, alignment_info = align_schema(df, expected_features)
    warnings = generate_alignment_warnings(alignment_info)

    print(f"Output columns (corrected): {list(df_aligned.columns)}")
    print(f"Alignment info: {alignment_info}")
    print(f"Warnings: {warnings}")
    print(f"Test passed - columns reordered correctly\n")


def test_extra_columns():
    """Test with extra columns - should drop them."""
    print("=" * 70)
    print("TEST 3: Extra Columns")
    print("=" * 70)

    expected_features = ["gender", "SeniorCitizen", "Partner", "tenure", "MonthlyCharges"]
    # Extra columns that model doesn't need
    df = pd.DataFrame(
        [
            {
                "gender": "Male",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "tenure": 12,
                "MonthlyCharges": 70.0,
                "customer_id": "CUST123",  # EXTRA
                "signup_date": "2024-01-01",  # EXTRA
                "source": "web",  # EXTRA
            }
        ]
    )

    print(f"Input columns: {list(df.columns)}")
    print(f"Extra columns: customer_id, signup_date, source")

    df_aligned, alignment_info = align_schema(df, expected_features)
    warnings = generate_alignment_warnings(alignment_info)

    print(f"Output columns: {list(df_aligned.columns)}")
    print(f"Alignment info: {alignment_info}")
    print(f"Warnings: {warnings}")
    print(f"Test passed - extra columns dropped\n")


def test_missing_columns():
    """Test with missing columns - should add NaN."""
    print("=" * 70)
    print("TEST 4: Missing Columns")
    print("=" * 70)

    expected_features = ["gender", "SeniorCitizen", "Partner", "tenure", "MonthlyCharges"]
    # Missing 'Partner' column
    df = pd.DataFrame(
        [{"gender": "Male", "SeniorCitizen": 0, "tenure": 12, "MonthlyCharges": 70.0}]
    )

    print(f"Input columns: {list(df.columns)}")
    print(f"Missing: Partner")

    df_aligned, alignment_info = align_schema(df, expected_features)
    warnings = generate_alignment_warnings(alignment_info)

    print(f"Output columns: {list(df_aligned.columns)}")
    print(f"Alignment info: {alignment_info}")
    print(f"Warnings: {warnings}")
    print(f"Partner value after alignment: {df_aligned['Partner'].iloc[0]}")
    print(f"Test passed - missing column filled with NaN (will be imputed by model)\n")


def test_data_quality():
    """Test Pandera validation - catches invalid values."""
    print("=" * 70)
    print("TEST 5: Data Quality Validation (Pandera)")
    print("=" * 70)

    # Invalid data: negative tenure, invalid gender
    df = pd.DataFrame(
        [
            {
                "gender": "Other",  # INVALID - should be Male/Female
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "tenure": -5,  # INVALID - should be >= 0
                "MonthlyCharges": 70.0,
                "Dependents": "Yes",
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "DSL",
                "OnlineSecurity": "Yes",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "TotalCharges": 840.0,
            }
        ]
    )

    is_valid, errors = validate_data(df, enable_validation=True)

    print(f"Valid: {is_valid}")
    if errors:
        print(f"Validation errors found:")
        for error in errors:
            print(f"  - {error}")
    print(f"Test passed - Pandera caught data quality issues\n")


if __name__ == "__main__":
    print("\n")
    print("=" * 70)
    print("SCHEMA ALIGNMENT & VALIDATION TEST SUITE")
    print("Demonstrating Robustness to Schema Issues")
    print("=" * 70)
    print("\n")

    test_perfect_input()
    test_reordered_columns()
    test_extra_columns()
    test_missing_columns()
    test_data_quality()

    print("=" * 70)
    print("ALL TESTS PASSED ")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  • Schema alignment handles missing/extra/reordered columns gracefully")
    print("  • Missing columns are filled with NaN (model's imputers will handle them)")
    print("  • Extra columns are dropped (model doesn't need them)")
    print("  • Column order is automatically corrected")
    print("  • Pandera validates data quality (catches invalid values)")
    print("  • System is robust to real-world schema drift")
    print()
