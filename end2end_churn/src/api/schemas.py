"""Pydantic schemas for API request/response validation."""

from typing import Literal

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """
    Request schema for churn prediction.

    All fields match the features expected by the trained model.
    """

    # Demographics
    gender: Literal["Male", "Female"] = Field(..., description="Customer gender")
    SeniorCitizen: Literal[0, 1] = Field(
        ..., description="Whether customer is senior (0=No, 1=Yes)"
    )
    Partner: Literal["Yes", "No"] = Field(..., description="Whether customer has a partner")
    Dependents: Literal["Yes", "No"] = Field(..., description="Whether customer has dependents")

    # Account info
    tenure: int = Field(..., ge=0, description="Number of months as customer")
    Contract: Literal["Month-to-month", "One year", "Two year"] = Field(
        ..., description="Contract type"
    )
    PaperlessBilling: Literal["Yes", "No"] = Field(
        ..., description="Whether customer has paperless billing"
    )
    PaymentMethod: Literal[
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ] = Field(..., description="Payment method")
    MonthlyCharges: float = Field(..., gt=0, description="Monthly charges in dollars")
    # No cross-field cap against tenure * MonthlyCharges: MonthlyCharges is the
    # CURRENT price while TotalCharges is a historical accumulation, so any
    # customer whose price ever decreased would violate such a cap (and tenure=0
    # would make the cap 0). Range plausibility is handled as soft warnings by
    # the Pandera layer instead of hard 422s.
    TotalCharges: float = Field(..., ge=0, description="Total charges to date in dollars")

    # Services - Phone
    PhoneService: Literal["Yes", "No"] = Field(
        ..., description="Whether customer has phone service"
    )
    MultipleLines: Literal["Yes", "No", "No phone service"] = Field(
        ..., description="Whether customer has multiple lines"
    )

    # Services - Internet
    InternetService: Literal["DSL", "Fiber optic", "No"] = Field(
        ..., description="Internet service type"
    )
    OnlineSecurity: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Whether customer has online security"
    )
    OnlineBackup: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Whether customer has online backup"
    )
    DeviceProtection: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Whether customer has device protection"
    )
    TechSupport: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Whether customer has tech support"
    )
    StreamingTV: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Whether customer has streaming TV"
    )
    StreamingMovies: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Whether customer has streaming movies"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "gender": "Female",
                    "SeniorCitizen": 0,
                    "Partner": "Yes",
                    "Dependents": "No",
                    "tenure": 12,
                    "Contract": "Month-to-month",
                    "PaperlessBilling": "Yes",
                    "PaymentMethod": "Electronic check",
                    "MonthlyCharges": 70.35,
                    "TotalCharges": 844.20,
                    "PhoneService": "Yes",
                    "MultipleLines": "No",
                    "InternetService": "Fiber optic",
                    "OnlineSecurity": "No",
                    "OnlineBackup": "Yes",
                    "DeviceProtection": "No",
                    "TechSupport": "No",
                    "StreamingTV": "Yes",
                    "StreamingMovies": "No",
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Response schema for churn prediction with request tracing support."""

    churn_probability: float = Field(
        ..., ge=0.0, le=1.0, description="Probability of customer churning (0-1)"
    )
    churn_prediction: Literal["Yes", "No"] = Field(
        ..., description="Binary prediction (Yes=churn, No=stay)"
    )
    risk_level: Literal["Low", "Medium", "High"] = Field(
        ..., description="Risk level based on probability"
    )
    model_version: str = Field(..., description="Model version used for prediction")
    warnings: list[str] | None = Field(None, description="Schema alignment or validation warnings")
    request_id: str = Field(..., description="Unique request ID for tracing logs")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "churn_probability": 0.73,
                    "churn_prediction": "Yes",
                    "risk_level": "High",
                    "model_version": "20251024_183147",
                    "warnings": None,
                    "request_id": "abc-123-def-456",
                }
            ]
        }
    }
