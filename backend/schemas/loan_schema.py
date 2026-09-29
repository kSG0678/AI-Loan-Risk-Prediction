"""Validated request and response shapes for loan-risk predictions."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ApplicantRequest(BaseModel):
    """The eleven original applicant features expected by the ML pipeline."""

    # These Literal categories match the values learned by the artifact's
    # OneHotEncoder; restricting them here catches typos before inference.
    Gender: Literal["Female", "Male"]
    Married: Literal["No", "Yes"]
    Dependents: Literal["0", "1", "2", "3+"]
    Education: Literal["Graduate", "Not Graduate"]
    Self_Employed: Literal["No", "Yes"]
    ApplicantIncome: float = Field(ge=0, allow_inf_nan=False)
    CoapplicantIncome: float = Field(ge=0, allow_inf_nan=False)
    LoanAmount: float = Field(ge=0, allow_inf_nan=False)
    Loan_Amount_Term: float = Field(gt=0, allow_inf_nan=False)
    Credit_History: float = Field(allow_inf_nan=False)
    Property_Area: Literal["Rural", "Semiurban", "Urban"]

    # Reject extra keys such as Loan_ID or Loan_Status rather than silently
    # passing identifiers or target labels alongside applicant information.
    model_config = ConfigDict(extra="forbid")

    @field_validator("Credit_History")
    @classmethod
    def validate_credit_history(cls, value: float) -> float:
        """Credit history is a binary feature in the trained dataset."""
        if value not in (0.0, 1.0):
            raise ValueError("Credit_History must be either 0 or 1.")
        return value


class LoanPredictionResponse(BaseModel):
    """The prediction and both class probabilities returned to clients."""

    predicted_class: Literal["Y", "N"]
    predicted_class_label: Literal["Approved", "Rejected"]
    approval_probability: float = Field(ge=0, le=1)
    rejection_probability: float = Field(ge=0, le=1)


class PredictionHistoryRecord(BaseModel):
    """A persisted applicant and its prediction result."""

    application_id: int
    prediction_id: int
    applicant_income: float
    coapplicant_income: float
    loan_amount: float
    loan_amount_term: float
    credit_history: int
    education: str
    property_area: str
    predicted_class: Literal["Y", "N"]
    approval_probability: float = Field(ge=0, le=1)
    rejection_probability: float = Field(ge=0, le=1)
    created_at: datetime
