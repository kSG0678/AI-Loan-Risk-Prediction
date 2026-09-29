"""Reusable single-applicant prediction from the saved inference pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

if __package__:
    from .data_preprocessing import TARGET_COLUMN
    from .pipeline import MODEL_PATH, load_inference_pipeline
else:
    from data_preprocessing import TARGET_COLUMN
    from pipeline import MODEL_PATH, load_inference_pipeline


EXPECTED_INPUT_FIELDS = (
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "Property_Area",
)
FORBIDDEN_INPUT_FIELDS = frozenset({"Loan_ID", TARGET_COLUMN})
CLASS_LABELS = {"Y": "Approved", "N": "Rejected"}


def _validate_record(record: Mapping[str, Any]) -> pd.DataFrame:
    if not isinstance(record, Mapping):
        raise TypeError("Applicant record must be a mapping of input fields to values.")

    forbidden = FORBIDDEN_INPUT_FIELDS.intersection(record)
    if forbidden:
        raise ValueError(
            f"Applicant input must not include target or identifier fields: "
            f"{sorted(forbidden)}"
        )
    missing = [field for field in EXPECTED_INPUT_FIELDS if field not in record]
    if missing:
        raise ValueError(f"Applicant record is missing required fields: {missing}")
    unexpected = sorted(set(record) - set(EXPECTED_INPUT_FIELDS))
    if unexpected:
        raise ValueError(f"Applicant record contains unexpected fields: {unexpected}")

    # Keep dataset field ordering and numeric dtypes consistent with training.
    frame = pd.DataFrame(
        [{field: record[field] for field in EXPECTED_INPUT_FIELDS}],
        columns=EXPECTED_INPUT_FIELDS,
    )
    numeric_fields = (
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "Loan_Amount_Term",
        "Credit_History",
    )
    for field in numeric_fields:
        frame[field] = pd.to_numeric(frame[field], errors="raise")
        value = frame[field].iloc[0]
        if pd.notna(value) and not np.isfinite(value):
            raise ValueError(f"{field} must be finite or missing.")
    return frame


def predict_with_pipeline(
    record: Mapping[str, Any], pipeline: Pipeline
) -> dict[str, Any]:
    """Predict one applicant using an already-loaded complete inference pipeline."""
    applicant = _validate_record(record)
    if not isinstance(pipeline, Pipeline):
        raise TypeError("pipeline must be a loaded scikit-learn Pipeline.")

    predicted_class = str(pipeline.predict(applicant)[0])
    if predicted_class not in CLASS_LABELS:
        raise ValueError(f"Pipeline returned an unknown class: {predicted_class!r}")

    classifier = pipeline.named_steps["classifier"]
    probabilities = pipeline.predict_proba(applicant)[0]
    class_indices = {str(label): index for index, label in enumerate(classifier.classes_)}
    if set(class_indices) != set(CLASS_LABELS):
        raise ValueError(
            f"Pipeline classes must be {sorted(CLASS_LABELS)}, "
            f"got {sorted(class_indices)}."
        )

    approval_probability = float(probabilities[class_indices["Y"]])
    rejection_probability = float(probabilities[class_indices["N"]])
    if not np.isclose(approval_probability + rejection_probability, 1.0):
        raise RuntimeError("Predicted class probabilities do not sum to 1.")

    return {
        "predicted_class": predicted_class,
        "predicted_class_label": CLASS_LABELS[predicted_class],
        "approval_probability": approval_probability,
        "rejection_probability": rejection_probability,
    }


def predict_applicant(
    record: Mapping[str, Any], artifact_path: str | Path = MODEL_PATH
) -> dict[str, Any]:
    """Load the saved artifact and predict one applicant without retraining."""
    pipeline = load_inference_pipeline(artifact_path)
    return predict_with_pipeline(record, pipeline)
