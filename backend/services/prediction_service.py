"""Load the saved ML pipeline once and use it for applicant inference."""

from __future__ import annotations

from functools import lru_cache
import logging
from typing import Any, Mapping

from sklearn.pipeline import Pipeline
from sqlalchemy.orm import Session

from backend.models import LoanApplication, Prediction
from backend.schemas.loan_schema import ApplicantRequest
from backend.utils.config import MODEL_PATH
from ml.src.pipeline import load_inference_pipeline
from ml.src.predict import predict_with_pipeline

logger = logging.getLogger(__name__)


class ModelLoadError(RuntimeError):
    """Raised when the saved inference artifact cannot be loaded."""


class PredictionError(RuntimeError):
    """Raised when the loaded pipeline cannot produce a prediction."""


class PredictionPersistenceError(RuntimeError):
    """Raised when an inferred prediction cannot be saved atomically."""


@lru_cache(maxsize=1)
def get_prediction_pipeline() -> Pipeline:
    """Return the in-memory model, loading the existing artifact on first use.

    Keeping this result cached avoids repeatedly reading and deserializing the
    artifact for every HTTP request; the same fitted transformers and classifier
    are reused, and no training data or training code is run.
    """
    try:
        return load_inference_pipeline(MODEL_PATH)
    except Exception as exc:
        # Log the underlying details for the operator, but expose only a safe
        # service error to API clients through the registered exception handler.
        logger.exception("Could not load prediction artifact from %s", MODEL_PATH)
        raise ModelLoadError("The saved prediction model could not be loaded.") from exc


def predict_applicant(record: Mapping[str, Any]) -> dict[str, Any]:
    """Run the shared ML prediction logic against validated applicant fields."""
    try:
        return predict_with_pipeline(record, get_prediction_pipeline())
    except ModelLoadError:
        raise
    except Exception as exc:
        logger.exception("Prediction failed for a validated applicant")
        raise PredictionError("The applicant prediction could not be completed.") from exc


def persist_prediction(
    applicant: ApplicantRequest,
    result: Mapping[str, Any],
    session: Session,
) -> None:
    """Persist the validated input and its prediction in one transaction."""
    try:
        values = applicant.model_dump()
        application = LoanApplication(
            gender=values["Gender"],
            married=values["Married"],
            dependents=values["Dependents"],
            education=values["Education"],
            self_employed=values["Self_Employed"],
            applicant_income=values["ApplicantIncome"],
            coapplicant_income=values["CoapplicantIncome"],
            loan_amount=values["LoanAmount"],
            loan_amount_term=values["Loan_Amount_Term"],
            credit_history=int(values["Credit_History"]),
            property_area=values["Property_Area"],
        )
        prediction = Prediction(
            predicted_class=result["predicted_class"],
            approval_probability=result["approval_probability"],
            rejection_probability=result["rejection_probability"],
        )
        application.predictions.append(prediction)

        with session.begin():
            session.add(application)
    except Exception as exc:
        logger.exception("Could not persist applicant prediction")
        raise PredictionPersistenceError(
            "The prediction could not be saved."
        ) from exc
