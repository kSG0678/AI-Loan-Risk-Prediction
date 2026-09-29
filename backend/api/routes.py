"""HTTP endpoints that connect applicant requests to prediction services."""

import logging

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from backend.database import get_database_session
from backend.models import LoanApplication, Prediction
from backend.schemas.loan_schema import (
    ApplicantRequest,
    LoanPredictionResponse,
    PredictionHistoryRecord,
)
from backend.services.prediction_service import (
    ModelLoadError,
    PredictionError,
    PredictionPersistenceError,
    predict_applicant,
    persist_prediction,
)

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/")
def root() -> dict[str, str]:
    """Tell API clients that the service is running."""
    return {"status": "online", "message": "Loan risk prediction API"}


@router.get("/health")
def health() -> dict[str, str]:
    """Provide a lightweight health check without loading the model."""
    return {"status": "healthy"}


@router.post("/predict", response_model=LoanPredictionResponse)
def predict(
    applicant: ApplicantRequest,
    session: Session = Depends(get_database_session),
) -> LoanPredictionResponse:
    """Validate an applicant, run the saved pipeline, and return its result."""
    # Pydantic has validated the request before this function receives it.
    result = predict_applicant(applicant.model_dump())
    persist_prediction(applicant, result, session)
    return LoanPredictionResponse(**result)


@router.get(
    "/predictions/history",
    response_model=list[PredictionHistoryRecord],
)
def prediction_history(
    session: Session = Depends(get_database_session),
) -> list[PredictionHistoryRecord] | JSONResponse:
    """Return persisted predictions with their applicant details, newest first."""
    statement = (
        select(LoanApplication, Prediction)
        .join(Prediction, Prediction.application_id == LoanApplication.id)
        .order_by(Prediction.created_at.desc(), Prediction.id.desc())
    )
    try:
        records = session.execute(statement).all()
    except SQLAlchemyError:
        logger.error("Prediction history query failed")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Prediction history unavailable",
                "message": "Prediction history could not be retrieved.",
            },
        )

    return [
        PredictionHistoryRecord(
            application_id=application.id,
            prediction_id=prediction.id,
            applicant_income=float(application.applicant_income),
            coapplicant_income=float(application.coapplicant_income),
            loan_amount=float(application.loan_amount),
            loan_amount_term=float(application.loan_amount_term),
            credit_history=application.credit_history,
            education=application.education,
            property_area=application.property_area,
            predicted_class=prediction.predicted_class,
            approval_probability=float(prediction.approval_probability),
            rejection_probability=float(prediction.rejection_probability),
            created_at=prediction.created_at,
        )
        for application, prediction in records
    ]


def register_exception_handlers(app: FastAPI) -> None:
    """Keep predictable client and model errors in a consistent JSON format."""

    @app.exception_handler(RequestValidationError)
    async def invalid_request(
        _request: Request, exception: RequestValidationError
    ) -> JSONResponse:
        # Return useful field-level messages without echoing submitted values or
        # exposing framework internals to clients.
        errors = [
            {
                "field": ".".join(str(part) for part in error["loc"]),
                "message": error["msg"],
            }
            for error in exception.errors()
        ]
        return JSONResponse(
            status_code=422,
            content={"error": "Invalid request data", "details": errors},
        )

    @app.exception_handler(ModelLoadError)
    async def model_unavailable(
        _request: Request, exception: ModelLoadError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={"error": "Prediction model unavailable", "message": str(exception)},
        )

    @app.exception_handler(PredictionError)
    async def prediction_failed(
        _request: Request, exception: PredictionError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={"error": "Prediction failed", "message": str(exception)},
        )

    @app.exception_handler(PredictionPersistenceError)
    async def prediction_persistence_failed(
        _request: Request, exception: PredictionPersistenceError
    ) -> JSONResponse:
        logger.error("Prediction persistence failed")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Prediction persistence failed",
                "message": str(exception),
            },
        )
