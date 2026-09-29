"""End-to-end HTTP tests for the FastAPI loan prediction endpoints."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest import TestCase
from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from backend.database import get_database_session
from backend.main import app
from backend.models import Base, LoanApplication, Prediction
from backend.services.prediction_service import get_prediction_pipeline

client = TestClient(app)

VALID_APPLICANT = {
    "Gender": "Male",
    "Married": "Yes",
    "Dependents": "1",
    "Education": "Graduate",
    "Self_Employed": "No",
    "ApplicantIncome": 4583,
    "CoapplicantIncome": 1508,
    "LoanAmount": 128,
    "Loan_Amount_Term": 360,
    "Credit_History": 1,
    "Property_Area": "Semiurban",
}


class LoanPredictionApiTests(TestCase):
    def setUp(self) -> None:
        self.engine = create_engine(
            "sqlite+pysqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )

        @event.listens_for(self.engine, "connect")
        def enable_foreign_keys(connection, _record) -> None:
            connection.execute("PRAGMA foreign_keys=ON")

        Base.metadata.create_all(self.engine)
        self.session_factory = sessionmaker(
            bind=self.engine, autoflush=False, expire_on_commit=False
        )

        def test_database_session():
            session = self.session_factory()
            try:
                yield session
            finally:
                session.close()

        app.dependency_overrides[get_database_session] = test_database_session

    def tearDown(self) -> None:
        app.dependency_overrides.pop(get_database_session, None)
        self.engine.dispose()

    def insert_history_record(
        self, created_at: datetime, *, applicant_income: float = 4583
    ) -> tuple[int, int]:
        with Session(self.engine) as session:
            application = LoanApplication(
                gender="Male",
                married="Yes",
                dependents="1",
                education="Graduate",
                self_employed="No",
                applicant_income=Decimal(str(applicant_income)),
                coapplicant_income=Decimal("1508.00"),
                loan_amount=Decimal("128.00"),
                loan_amount_term=Decimal("360.00"),
                credit_history=1,
                property_area="Semiurban",
            )
            application.predictions.append(
                Prediction(
                    predicted_class="Y",
                    approval_probability=Decimal("0.900000"),
                    rejection_probability=Decimal("0.100000"),
                    created_at=created_at,
                )
            )
            session.add(application)
            session.commit()
            return application.id, application.predictions[0].id

    def test_root_reports_api_status(self) -> None:
        response = client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "online")

    def test_health_reports_healthy(self) -> None:
        response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "healthy"})

    def test_predict_returns_real_model_result_and_loads_model_once(self) -> None:
        # Inference must load the saved artifact only; training and data
        # preparation functions are patched to fail if an API call invokes them.
        get_prediction_pipeline.cache_clear()
        with (
            patch(
                "ml.src.pipeline.build_inference_pipeline",
                side_effect=AssertionError("API requests must not train a model."),
            ),
            patch(
                "ml.src.pipeline.prepare_data",
                side_effect=AssertionError("API requests must not read training data."),
            ),
        ):
            first_response = client.post("/predict", json=VALID_APPLICANT)
            second_response = client.post("/predict", json=VALID_APPLICANT)

        self.assertEqual(first_response.status_code, 200)
        self.assertEqual(second_response.status_code, 200)
        result = first_response.json()
        self.assertIn(result["predicted_class"], {"Y", "N"})
        self.assertEqual(
            result["predicted_class_label"],
            {"Y": "Approved", "N": "Rejected"}[result["predicted_class"]],
        )
        self.assertEqual(
            set(result),
            {
                "predicted_class",
                "predicted_class_label",
                "approval_probability",
                "rejection_probability",
            },
        )
        self.assertAlmostEqual(
            result["approval_probability"] + result["rejection_probability"], 1.0
        )
        self.assertEqual(get_prediction_pipeline.cache_info().misses, 1)
        self.assertEqual(get_prediction_pipeline.cache_info().hits, 1)

    def test_successful_prediction_saves_application_and_linked_prediction(self) -> None:
        prediction_result = {
            "predicted_class": "Y",
            "predicted_class_label": "Approved",
            "approval_probability": 0.9,
            "rejection_probability": 0.1,
        }
        with patch(
            "backend.api.routes.predict_applicant", return_value=prediction_result
        ):
            response = client.post("/predict", json=VALID_APPLICANT)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), prediction_result)
        with Session(self.engine) as session:
            applications = session.scalars(select(LoanApplication)).all()
            predictions = session.scalars(select(Prediction)).all()

        self.assertEqual(len(applications), 1)
        self.assertEqual(len(predictions), 1)
        self.assertEqual(applications[0].applicant_income, 4583.0)
        self.assertEqual(predictions[0].application_id, applications[0].id)
        self.assertEqual(predictions[0].predicted_class, "Y")
        self.assertEqual(float(predictions[0].approval_probability), 0.9)
        self.assertEqual(float(predictions[0].rejection_probability), 0.1)

    def test_prediction_history_returns_records_newest_first(self) -> None:
        newer_application_id, newer_prediction_id = self.insert_history_record(
            datetime(2025, 1, 2, tzinfo=timezone.utc), applicant_income=4583
        )
        older_application_id, older_prediction_id = self.insert_history_record(
            datetime(2025, 1, 1, tzinfo=timezone.utc), applicant_income=3000
        )

        response = client.get("/predictions/history")

        self.assertEqual(response.status_code, 200)
        records = response.json()
        self.assertEqual(len(records), 2)
        self.assertEqual(
            set(records[0]),
            {
                "application_id",
                "prediction_id",
                "applicant_income",
                "coapplicant_income",
                "loan_amount",
                "loan_amount_term",
                "credit_history",
                "education",
                "property_area",
                "predicted_class",
                "approval_probability",
                "rejection_probability",
                "created_at",
            },
        )
        self.assertEqual(records[0]["application_id"], newer_application_id)
        self.assertEqual(records[0]["prediction_id"], newer_prediction_id)
        self.assertEqual(records[0]["applicant_income"], 4583)
        self.assertEqual(records[0]["coapplicant_income"], 1508)
        self.assertEqual(records[0]["loan_amount"], 128)
        self.assertEqual(records[0]["loan_amount_term"], 360)
        self.assertEqual(records[0]["credit_history"], 1)
        self.assertEqual(records[0]["education"], "Graduate")
        self.assertEqual(records[0]["property_area"], "Semiurban")
        self.assertEqual(records[0]["predicted_class"], "Y")
        self.assertEqual(records[0]["approval_probability"], 0.9)
        self.assertEqual(records[0]["rejection_probability"], 0.1)
        self.assertEqual(records[1]["application_id"], older_application_id)
        self.assertEqual(records[1]["prediction_id"], older_prediction_id)

    def test_prediction_history_returns_empty_list_when_no_records_exist(self) -> None:
        with patch(
            "backend.api.routes.predict_applicant",
            side_effect=AssertionError("History reads must not run inference."),
        ):
            response = client.get("/predictions/history")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), [])

    def test_prediction_history_database_failure_returns_safe_error(self) -> None:
        def fail_history_query(_connection, _cursor, statement, _parameters, _context, _executemany):
            if "FROM loan_applications" in statement:
                raise SQLAlchemyError("private database details")

        event.listen(self.engine, "before_cursor_execute", fail_history_query)
        try:
            response = client.get("/predictions/history")
        finally:
            event.remove(
                self.engine, "before_cursor_execute", fail_history_query
            )

        self.assertEqual(response.status_code, 500)
        self.assertEqual(
            response.json(),
            {
                "error": "Prediction history unavailable",
                "message": "Prediction history could not be retrieved.",
            },
        )
        self.assertNotIn("private database details", response.text)

    def test_persistence_failure_rolls_back_and_returns_safe_error(self) -> None:
        invalid_prediction = {
            "predicted_class": "INVALID",
            "predicted_class_label": "Approved",
            "approval_probability": 0.9,
            "rejection_probability": 0.1,
        }
        with patch(
            "backend.api.routes.predict_applicant", return_value=invalid_prediction
        ):
            response = client.post("/predict", json=VALID_APPLICANT)

        self.assertEqual(response.status_code, 500)
        self.assertEqual(
            response.json(),
            {
                "error": "Prediction persistence failed",
                "message": "The prediction could not be saved.",
            },
        )
        self.assertNotIn("INVALID", response.text)
        self.assertNotIn("INSERT", response.text)
        with Session(self.engine) as session:
            application_count = session.scalar(
                select(func.count()).select_from(LoanApplication)
            )
            prediction_count = session.scalar(
                select(func.count()).select_from(Prediction)
            )

        self.assertEqual(application_count, 0)
        self.assertEqual(prediction_count, 0)

    def test_negative_income_is_rejected(self) -> None:
        response = client.post(
            "/predict", json={**VALID_APPLICANT, "ApplicantIncome": -1}
        )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"], "Invalid request data")

    def test_missing_required_field_is_rejected(self) -> None:
        applicant = dict(VALID_APPLICANT)
        del applicant["Property_Area"]

        response = client.post("/predict", json=applicant)

        self.assertEqual(response.status_code, 422)
        self.assertTrue(
            any(
                "Property_Area" in error["field"]
                for error in response.json()["details"]
            )
        )

    def test_invalid_categorical_value_is_rejected(self) -> None:
        response = client.post(
            "/predict", json={**VALID_APPLICANT, "Property_Area": "Suburban"}
        )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"], "Invalid request data")

    def test_model_loading_error_returns_safe_response(self) -> None:
        get_prediction_pipeline.cache_clear()
        with patch(
            "backend.services.prediction_service.load_inference_pipeline",
            side_effect=FileNotFoundError("private artifact path"),
        ):
            response = client.post("/predict", json=VALID_APPLICANT)

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"], "Prediction model unavailable")
        self.assertNotIn("private artifact path", response.text)

    def test_prediction_error_returns_safe_response(self) -> None:
        get_prediction_pipeline.cache_clear()
        with patch(
            "backend.services.prediction_service.predict_with_pipeline",
            side_effect=RuntimeError("private internal details"),
        ):
            response = client.post("/predict", json=VALID_APPLICANT)

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["error"], "Prediction failed")
        self.assertNotIn("private internal details", response.text)


if __name__ == "__main__":
    import unittest

    unittest.main()
