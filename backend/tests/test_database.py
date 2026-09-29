"""Focused checks for the database configuration and persistence models."""

from decimal import Decimal
import secrets
from unittest import TestCase

from sqlalchemy import create_engine, event, inspect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.database.config import (
    check_database_connection,
    create_database_engine,
    get_database_url,
)
from backend.models import Base, LoanApplication, Prediction


VALID_DB_ENV = {
    "DB_HOST": "localhost",
    "DB_PORT": "3306",
    "DB_NAME": "loan_risk_db",
    "DB_USER": "test_user",
    "DB_PASSWORD": secrets.token_urlsafe(24),
}


class DatabaseConfigurationTests(TestCase):
    def test_url_uses_environment_values_and_masks_password(self) -> None:
        url = get_database_url(VALID_DB_ENV)

        self.assertEqual(url.drivername, "mysql+pymysql")
        self.assertEqual(url.host, "localhost")
        self.assertEqual(url.port, 3306)
        self.assertEqual(url.database, "loan_risk_db")
        self.assertEqual(url.username, "test_user")
        self.assertEqual(url.password, VALID_DB_ENV["DB_PASSWORD"])
        self.assertNotIn(VALID_DB_ENV["DB_PASSWORD"], url.render_as_string())

    def test_missing_and_invalid_environment_values_are_reported(self) -> None:
        with self.assertRaisesRegex(ValueError, "DB_PASSWORD"):
            get_database_url(
                {
                    key: value
                    for key, value in VALID_DB_ENV.items()
                    if key != "DB_PASSWORD"
                }
            )

        invalid_port = {**VALID_DB_ENV, "DB_PORT": "not-a-port"}
        with self.assertRaisesRegex(ValueError, "DB_PORT must be an integer"):
            get_database_url(invalid_port)

    def test_engine_creation_is_lazy(self) -> None:
        engine = create_database_engine(VALID_DB_ENV)
        try:
            self.assertEqual(engine.url.drivername, "mysql+pymysql")
        finally:
            engine.dispose()


class DatabaseModelTests(TestCase):
    def setUp(self) -> None:
        self.engine = create_engine("sqlite+pysqlite:///:memory:")

        @event.listens_for(self.engine, "connect")
        def enable_foreign_keys(connection, _record) -> None:
            connection.execute("PRAGMA foreign_keys=ON")

        Base.metadata.create_all(self.engine)

    def tearDown(self) -> None:
        self.engine.dispose()

    def test_tables_columns_constraints_and_indexes_are_declared(self) -> None:
        inspector = inspect(self.engine)

        self.assertEqual(
            set(inspector.get_table_names()),
            {"loan_applications", "predictions"},
        )
        application_columns = {
            column["name"]: column
            for column in inspector.get_columns("loan_applications")
        }
        self.assertEqual(
            set(application_columns),
            {
                "id",
                "gender",
                "married",
                "dependents",
                "education",
                "self_employed",
                "applicant_income",
                "coapplicant_income",
                "loan_amount",
                "loan_amount_term",
                "credit_history",
                "property_area",
                "created_at",
            },
        )
        self.assertTrue(
            all(
                application_columns[name]["nullable"] is False
                for name in application_columns
            )
        )
        self.assertEqual(
            inspector.get_foreign_keys("predictions")[0]["referred_table"],
            "loan_applications",
        )
        self.assertIn(
            "ix_predictions_application_id",
            {index["name"] for index in inspector.get_indexes("predictions")},
        )

    def test_application_prediction_relationship_and_foreign_key(self) -> None:
        with Session(self.engine) as session:
            application = LoanApplication(
                gender="Male",
                married="Yes",
                dependents="1",
                education="Graduate",
                self_employed="No",
                applicant_income=Decimal("4583.00"),
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
                )
            )
            session.add(application)
            session.commit()

            self.assertIsNotNone(application.created_at)
            self.assertEqual(application.predictions[0].application_id, application.id)

    def test_connection_check_executes_select_one(self) -> None:
        check_database_connection(self.engine)

    def test_prediction_requires_an_existing_application(self) -> None:
        with Session(self.engine) as session:
            session.add(
                Prediction(
                    application_id=999,
                    predicted_class="Y",
                    approval_probability=Decimal("0.900000"),
                    rejection_probability=Decimal("0.100000"),
                )
            )
            with self.assertRaises(IntegrityError):
                session.commit()

    def test_class_and_probability_constraints_reject_invalid_values(self) -> None:
        with Session(self.engine) as session:
            application = LoanApplication(
                gender="Male",
                married="Yes",
                dependents="1",
                education="Graduate",
                self_employed="No",
                applicant_income=Decimal("4583.00"),
                coapplicant_income=Decimal("1508.00"),
                loan_amount=Decimal("128.00"),
                loan_amount_term=Decimal("360.00"),
                credit_history=1,
                property_area="Semiurban",
            )
            application.predictions.append(
                Prediction(
                    predicted_class="X",
                    approval_probability=Decimal("1.200000"),
                    rejection_probability=Decimal("-0.200000"),
                )
            )
            session.add(application)
            with self.assertRaises(IntegrityError):
                session.commit()


if __name__ == "__main__":
    import unittest

    unittest.main()
