"""Persistence model for validated loan application inputs."""

from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    CheckConstraint,
    Index,
    Numeric,
    SmallInteger,
    String,
    TIMESTAMP,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.models.base import Base


class LoanApplication(Base):
    """An applicant's validated features submitted for prediction."""

    __tablename__ = "loan_applications"
    __table_args__ = (
        CheckConstraint("applicant_income >= 0", name="ck_application_income"),
        CheckConstraint("coapplicant_income >= 0", name="ck_coapplicant_income"),
        CheckConstraint("loan_amount >= 0", name="ck_loan_amount"),
        CheckConstraint("loan_amount_term > 0", name="ck_loan_amount_term"),
        CheckConstraint("credit_history IN (0, 1)", name="ck_credit_history"),
        Index("ix_loan_applications_created_at", "created_at"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    gender: Mapped[str] = mapped_column(String(16), nullable=False)
    married: Mapped[str] = mapped_column(String(8), nullable=False)
    dependents: Mapped[str] = mapped_column(String(8), nullable=False)
    education: Mapped[str] = mapped_column(String(16), nullable=False)
    self_employed: Mapped[str] = mapped_column(String(8), nullable=False)
    applicant_income: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    coapplicant_income: Mapped[Decimal] = mapped_column(
        Numeric(12, 2), nullable=False
    )
    loan_amount: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    loan_amount_term: Mapped[Decimal] = mapped_column(Numeric(8, 2), nullable=False)
    credit_history: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    property_area: Mapped[str] = mapped_column(String(16), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP, nullable=False, server_default=func.current_timestamp()
    )

    predictions: Mapped[list["Prediction"]] = relationship(
        back_populates="application"
    )
