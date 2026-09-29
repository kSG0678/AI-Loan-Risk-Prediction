"""Persistence model for loan prediction outcomes."""

from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    CHAR,
    CheckConstraint,
    ForeignKey,
    Index,
    Numeric,
    TIMESTAMP,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.models.base import Base


class Prediction(Base):
    """The classification and probabilities generated for an application."""

    __tablename__ = "predictions"
    __table_args__ = (
        CheckConstraint("predicted_class IN ('Y', 'N')", name="ck_predicted_class"),
        CheckConstraint(
            "approval_probability BETWEEN 0 AND 1",
            name="ck_approval_probability",
        ),
        CheckConstraint(
            "rejection_probability BETWEEN 0 AND 1",
            name="ck_rejection_probability",
        ),
        Index("ix_predictions_created_at", "created_at"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    application_id: Mapped[int] = mapped_column(
        ForeignKey("loan_applications.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    predicted_class: Mapped[str] = mapped_column(CHAR(1), nullable=False)
    approval_probability: Mapped[Decimal] = mapped_column(
        Numeric(7, 6), nullable=False
    )
    rejection_probability: Mapped[Decimal] = mapped_column(
        Numeric(7, 6), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP, nullable=False, server_default=func.current_timestamp()
    )

    application: Mapped["LoanApplication"] = relationship(
        back_populates="predictions"
    )
