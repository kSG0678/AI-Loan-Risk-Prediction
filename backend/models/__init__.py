"""SQLAlchemy models used by backend persistence."""

from backend.models.base import Base
from backend.models.loan_application import LoanApplication
from backend.models.prediction import Prediction

__all__ = ["Base", "LoanApplication", "Prediction"]
