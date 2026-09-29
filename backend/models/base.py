"""Shared SQLAlchemy declarative base for persistence models."""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Base metadata shared by all backend database models."""
