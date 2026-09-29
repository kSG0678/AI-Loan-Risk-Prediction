"""Environment-based SQLAlchemy configuration for MySQL."""

import os
from collections.abc import Iterator, Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.engine import URL
from sqlalchemy.orm import Session, sessionmaker


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def get_database_url(
    environ: Mapping[str, str] | None = None,
) -> URL:
    """Build a MySQL URL from environment variables without exposing secrets."""
    settings = os.environ if environ is None else environ
    required = ("DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD")
    missing = [name for name in required if not settings.get(name)]
    if missing:
        raise ValueError(
            "Missing required database environment variables: "
            + ", ".join(missing)
        )

    try:
        port = int(settings["DB_PORT"])
    except ValueError as exc:
        raise ValueError("DB_PORT must be an integer.") from exc
    if not 1 <= port <= 65535:
        raise ValueError("DB_PORT must be between 1 and 65535.")

    return URL.create(
        drivername="mysql+pymysql",
        username=settings["DB_USER"],
        password=settings["DB_PASSWORD"],
        host=settings["DB_HOST"],
        port=port,
        database=settings["DB_NAME"],
        query={"charset": "utf8mb4"},
    )


def create_database_engine(
    environ: Mapping[str, str] | None = None,
    **engine_options: Any,
) -> Engine:
    """Create a SQLAlchemy engine; connecting remains lazy until first use."""
    return create_engine(get_database_url(environ), pool_pre_ping=True, **engine_options)


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Create a session factory bound to the provided engine."""
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


@lru_cache(maxsize=1)
def get_database_engine() -> Engine:
    """Return the shared lazily-connected engine for the configured database."""
    return create_database_engine()


@lru_cache(maxsize=1)
def get_database_session_factory() -> sessionmaker[Session]:
    """Return the shared session factory, creating the engine only once."""
    return create_session_factory(get_database_engine())


def get_database_session() -> Iterator[Session]:
    """Yield a request-scoped session from the shared factory."""
    session = get_database_session_factory()()
    try:
        yield session
    finally:
        session.close()


def check_database_connection(engine: Engine) -> None:
    """Raise the database driver's error if a basic connection check fails."""
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))
