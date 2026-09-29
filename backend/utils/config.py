"""Filesystem paths and environment-based CORS settings for the backend."""

import os
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import urlsplit

# Resolve paths from this source file rather than the process working directory,
# so the artifact is found whether Uvicorn starts at the project root or elsewhere.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "ml" / "models" / "loan_risk_pipeline.joblib"

DEVELOPMENT_ORIGINS = (
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
)


def get_allowed_origins(
    environ: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """Read an explicit comma-separated CORS allowlist or use local defaults."""
    settings = os.environ if environ is None else environ
    configured_origins = settings.get("CORS_ORIGINS")
    if configured_origins is None:
        return DEVELOPMENT_ORIGINS

    origins = tuple(
        origin.strip().rstrip("/") for origin in configured_origins.split(",")
    )
    if not origins or any(not origin for origin in origins):
        raise ValueError("CORS_ORIGINS must contain one or more valid origins.")

    for origin in origins:
        parsed = urlsplit(origin)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.path
            or parsed.query
            or parsed.fragment
            or parsed.username
            or parsed.password
            or "*" in origin
        ):
            raise ValueError("CORS_ORIGINS contains an invalid origin.")

    return origins
