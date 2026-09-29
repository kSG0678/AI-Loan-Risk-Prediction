"""Filesystem paths and development CORS settings for the backend."""

from pathlib import Path

# Resolve paths from this source file rather than the process working directory,
# so the artifact is found whether Uvicorn starts at the project root or elsewhere.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "ml" / "models" / "loan_risk_pipeline.joblib"

# Allow only common local React development servers; production origins can be
# added deliberately when a frontend is introduced in a later phase.
DEVELOPMENT_ORIGINS = (
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
)
