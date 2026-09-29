"""FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import register_exception_handlers, router
from backend.utils.config import DEVELOPMENT_ORIGINS

app = FastAPI(
    title="AI Loan Risk Prediction API",
    description="Predict loan approval risk with the project's saved ML pipeline.",
    version="1.0.0",
)

# Browsers enforce CORS when a frontend and API use different local ports.
# Restrict development access to likely local frontend origins instead of "*".
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(DEVELOPMENT_ORIGINS),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

register_exception_handlers(app)
app.include_router(router)
