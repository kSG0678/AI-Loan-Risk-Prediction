"""FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import register_exception_handlers, router
from backend.utils.config import get_allowed_origins

app = FastAPI(
    title="AI Loan Risk Prediction API",
    description="Predict loan approval risk with the project's saved ML pipeline.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(get_allowed_origins()),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

register_exception_handlers(app)
app.include_router(router)
