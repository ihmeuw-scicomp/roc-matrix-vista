from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
import logging 

from backend.config import settings
from backend.api.routes import roc_routes
from backend.db import engine, Base, get_db
from backend.utils.logging_config import setup_logging
from backend.services.roc_analysis_service import (
    create_roc_analysis, 
    process_dataframe,
    find_column_by_suffix
)
from backend.models.roc_data import ROCAnalysis, ConfusionMatrix
import pandas as pd

setup_logging(log_level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="API for ROC curve analysis and visualization"
)

# Set up CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,  # Focus on only the problematic origin for now
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "Accept"],
)

@app.on_event("startup")
def startup():
    Base.metadata.create_all(bind=engine)
    

# Include routers
app.include_router(roc_routes.router, prefix=settings.API_V1_STR)

@app.get("/")
def root():
    return {
        "message": f"Welcome to {settings.PROJECT_NAME} API",
        "version": settings.VERSION,
        "docs_url": "/docs",
        "redoc_url": "/redoc"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True  # Enable auto-reload during development
    )