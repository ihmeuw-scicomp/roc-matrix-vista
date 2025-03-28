from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from backend.config import settings
from backend.api.routes import roc_routes
from backend.db import engine, Base, get_db
from backend.services.roc_analysis_service import (
    create_roc_analysis, 
    process_dataframe,
    find_column_by_suffix
)
from backend.models.roc_data import ROCAnalysis, ConfusionMatrix
import pandas as pd

def load_test_data():
    db = next(get_db())
    # Check if data is already loaded to avoid duplicates
    if db.query(ROCAnalysis).first():
        return
    # Load CSV from disk
    df = pd.read_csv("../data/test_data.csv")
    # Process the dataframe using the existing function
    processed_df = process_dataframe(df)
    
    # Find the adjusted confidence column dynamically by suffix
    adjusted_conf_column = find_column_by_suffix(processed_df, '_confidence_adjusted')
    
    if not adjusted_conf_column:
        raise ValueError("Could not find a column with suffix '_confidence_adjusted' in the processed dataframe")
    
    # Extract true labels and predicted probabilities from the processed dataframe
    true_labels = processed_df['Extracted'].map({
        'EXTRACTED': 1, 'Include': 1, 1: 1
    }).fillna(0).astype(int).values
    
    predicted_probs = processed_df[adjusted_conf_column].values
    
    # Create and store the ROC analysis in the database
    roc_analysis = create_roc_analysis(
        name="Test Analysis",
        description="Loaded from test_data.csv",
        true_labels=true_labels,
        predicted_probs=predicted_probs,
        default_threshold=0.5,
        db=db
    )
    
    return roc_analysis

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="API for ROC curve analysis and visualization"
)

# Set up CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],  # Focus on only the problematic origin for now
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "Accept"],
)

@app.on_event("startup")
def startup():
    Base.metadata.create_all(bind=engine)
    load_test_data()

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