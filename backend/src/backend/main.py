from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings
from backend.api.routes import roc_routes
from backend.db import engine, Base
from backend.services.roc_analysis_service import create_roc_analysis, process_dataframe
from backend.models.roc_data import ROCAnalysis
from backend.db import get_db
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

    # Extract true labels and predicted probabilities from the processed dataframe
    true_labels = processed_df['Extracted'].map({
        'EXTRACTED': 1, 'Include': 1, 1: 1
    }).fillna(0).astype(int).values
    predicted_probs = processed_df['prediction_conf'].values
    # Create and store the ROC analysis in the database
    roc_analysis = create_roc_analysis(
        name="Test Analysis",
        description="Loaded from test_data.csv",
        true_labels=true_labels,
        predicted_probs=predicted_probs,
        default_threshold=0.5,
        db=db
    )

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="API for ROC curve analysis and visualization"
)

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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