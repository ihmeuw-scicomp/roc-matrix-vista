from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query
from sqlalchemy.orm import Session
import pandas as pd
import numpy as np
from typing import List, Optional

from backend.db.session import get_db
from backend.schemas.roc_schemas import ROCAnalysisSchema, ROCAnalysisCreate, ROCMetricsResponse
from backend.services.roc_analysis_service import (
    process_dataframe, create_roc_analysis, compute_confusion_matrix
)
from backend.crud.roc_crud import (
    get_roc_analysis, get_all_roc_analyses, get_closest_confusion_matrix, delete_roc_analysis
)

router = APIRouter()

@router.post("/upload-data", response_model=ROCAnalysisSchema)
async def upload_data(
    file: UploadFile = File(...),
    name: str = Form(...),
    description: Optional[str] = Form(None),
    default_threshold: float = Form(0.5),
    db: Session = Depends(get_db)
):
    """Upload a CSV file and create a new ROC analysis"""
    try:
        # Read and process CSV
        contents = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(contents))
        processed_df = process_dataframe(df)
        
        # Extract true labels and predicted probabilities
        true_labels = processed_df['Full.Text.Incl/Excl'].map({
            'EXTRACTED': 1, 'Include': 1, 1: 1
        }).fillna(0).astype(int).values
        
        predicted_probs = processed_df['prediction_conf'].values
        
        # Create ROC analysis
        roc_analysis = create_roc_analysis(
            name=name,
            description=description,
            true_labels=true_labels,
            predicted_probs=predicted_probs,
            default_threshold=default_threshold,
            db=db
        )
        
        return roc_analysis
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")

@router.get("/analyses", response_model=List[ROCAnalysisSchema])
def get_analyses(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Get all ROC analyses"""
    return get_all_roc_analyses(db, skip, limit)

@router.get("/analyses/{analysis_id}", response_model=ROCAnalysisSchema)
def get_analysis(
    analysis_id: int,
    db: Session = Depends(get_db)
):
    """Get a specific ROC analysis"""
    analysis = get_roc_analysis(db, analysis_id)
    if analysis is None:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return analysis

@router.get("/analyses/{analysis_id}/metrics", response_model=ROCMetricsResponse)
def get_metrics(
    analysis_id: int,
    threshold: float = Query(0.5, ge=0.0, le=1.0),
    db: Session = Depends(get_db)
):
    """Get metrics for a specific ROC analysis at a given threshold"""
    analysis = get_roc_analysis(db, analysis_id)
    if analysis is None:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    # Get confusion matrix for this threshold (or closest)
    confusion_matrix = get_closest_confusion_matrix(db, analysis_id, threshold)
    if confusion_matrix is None:
        raise HTTPException(status_code=404, detail="No confusion matrix found")
    
    # Format the response
    roc_points = [
        {"threshold": point["threshold"], "tpr": point["tpr"], "fpr": point["fpr"]}
        for point in analysis.roc_curve_data
    ]
    
    return {
        "threshold": confusion_matrix.threshold,
        "roc_curve": roc_points,
        "confusion_matrix": confusion_matrix
    }

@router.delete("/analyses/{analysis_id}")
def delete_analysis(
    analysis_id: int,
    db: Session = Depends(get_db)
):
    """Delete a ROC analysis"""
    success = delete_roc_analysis(db, analysis_id)
    if not success:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return {"message": "Analysis deleted successfully"}
