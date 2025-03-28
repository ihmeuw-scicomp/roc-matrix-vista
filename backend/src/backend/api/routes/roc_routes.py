from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query
from sqlalchemy.orm import Session
import pandas as pd
import numpy as np
from typing import List, Optional
import io

from backend.db import get_db
from backend.schemas.roc_schemas import ROCAnalysisSchema, ROCAnalysisCreate, ROCMetricsResponse
from backend.services.roc_analysis_service import (
    process_dataframe, create_roc_analysis, compute_confusion_matrix
)
from backend.repositories.roc_analysis_repository import (
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
    
    # Find the current point corresponding to the requested threshold
    current_point = next(
        (point for point in roc_points if point["threshold"] == threshold),
        min(roc_points, key=lambda p: abs(p["threshold"] - threshold))
    )
    
    # Handle infinite values by replacing with None to make JSON serializable
    response_data = {
        "threshold": float(confusion_matrix.threshold),
        "roc_curve": roc_points,
        "confusion_matrix": {
            "threshold": float(confusion_matrix.threshold),
            "true_positives": confusion_matrix.true_positives,
            "false_positives": confusion_matrix.false_positives,
            "true_negatives": confusion_matrix.true_negatives,
            "false_negatives": confusion_matrix.false_negatives,
            "precision": None if np.isinf(confusion_matrix.precision) else float(confusion_matrix.precision),
            "recall": None if np.isinf(confusion_matrix.recall) else float(confusion_matrix.recall),
            "f1_score": None if np.isinf(confusion_matrix.f1_score) else float(confusion_matrix.f1_score),
            "accuracy": None if np.isinf(confusion_matrix.accuracy) else float(confusion_matrix.accuracy)
        },
        "current_metrics": {
            "tpr": None if np.isinf(current_point["tpr"]) else float(current_point["tpr"]),
            "fpr": None if np.isinf(current_point["fpr"]) else float(current_point["fpr"])
        }
    }
    
    return response_data

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

@router.post("/roc-analysis/", response_model=dict)
async def create_new_analysis(
    name: str = Form(...),
    description: Optional[str] = Form(None),
    file: UploadFile = File(...),
    threshold: float = Form(0.5),
    true_label_col: str = Form(...),
    pred_prob_col: str = Form(...),
    db: Session = Depends(get_db)
):
    """Create a new ROC analysis from uploaded CSV data with column specification"""
    try:
        # Read CSV file
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        
        # Process data
        if not true_label_col in df.columns or not pred_prob_col in df.columns:
            raise HTTPException(status_code=400, detail="Specified columns not found in CSV")
        
        # Extract true labels and predicted probabilities
        true_labels = df[true_label_col].values.astype(np.int32)
        pred_probs = df[pred_prob_col].values.astype(np.float32)
        
        # Create ROC analysis
        roc_analysis = create_roc_analysis(
            name=name,
            description=description,
            true_labels=true_labels,
            predicted_probs=pred_probs,
            default_threshold=threshold,
            db=db
        )
        
        return {
            "id": roc_analysis.id,
            "name": roc_analysis.name, 
            "auc_score": roc_analysis.auc_score
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@router.get("/roc-analysis/{analysis_id}/confusion-matrices")
def get_confusion_matrices(analysis_id: int, db: Session = Depends(get_db)):
    """Get confusion matrices for a specific ROC analysis"""
    matrices = db.query(ConfusionMatrix).filter(ConfusionMatrix.roc_analysis_id == analysis_id).all()
    if not matrices:
        raise HTTPException(status_code=404, detail="No confusion matrices found")
    
    return matrices
