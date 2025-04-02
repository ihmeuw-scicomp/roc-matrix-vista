from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query
from sqlalchemy.orm import Session
import pandas as pd
import numpy as np
from typing import List, Optional
import io
import logging

from backend.db import get_db
from backend.schemas.roc_schemas import ROCAnalysisSchema, ROCAnalysisCreate, ROCMetricsResponse
from backend.services.extended_metrics_service import get_extended_metrics
from backend.services.roc_analysis_service import (
    process_dataframe, create_roc_analysis, find_column_by_suffix,
)
from backend.repositories.roc_analysis_repository import (
    get_roc_analysis, get_all_roc_analyses, get_closest_confusion_matrix, delete_roc_analysis, get_or_create_confusion_matrix
)

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/analyses/{analysis_id}/upload-data", response_model=ROCAnalysisSchema)
async def upload_data(
    analysis_id: int,
    file: UploadFile = File(...),
    name: str = Form(...),
    description: Optional[str] = Form(None),
    default_threshold: float = Form(0.5),
    db: Session = Depends(get_db)
):
    """Upload a CSV file and create a new ROC analysis"""
    try:
        # Read and validate the file
        contents = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(contents))
        
        # Process the dataframe using the service
        processed_df = process_dataframe(df)
        
        # Find the adjusted confidence column dynamically by suffix
        adjusted_conf_column = find_column_by_suffix(processed_df, '_confidence_adjusted')

        if not adjusted_conf_column:
            raise ValueError("Could not find a column with suffix '_confidence_adjusted' in the processed dataframe")
        
        # Split data into labeled and unlabeled sets
        labeled_df = processed_df[processed_df['Extracted'].notna()].copy()
        unlabeled_df = processed_df[processed_df['Extracted'].isna()].copy()
        
        # Extract true labels and predicted probabilities from the labeled data
        true_labels = labeled_df['Extracted'].map({
            'EXTRACTED': 1, 'Include': 1, 1: 1
        }).fillna(0).astype(int).values
        
        predicted_probs = labeled_df[adjusted_conf_column].values
        
        # Extract predictions for unlabeled data
        unlabeled_predictions = unlabeled_df[adjusted_conf_column].values.tolist() if not unlabeled_df.empty else []
        
        # Create ROC analysis using the service
        roc_analysis = create_roc_analysis(
            id=analysis_id,
            name=name,
            description=description,
            true_labels=true_labels.tolist(),
            predicted_probs=predicted_probs.tolist(),
            unlabeled_predictions=unlabeled_predictions,
            default_threshold=default_threshold,
            db=db
        )
        
        return roc_analysis
        
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")
    
@router.get("/analyses-status/{analysis_id}")
def get_analysis_status(
    analysis_id: int,
    db: Session = Depends(get_db)
):
    """
    Get the status of an analysis.
    """
    # Use repository function to get analysis
    analysis = get_roc_analysis(db, analysis_id)
    
    # Instead of raising a 404, return a valid response indicating the analysis doesn't exist
    if not analysis:
        logger.info(f"Analysis with ID {analysis_id} not found")
        return {
            "analysis_id": analysis_id,
            "exists": False,
            "has_roc_data": False,
            "has_confusion_matrix": False
        }
    
    # Analysis exists, return its status
    status_response = {
        "analysis_id": analysis_id,
        "exists": True,
        "has_roc_data": bool(getattr(analysis, 'roc_curve_data', None) and len(analysis.roc_curve_data) > 0),
        "has_confusion_matrix": bool(getattr(analysis, 'confusion_matrices', None) and len(analysis.confusion_matrices) > 0)
    }
    return status_response

@router.get("/analyses", response_model=List[ROCAnalysisSchema])
def get_analyses(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Get all ROC analyses"""
    # Use repository function directly
    return get_all_roc_analyses(db, skip, limit)

@router.get("/analyses/{analysis_id}", response_model=ROCAnalysisSchema)
def get_analysis(
    analysis_id: int,
    db: Session = Depends(get_db)
):
    """Get a specific ROC analysis"""
    # Use repository function to get analysis
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
    # Use repository function to get analysis
    analysis = get_roc_analysis(db, analysis_id)
    if analysis is None:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    # Get confusion matrix using repository function
    confusion_matrix = get_or_create_confusion_matrix(db, analysis_id, threshold)
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
    # Use repository function to delete analysis
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
        df = pd.read_csv(pd.io.common.BytesIO(content))
        
        # Validate required columns
        if true_label_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"True label column '{true_label_col}' not found in CSV")
        if pred_prob_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Prediction probability column '{pred_prob_col}' not found in CSV")
        
        # Extract data
        true_labels = df[true_label_col].map({
            'EXTRACTED': 1, 'Include': 1, 'True': 1, 'Yes': 1, 1: 1, '1': 1
        }).fillna(0).astype(int).values.tolist()
        
        pred_probs = df[pred_prob_col].astype(float).values.tolist()
        
        # Create ROC analysis using the service
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
        
    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise e
    except Exception as e:
        logger.error(f"Error creating analysis: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error creating analysis: {str(e)}")

@router.get("/analyses/{analysis_id}/extended-metrics")
def get_extended_metrics_endpoint(
    analysis_id: str,
    threshold: float = Query(..., ge=0.0, le=1.0, description="Classification threshold"),
    db: Session = Depends(get_db)
):
    """
    Get extended metrics for a given ROC analysis at the specified threshold.
    """
    # Call service function to get extended metrics
    result = get_extended_metrics(db, analysis_id, threshold)
    if not result:
        raise HTTPException(status_code=404, detail=f"Could not retrieve extended metrics for analysis {analysis_id}")
    return result

@router.get("/roc-analysis/{analysis_id}/confusion-matrices")
def get_confusion_matrices(analysis_id: int, db: Session = Depends(get_db)):
    logger.critical("THIS IS A TEST CRITICAL LOG")
    """Get confusion matrices for a specific ROC analysis"""
    matrices = db.query(ConfusionMatrix).filter(ConfusionMatrix.roc_analysis_id == analysis_id).all()
    if not matrices:
        raise HTTPException(status_code=404, detail="No confusion matrices found")
    
    return matrices
