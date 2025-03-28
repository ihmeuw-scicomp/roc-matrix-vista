from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
import io

from backend.db.session import get_db
from backend.models.roc_data import ROCAnalysis, ConfusionMatrix
from backend.services.roc_analysis_service import create_roc_analysis, process_dataframe

router = APIRouter()

@router.post("/roc-analysis/")
async def create_new_analysis(
    name: str = Form(...),
    description: Optional[str] = Form(None),
    file: UploadFile = File(...),
    threshold: float = Form(0.5),
    true_label_col: str = Form(...),
    pred_prob_col: str = Form(...),
    db: Session = Depends(get_db)
):
    """Create a new ROC analysis from uploaded CSV data"""
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

@router.get("/roc-analysis/{analysis_id}")
def get_roc_analysis(analysis_id: int, db: Session = Depends(get_db)):
    """Get ROC analysis data by ID"""
    roc_analysis = db.query(ROCAnalysis).filter(ROCAnalysis.id == analysis_id).first()
    if not roc_analysis:
        raise HTTPException(status_code=404, detail="ROC analysis not found")
    
    return roc_analysis

@router.get("/roc-analysis/{analysis_id}/confusion-matrices")
def get_confusion_matrices(analysis_id: int, db: Session = Depends(get_db)):
    """Get confusion matrices for a specific ROC analysis"""
    matrices = db.query(ConfusionMatrix).filter(ConfusionMatrix.roc_analysis_id == analysis_id).all()
    if not matrices:
        raise HTTPException(status_code=404, detail="No confusion matrices found")
    
    return matrices
