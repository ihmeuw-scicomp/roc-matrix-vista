from sqlalchemy import func
from sqlalchemy.orm import Session
from typing import List, Optional
import numpy as np

from backend.models.roc_data import ROCAnalysis, ConfusionMatrix

def get_roc_analysis(db: Session, analysis_id: int) -> Optional[ROCAnalysis]:
    """Get a ROC analysis by ID"""
    return db.query(ROCAnalysis).filter(ROCAnalysis.id == analysis_id).first()

def get_all_roc_analyses(db: Session, skip: int = 0, limit: int = 100) -> List[ROCAnalysis]:
    """Get all ROC analyses with pagination"""
    return db.query(ROCAnalysis).order_by(ROCAnalysis.id.desc()).offset(skip).limit(limit).all()

def get_confusion_matrix(db: Session, analysis_id: int, threshold: float) -> Optional[ConfusionMatrix]:
    """Get confusion matrix for a specific analysis and threshold"""
    return db.query(ConfusionMatrix).filter(
        ConfusionMatrix.roc_analysis_id == analysis_id,
        ConfusionMatrix.threshold == threshold
    ).first()

def get_closest_confusion_matrix(db: Session, analysis_id: int, threshold: float) -> Optional[ConfusionMatrix]:
    """Get the confusion matrix with the closest threshold value for a given analysis"""
    analysis = get_roc_analysis(db, analysis_id)
    if not analysis:
        return None
        
    closest_matrix = db.query(ConfusionMatrix)\
        .filter(ConfusionMatrix.roc_analysis_id == analysis_id)\
        .order_by(func.abs(ConfusionMatrix.threshold - threshold))\
        .first()
    
    return closest_matrix

def delete_roc_analysis(db: Session, analysis_id: int) -> bool:
    """Delete a ROC analysis by ID"""
    analysis = get_roc_analysis(db, analysis_id)
    if analysis is None:
        return False
    
    db.delete(analysis)
    db.commit()
    return True

def get_or_create_confusion_matrix(db: Session, analysis_id: int, threshold: float) -> ConfusionMatrix:
    """
    Get an existing confusion matrix or create a new one for the exact threshold.
    This should be used by the frontend slider to ensure exact threshold matching.
    """
    from backend.services.roc_analysis_service import compute_confusion_matrix
    
    # Try to find an existing confusion matrix with this exact threshold
    existing_cm = get_confusion_matrix(db, analysis_id, threshold)
    if existing_cm:
        return existing_cm
    
    # If not found, compute a new one
    analysis = get_roc_analysis(db, analysis_id)
    if not analysis:
        return None
        
    # Compute the confusion matrix using the raw data
    y_true = np.array(analysis.true_labels)
    y_score = np.array(analysis.predicted_probs)
    cm_data = compute_confusion_matrix(y_true, y_score, threshold)
    
    # Save this confusion matrix
    cm = ConfusionMatrix(
        roc_analysis_id=analysis_id,
        threshold=threshold,
        **cm_data
    )
    db.add(cm)
    db.commit()
    db.refresh(cm)
    
    return cm