from sqlalchemy import func
from sqlalchemy.orm import Session
from typing import List, Optional
import numpy as np

from backend.models.roc_data import ROCAnalysis, ConfusionMatrix

def get_roc_analysis(db: Session, analysis_id: int) -> Optional[ROCAnalysis]:
    """
    Get a ROC analysis by ID.
    
    Args:
        db: Database session
        analysis_id: ID of the analysis to retrieve
        
    Returns:
        ROCAnalysis object if found, None otherwise
    """
    return db.query(ROCAnalysis).filter(ROCAnalysis.id == analysis_id).first()

def get_all_roc_analyses(db: Session, skip: int = 0, limit: int = 100) -> List[ROCAnalysis]:
    """
    Get all ROC analyses with pagination.
    
    Args:
        db: Database session
        skip: Number of records to skip for pagination
        limit: Maximum number of records to return
        
    Returns:
        List of ROCAnalysis objects
    """
    return db.query(ROCAnalysis).order_by(ROCAnalysis.id.desc()).offset(skip).limit(limit).all()

def get_confusion_matrix(db: Session, analysis_id: int, threshold: float) -> Optional[ConfusionMatrix]:
    """
    Get confusion matrix for a specific analysis and threshold.
    
    Args:
        db: Database session
        analysis_id: ID of the analysis
        threshold: Classification threshold
        
    Returns:
        ConfusionMatrix object if found, None otherwise
    """
    return db.query(ConfusionMatrix).filter(
        ConfusionMatrix.roc_analysis_id == analysis_id,
        ConfusionMatrix.threshold == threshold
    ).first()

def get_closest_confusion_matrix(db: Session, analysis_id: int, threshold: float) -> Optional[ConfusionMatrix]:
    """
    Get the confusion matrix with the closest threshold value for a given analysis.
    
    Args:
        db: Database session
        analysis_id: ID of the analysis
        threshold: Target threshold value
        
    Returns:
        ConfusionMatrix object with the closest threshold, None if no matrices exist
    """
    analysis = get_roc_analysis(db, analysis_id)
    if not analysis:
        return None
        
    closest_matrix = db.query(ConfusionMatrix)\
        .filter(ConfusionMatrix.roc_analysis_id == analysis_id)\
        .order_by(func.abs(ConfusionMatrix.threshold - threshold))\
        .first()
    
    return closest_matrix

def delete_roc_analysis(db: Session, analysis_id: int) -> bool:
    """
    Delete a ROC analysis by ID.
    
    Args:
        db: Database session
        analysis_id: ID of the analysis to delete
        
    Returns:
        True if deleted successfully, False if not found
    """
    analysis = get_roc_analysis(db, analysis_id)
    if analysis is None:
        return False
    
    db.delete(analysis)
    db.commit()
    return True

def create_confusion_matrix(db: Session, analysis_id: int, threshold: float, cm_data: dict) -> ConfusionMatrix:
    """
    Create a new confusion matrix entry.
    
    Args:
        db: Database session
        analysis_id: ID of the associated ROC analysis
        threshold: Classification threshold
        cm_data: Dictionary containing confusion matrix values
        
    Returns:
        Created ConfusionMatrix object
    """
    cm = ConfusionMatrix(
        roc_analysis_id=analysis_id,
        threshold=threshold,
        **cm_data
    )
    db.add(cm)
    db.commit()
    db.refresh(cm)
    return cm

def get_or_create_confusion_matrix(db: Session, analysis_id: int, threshold: float) -> Optional[ConfusionMatrix]:
    """
    Get an existing confusion matrix or create a new one for the exact threshold.
    This should be used by the frontend slider to ensure exact threshold matching.
    
    Args:
        db: Database session
        analysis_id: ID of the ROC analysis
        threshold: Classification threshold
        
    Returns:
        ConfusionMatrix object, None if analysis doesn't exist
    """
    # Try to find an existing confusion matrix with this exact threshold
    existing_cm = get_confusion_matrix(db, analysis_id, threshold)
    if existing_cm:
        return existing_cm
    
    # If not found, get the analysis
    analysis = get_roc_analysis(db, analysis_id)
    if not analysis:
        return None
    
    # Compute and create a new confusion matrix
    return compute_and_save_confusion_matrix(db, analysis, threshold)
    
def compute_and_save_confusion_matrix(db: Session, analysis: ROCAnalysis, threshold: float) -> ConfusionMatrix:
    """
    Compute a confusion matrix using raw data and save it to the database.
    
    Args:
        db: Database session
        analysis: ROCAnalysis object containing the data
        threshold: Classification threshold
        
    Returns:
        Created ConfusionMatrix object
    """
    from backend.services.metrics_utils import compute_confusion_matrix
    
    # Compute the confusion matrix using the raw data
    y_true = np.array(analysis.true_labels)
    y_score = np.array(analysis.predicted_probs)
    cm_data = compute_confusion_matrix(y_true, y_score, threshold)
    
    # Save this confusion matrix
    return create_confusion_matrix(db, analysis.id, threshold, cm_data)