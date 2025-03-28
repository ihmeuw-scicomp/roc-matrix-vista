from sqlalchemy import func
from sqlalchemy.orm import Session
from typing import List, Optional

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