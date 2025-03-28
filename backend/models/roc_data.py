from sqlalchemy import Column, Integer, String, Float, ForeignKey, JSON
from sqlalchemy.orm import relationship
from typing import List, Dict

from backend.db.base_class import Base

class ROCAnalysis(Base):
    __tablename__ = "roc_analyses"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(String, nullable=True)
    default_threshold = Column(Float, default=0.5)
    roc_curve_data = Column(JSON)
    auc_score = Column(Float)
    
    confusion_matrices = relationship("ConfusionMatrix", back_populates="roc_analysis")
    
class ConfusionMatrix(Base):
    __tablename__ = "confusion_matrices"
    
    id = Column(Integer, primary_key=True, index=True)
    roc_analysis_id = Column(Integer, ForeignKey("roc_analyses.id"))
    threshold = Column(Float)
    
    # Confusion matrix values
    true_positives = Column(Integer)
    false_positives = Column(Integer)
    true_negatives = Column(Integer)
    false_negatives = Column(Integer)
    
    # Derived metrics
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    
    roc_analysis = relationship("ROCAnalysis", back_populates="confusion_matrices")
