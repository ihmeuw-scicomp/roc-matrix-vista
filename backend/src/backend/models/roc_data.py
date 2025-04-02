from sqlalchemy import Column, Integer, String, Float, ForeignKey, JSON, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime

from backend.db import Base

class ROCAnalysis(Base):
    __tablename__ = "roc_analyses"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(String, nullable=True)
    auc_score = Column(Float, nullable=False)
    default_threshold = Column(Float, nullable=False, default=0.5)
    roc_curve_data = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Fields to store original data
    true_labels = Column(JSON, nullable=True)  # Only labeled data
    predicted_probs = Column(JSON, nullable=True)  # Only for labeled data
    unlabeled_predictions = Column(JSON, nullable=True)  # Only for unlabeled data
    
    confusion_matrices = relationship("ConfusionMatrix", back_populates="roc_analysis", cascade="all, delete")

class ConfusionMatrix(Base):
    __tablename__ = "confusion_matrices"

    id = Column(Integer, primary_key=True, index=True)
    roc_analysis_id = Column(Integer, ForeignKey("roc_analyses.id"))
    threshold = Column(Float, nullable=False)
    true_positives = Column(Integer, nullable=False)
    false_positives = Column(Integer, nullable=False)
    true_negatives = Column(Integer, nullable=False)
    false_negatives = Column(Integer, nullable=False)
    precision = Column(Float, nullable=False)
    recall = Column(Float, nullable=False)
    f1_score = Column(Float, nullable=False)
    accuracy = Column(Float, nullable=False)
    
    roc_analysis = relationship("ROCAnalysis", back_populates="confusion_matrices")
