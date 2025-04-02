from sqlalchemy import Column, String, Float, Integer, Boolean, JSON, DateTime, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

Base = declarative_base()

class ROCAnalysis(Base):
    __tablename__ = "roc_analyses"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    description = Column(Text)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    # ROC curve data
    fpr = Column(JSON)  # List of false positive rates
    tpr = Column(JSON)  # List of true positive rates
    thresholds = Column(JSON)  # List of thresholds
    auc = Column(Float)  # Area under the curve
    
    # Additional metrics
    precision = Column(JSON)  # List of precision values
    recall = Column(JSON)  # Same as TPR, but kept for clarity
    f1_scores = Column(JSON)  # List of F1 scores
    
    # Store original data
    true_labels = Column(JSON)  # Binary truth values
    predicted_probs = Column(JSON)  # Predicted probabilities for labeled data
    
    # Store unlabeled predictions
    unlabeled_predictions = Column(JSON)  # Predicted probabilities for unlabeled data
    
    # Confusion matrices at different thresholds can be stored as relationships
    confusion_matrices = relationship("ConfusionMatrix", back_populates="roc_analysis", cascade="all, delete-orphan")

class ConfusionMatrix(Base):
    __tablename__ = "confusion_matrices"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    roc_analysis_id = Column(String, ForeignKey("roc_analyses.id", ondelete="CASCADE"))
    threshold = Column(Float, nullable=False)
    
    true_positives = Column(Integer)
    false_positives = Column(Integer)
    true_negatives = Column(Integer)
    false_negatives = Column(Integer)
    
    roc_analysis = relationship("ROCAnalysis", back_populates="confusion_matrices")
