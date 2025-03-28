from pydantic import BaseModel
from typing import List, Dict, Optional

class ROCPoint(BaseModel):
    threshold: float
    tpr: float
    fpr: float

class ConfusionMatrixSchema(BaseModel):
    threshold: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float

    class Config:
        from_attributes = True

class ROCAnalysisSchema(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    default_threshold: float = 0.5
    roc_curve_data: List[Dict]
    auc_score: float
    
    class Config:
        from_attributes = True

class ROCAnalysisCreate(BaseModel):
    name: str
    description: Optional[str] = None
    default_threshold: float = 0.5
    
class ROCMetricsResponse(BaseModel):
    threshold: float
    roc_curve: List[ROCPoint]
    confusion_matrix: ConfusionMatrixSchema
