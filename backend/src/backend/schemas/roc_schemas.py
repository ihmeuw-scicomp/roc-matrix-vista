from pydantic import BaseModel
from typing import List, Dict, Optional, Any

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
    precision: float
    recall: float
    f1_score: float
    accuracy: float

    class Config:
        orm_mode = True

class ROCPointSchema(BaseModel):
    threshold: float
    tpr: float
    fpr: float

class ROCAnalysisSchema(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    auc_score: float
    default_threshold: float
    roc_curve_data: List[Dict[str, float]]
    
    class Config:
        orm_mode = True

class ROCAnalysisCreate(BaseModel):
    name: str
    description: Optional[str] = None
    default_threshold: float = 0.5
    
class ROCMetricsResponse(BaseModel):
    threshold: float
    roc_curve: List[Dict[str, float]]
    confusion_matrix: ConfusionMatrixSchema
    current_metrics: Dict[str, float]

    class Config:
        orm_mode = True
