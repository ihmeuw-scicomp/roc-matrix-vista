from pydantic import BaseModel
from typing import List, Dict, Optional

class DistributionBin(BaseModel):
    bin_start: float
    bin_end: float
    count: int

class WorkloadEstimation(BaseModel):
    predicted_positives: int
    predicted_negatives: int
    expected_true_positives: int
    expected_false_positives: int
    expected_missed_relevant: int
    total_articles: int

class ExtendedMetricsResponse(BaseModel):
    distribution_data: List[DistributionBin]
    workload_estimation: WorkloadEstimation
    threshold: float
    analysis_id: str
