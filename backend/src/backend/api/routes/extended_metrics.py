from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import Optional

from backend.db import get_db
from backend.models.extended_metrics import ExtendedMetricsResponse
from backend.services.extended_metrics_service import get_extended_metrics

router = APIRouter()

@router.get("/metrics/extended", response_model=ExtendedMetricsResponse)
def get_extended_metrics_endpoint(
    threshold: float = Query(..., ge=0.0, le=1.0, description="Classification threshold"),
    analysis_id: str = Query(..., description="ROC analysis ID"),
    db: Session = Depends(get_db)
):
    """
    Get extended metrics for a given ROC analysis at the specified threshold.
    This includes distribution data and workload estimation.
    """
    result = get_extended_metrics(db, analysis_id, threshold)
    if not result:
        raise HTTPException(status_code=404, detail=f"Could not retrieve extended metrics for analysis {analysis_id}")
    return result
