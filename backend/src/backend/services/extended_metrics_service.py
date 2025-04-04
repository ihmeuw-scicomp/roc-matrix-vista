import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import logging

from backend.models.extended_metrics import DistributionBin, WorkloadEstimation, ExtendedMetricsResponse, ValidationMetrics
from backend.models.roc_data import ROCAnalysis
from backend.repositories.roc_analysis_repository import get_roc_analysis
from sqlalchemy.orm import Session
from backend.services.metrics_utils import get_validation_metrics

logger = logging.getLogger(__name__)

def calculate_bins(probabilities: np.ndarray, num_bins: int = 20) -> List[DistributionBin]:
    """
    Calculate histogram bins from probability distribution.
    """
    if len(probabilities) == 0:
        return []
        
    hist, bin_edges = np.histogram(probabilities, bins=num_bins, range=(0, 1))
    
    bins = []
    for i in range(len(hist)):
        bin_data = DistributionBin(
            bin_start=float(bin_edges[i]),
            bin_end=float(bin_edges[i+1]),
            count=int(hist[i])
        )
        bins.append(bin_data)
    
    return bins

def calculate_workload_estimation(
    probabilities: np.ndarray, 
    threshold: float, 
    tpr: float,
    precision: float
) -> WorkloadEstimation:
    """
    Calculate workload estimation based on the threshold and model performance metrics.
    
    Args:
        probabilities: Array of predicted probabilities for unlabeled data
        threshold: Classification threshold
        tpr: True positive rate from validation data at the selected threshold
        precision: Precision from validation data at the selected threshold
    """
    # Count predictions above and below threshold
    predicted_positives = int(np.sum(probabilities >= threshold))
    predicted_negatives = int(np.sum(probabilities < threshold))
    total_articles = len(probabilities)
    
    # Estimate true positives and false positives among predicted positives
    expected_true_positives = int(predicted_positives * precision)
    expected_false_positives = predicted_positives - expected_true_positives
    
    # Estimate missed relevant articles (using 1-TPR as false negative rate)
    false_negative_rate = 1.0 - tpr
    # This is an approximation based on the validation set performance
    expected_missed_relevant = int(predicted_negatives * false_negative_rate * (expected_true_positives / (total_articles * precision))) if precision > 0 else 0
    
    return WorkloadEstimation(
        predicted_positives=predicted_positives,
        predicted_negatives=predicted_negatives,
        expected_true_positives=expected_true_positives,
        expected_false_positives=expected_false_positives,
        expected_missed_relevant=expected_missed_relevant,
        total_articles=total_articles
    )

def get_unlabeled_distribution(analysis: ROCAnalysis, threshold: float, tpr: float, precision: float) -> Tuple[List[DistributionBin], WorkloadEstimation]:
    """
    Get distribution bins and workload estimation for unlabeled data.
    
    Args:
        analysis: ROC analysis containing unlabeled predictions
        threshold: Classification threshold
        tpr: True positive rate from validation data
        precision: Precision from validation data
    
    Returns:
        Tuple of (distribution_bins, workload_estimation)
    """
    # Get predicted probabilities for unlabeled data
    unlabeled_probs = np.array(analysis.unlabeled_predictions) if analysis.unlabeled_predictions else np.array([])
    
    if len(unlabeled_probs) == 0:
        logger.warning(f"No unlabeled predictions found for analysis {analysis.id}")
        return [], None
        
    # Calculate distribution bins
    distribution_bins = calculate_bins(unlabeled_probs)
    
    # Calculate workload estimation
    workload = calculate_workload_estimation(unlabeled_probs, threshold, tpr, precision)
    
    return distribution_bins, workload

def get_extended_metrics(
    db: Session, 
    analysis_id: str, 
    threshold: float
) -> Optional[ExtendedMetricsResponse]:
    """
    Get extended metrics for a given ROC analysis at the specified threshold.
    """
    # Get the ROC analysis from the database
    analysis = get_roc_analysis(db, analysis_id)
    if not analysis:
        logger.error(f"Analysis with ID {analysis_id} not found")
        return None
    
    # Get the relevant data from the analysis
    try:
        # 1) Get metrics from labeled data
        labeled_probs = np.array(analysis.predicted_probs) if analysis.predicted_probs else np.array([])
        true_labels = np.array(analysis.true_labels) if analysis.true_labels else np.array([])
        
        # Calculate TPR and precision from labeled data - using shared utility function
        tpr, precision = get_validation_metrics(labeled_probs, true_labels, threshold)
        
        # Ensure values are JSON-serializable (no infinity or NaN)
        def safe_float(value):
            try:
                val = float(value)
                return 0.0 if np.isinf(val) or np.isnan(val) else val
            except:
                return 0.0
                
        # Create validation metrics object with safe values
        validation_metrics = ValidationMetrics(
            tpr=safe_float(tpr),
            precision=safe_float(precision),
            labeled_count=len(labeled_probs)
        )
        
        # 2) Get distribution for unlabeled data
        distribution_bins, workload = get_unlabeled_distribution(analysis, threshold, safe_float(tpr), safe_float(precision))
        
        if not distribution_bins or not workload:
            logger.warning(f"Could not calculate distribution or workload for analysis {analysis_id}")
            return None
        
        return ExtendedMetricsResponse(
            distribution_data=distribution_bins,
            workload_estimation=workload,
            validation_metrics=validation_metrics,
            threshold=threshold,
            analysis_id=analysis_id
        )
        
    except Exception as e:
        logger.error(f"Error calculating extended metrics: {str(e)}")
        return None
