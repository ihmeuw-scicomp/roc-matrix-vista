import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def compute_confusion_matrix(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> Dict:
    """
    Compute confusion matrix and derived metrics for binary classification.
    
    Args:
        y_true: Array of true binary labels
        y_score: Array of predicted probabilities/scores
        threshold: Classification threshold to convert scores to binary predictions
    
    Returns:
        Dict containing confusion matrix values and metrics
    """
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Avoid division by zero and infinite values
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    
    return {
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1)
    }

def compute_roc_curve(y_true: np.ndarray, y_score: np.ndarray) -> List[Dict]:
    """
    Compute ROC curve points.
    
    Args:
        y_true: Array of true binary labels
        y_score: Array of predicted probabilities/scores
    
    Returns:
        List of dictionaries with threshold, tpr, and fpr values
    """
    # Handle edge case where all labels are the same
    if len(np.unique(y_true)) == 1:
        # Return a simplified ROC curve with 0 and 1 points
        if y_true[0] == 1:
            # All positive
            return [
                {"threshold": 0.0, "tpr": 1.0, "fpr": 1.0},
                {"threshold": 1.0, "tpr": 0.0, "fpr": 0.0}
            ]
        else:
            # All negative
            return [
                {"threshold": 0.0, "tpr": 0.0, "fpr": 1.0},
                {"threshold": 1.0, "tpr": 0.0, "fpr": 0.0}
            ]
    
    # Normal case with mixed labels
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    
    # Format into list of dictionaries
    roc_points = []
    for i in range(len(fpr)):
        roc_points.append({
            "threshold": float(thresholds[i]) if i < len(thresholds) else 1.0,
            "tpr": float(tpr[i]),
            "fpr": float(fpr[i])
        })
    
    return roc_points

def compute_auc(roc_points: List[Dict]) -> float:
    """
    Compute AUC score from ROC points.
    
    Args:
        roc_points: List of dictionaries with ROC curve points
    
    Returns:
        AUC score
    """
    # Extract tpr and fpr from roc_points
    tpr = [point["tpr"] for point in roc_points]
    fpr = [point["fpr"] for point in roc_points]
    
    # Compute AUC using trapezoidal rule
    try:
        auc_score = auc(fpr, tpr)
        return float(auc_score)
    except Exception:
        # In case of error, return 0.5 (random classifier)
        return 0.5

def calculate_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[List[Dict], float]:
    """
    Calculate ROC curve points and AUC score.
    
    Args:
        y_true: Array of true binary labels
        y_score: Array of predicted probabilities/scores
    
    Returns:
        Tuple with (roc_points, auc_score)
    """
    roc_points = compute_roc_curve(y_true, y_score)
    auc_score = compute_auc(roc_points)
    return roc_points, auc_score

def get_validation_metrics(labeled_probs: np.ndarray, true_labels: np.ndarray, threshold: float) -> Tuple[float, float]:
    """
    Calculate validation metrics (TPR, precision) from labeled data at the given threshold.
    
    Args:
        labeled_probs: Array of predicted probabilities for labeled data
        true_labels: Array of true labels (1 = included, 0 = excluded)
        threshold: Classification threshold
    
    Returns:
        Tuple of (tpr, precision)
    """
    if len(labeled_probs) == 0 or len(true_labels) == 0:
        return 0.5, 0.5  # Default values if no labeled data
        
    # Apply threshold to get binary predictions
    predictions = labeled_probs >= threshold
    
    # Calculate true positive rate (recall) and precision
    tp = np.sum((predictions == True) & (true_labels == 1))
    fp = np.sum((predictions == True) & (true_labels == 0))
    fn = np.sum((predictions == False) & (true_labels == 1))
    
    # Avoid division by zero
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    return tpr, precision 