import numpy as np
from typing import List, Dict, Tuple, Any

class ROCCalculator:
    """Business logic for calculating ROC curve and confusion matrix metrics"""
    
    def __init__(self, probabilities: np.ndarray = None, labels: np.ndarray = None):
        """Initialize with probabilities and true labels or generate synthetic data"""
        if probabilities is None or labels is None:
            self.probabilities, self.labels = self._generate_synthetic_data()
        else:
            self.probabilities = probabilities
            self.labels = labels
            
        # Precompute ROC curve points for efficiency
        self.roc_curve = self._compute_roc_curve()
        
    def _generate_synthetic_data(self, n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic probability and label data for demo purposes"""
        # Positive class (1) samples: probabilities tend to be higher
        pos_probs = np.random.beta(7, 3, size=n_samples//2)
        pos_labels = np.ones(n_samples//2)
        
        # Negative class (0) samples: probabilities tend to be lower
        neg_probs = np.random.beta(3, 7, size=n_samples//2)
        neg_labels = np.zeros(n_samples//2)
        
        # Combine and shuffle
        probs = np.concatenate([pos_probs, neg_probs])
        labels = np.concatenate([pos_labels, neg_labels])
        
        # Shuffle
        idx = np.random.permutation(n_samples)
        return probs[idx], labels[idx]
    
    def _compute_roc_curve(self, threshold_count: int = 100) -> List[Dict[str, float]]:
        """Compute ROC curve points"""
        thresholds = np.linspace(0, 1, threshold_count)
        roc_points = []
        
        for threshold in thresholds:
            cm = self._compute_confusion_matrix(threshold)
            tpr = cm["TP"] / (cm["TP"] + cm["FN"]) if (cm["TP"] + cm["FN"]) > 0 else 0
            fpr = cm["FP"] / (cm["FP"] + cm["TN"]) if (cm["FP"] + cm["TN"]) > 0 else 0
            
            roc_points.append({
                "threshold": float(threshold),
                "tpr": float(tpr),
                "fpr": float(fpr)
            })
            
        return roc_points
    
    def _compute_confusion_matrix(self, threshold: float) -> Dict[str, int]:
        """Compute confusion matrix for a given threshold"""
        predictions = (self.probabilities >= threshold).astype(int)
        
        TP = np.sum((predictions == 1) & (self.labels == 1))
        FP = np.sum((predictions == 1) & (self.labels == 0))
        TN = np.sum((predictions == 0) & (self.labels == 0))
        FN = np.sum((predictions == 0) & (self.labels == 1))
        
        return {
            "TP": int(TP),
            "FP": int(FP),
            "TN": int(TN),
            "FN": int(FN)
        }
    
    def get_metrics(self, threshold: float = 0.5) -> Dict[str, Any]:
        """Get ROC curve and confusion matrix for a specific threshold"""
        confusion_matrix = self._compute_confusion_matrix(threshold)
        
        # Calculate current TPR and FPR at the threshold
        tpr = confusion_matrix["TP"] / (confusion_matrix["TP"] + confusion_matrix["FN"]) if (confusion_matrix["TP"] + confusion_matrix["FN"]) > 0 else 0
        fpr = confusion_matrix["FP"] / (confusion_matrix["FP"] + confusion_matrix["TN"]) if (confusion_matrix["FP"] + confusion_matrix["TN"]) > 0 else 0
        
        return {
            "threshold": threshold,
            "roc_curve": self.roc_curve,
            "confusion_matrix": confusion_matrix,
            "current_metrics": {
                "tpr": float(tpr),
                "fpr": float(fpr)
            }
        }
