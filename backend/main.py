from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np

app = FastAPI()

# Configure CORS to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ROCPoint(BaseModel):
    threshold: float
    tpr: float
    fpr: float

class ConfusionMatrix(BaseModel):
    TP: int
    FP: int
    TN: int
    FN: int

class MetricsResponse(BaseModel):
    threshold: float
    roc_curve: list[ROCPoint]
    confusion_matrix: ConfusionMatrix
    current_metrics: dict[str, float]  # New field for exact TPR/FPR at threshold

# Generate demo data
np.random.seed(42)  # For reproducibility
n_samples = 1000
true_labels = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])  # 30% positives
predicted_probs = np.where(
    true_labels == 1,
    np.random.normal(0.7, 0.1, n_samples),  # Positives: mean 0.7, std 0.1
    np.random.normal(0.3, 0.1, n_samples)   # Negatives: mean 0.3, std 0.1
)
predicted_probs = np.clip(predicted_probs, 0, 1)  # Ensure probs are in [0,1]

# Precomputed ROC curve
roc_curve_data = []

# Computation functions
def compute_roc_curve(y_true: np.ndarray, y_score: np.ndarray) -> list[ROCPoint]:
    # Get unique thresholds (sorted)
    thresholds = np.unique(np.append(np.append(0, y_score), 1))
    thresholds = np.sort(thresholds)
    
    # Calculate ROC curve
    roc_points = []
    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        roc_points.append(ROCPoint(threshold=float(threshold), tpr=float(tpr), fpr=float(fpr)))
    
    return roc_points

def compute_confusion_matrix(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> ConfusionMatrix:
    y_pred = (y_score >= threshold).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    return ConfusionMatrix(TP=tp, FP=fp, TN=tn, FN=fn)

def compute_metrics_at_threshold(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    
    tpr = float(tp / (tp + fn) if (tp + fn) > 0 else 0)
    fpr = float(fp / (fp + tn) if (fp + tn) > 0 else 0)
    
    return {"tpr": tpr, "fpr": fpr}

@app.on_event("startup")
def startup_event():
    global roc_curve_data
    roc_curve_data = compute_roc_curve(true_labels, predicted_probs)

@app.get("/api/metrics", response_model=MetricsResponse)
def get_metrics(threshold: float = Query(0.5, ge=0.0, le=1.0)):
    # Use precomputed ROC curve
    
    # Compute confusion matrix for the given threshold
    confusion_matrix = compute_confusion_matrix(true_labels, predicted_probs, threshold)
    
    # Compute current TPR/FPR at the threshold
    current_metrics = compute_metrics_at_threshold(true_labels, predicted_probs, threshold)
    
    return MetricsResponse(
        threshold=threshold,
        roc_curve=roc_curve_data,
        confusion_matrix=confusion_matrix,
        current_metrics=current_metrics
    )

@app.get("/")
def read_root():
    return {"message": "FastAPI backend is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 