# ROC Matrix Vista Backend

This is the FastAPI backend for the ROC Matrix Vista application, which provides data processing for ROC curves and confusion matrices.

## Features

- Generates synthetic probability and label data for demo purposes
- Calculates ROC curve points (TPR-FPR pairs) for different thresholds
- Computes confusion matrix metrics (TP, FP, TN, FN) for a given threshold
- Provides exact TPR/FPR metrics at the specified threshold
- Optimized implementation with precomputed ROC data

## API Endpoints

- `GET /api/metrics` - Get ROC curve data and confusion matrix with default threshold (0.5)
- `GET /api/metrics?threshold=<value>` - Get ROC curve data and confusion matrix for a specific threshold

## Response Format

```json
{
  "threshold": 0.5,
  "roc_curve": [
    { "threshold": 0.0, "tpr": 1.0, "fpr": 1.0 },
    { "threshold": 0.1, "tpr": 0.98, "fpr": 0.85 },
    ...
  ],
  "confusion_matrix": {
    "TP": 250,
    "FP": 120,
    "TN": 580,
    "FN": 50
  },
  "current_metrics": {
    "tpr": 0.833,
    "fpr": 0.171
  }
}
```

## Requirements

- Python 3.9+
- FastAPI
- Uvicorn
- NumPy

## Setup and Installation

1. Install dependencies:
   ```
   pip install -e .
   ```

2. Run the server:
   ```
   python -m backend.main
   ```
   
   Alternatively, you can use Uvicorn directly:
   ```
   uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
   ```

3. The server will be available at http://localhost:8000

## API Documentation

FastAPI automatically generates interactive API documentation:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Development

- The backend generates synthetic data on startup for demo purposes
- The ROC curve is precomputed once on startup for performance
- Confusion matrix data is computed on-the-fly for each threshold request 