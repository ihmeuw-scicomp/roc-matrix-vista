export interface ROCPoint {
  threshold: number;
  tpr: number;
  fpr: number;
}

export interface ConfusionMatrixData {
  TP: number;
  FP: number;
  TN: number;
  FN: number;
}

export interface MetricsResponse {
  threshold: number;
  roc_curve: ROCPoint[];
  confusion_matrix: ConfusionMatrixData;
  current_metrics: {
    tpr: number;
    fpr: number;
  };
}
