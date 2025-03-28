export interface ROCPoint {
  threshold: number;
  tpr: number;
  fpr: number;
}

export interface ConfusionMatrixData {
  true_positives: number;
  false_positives: number;
  true_negatives: number;
  false_negatives: number;
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  threshold: number;
}

export interface MetricsResponse {
  threshold: number;
  roc_curve: ROCPoint[];
  confusion_matrix: ConfusionMatrixData;
  current_metrics?: {
    tpr: number;
    fpr: number;
  };
}
