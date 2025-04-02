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
  current_metrics: {
    tpr: number;
    fpr: number;
  };
}

export interface DistributionBin {
  bin_start: number;
  bin_end: number;
  count: number;
}

export interface WorkloadEstimation {
  predicted_positives: number;
  predicted_negatives: number;
  expected_true_positives: number;
  expected_false_positives: number;
  expected_missed_relevant: number;
  total_articles: number;
}

export interface ExtendedMetricsResponse {
  distribution_data: DistributionBin[];
  workload_estimation: WorkloadEstimation;
  threshold: number;
  analysis_id: string;
}
