
import { MetricsResponse } from "@/types";

// Generate ROC curve data points
const generateROCCurve = () => {
  const points = [];
  // Generate 100 points for the ROC curve
  for (let i = 0; i <= 100; i++) {
    const threshold = i / 100;
    // Create realistic ROC curve shape
    // In a real ROC curve, lower thresholds typically have higher TPR and FPR
    let tpr, fpr;
    
    if (threshold <= 0.2) {
      // For low thresholds, both TPR and FPR are high
      tpr = 1 - (threshold * 0.3);
      fpr = 1 - (threshold * 1.7);
    } else if (threshold <= 0.5) {
      // Mid-range thresholds
      tpr = 0.94 - (threshold - 0.2) * 0.75;
      fpr = 0.66 - (threshold - 0.2) * 1.2;
    } else {
      // Higher thresholds
      tpr = 0.7 - (threshold - 0.5) * 1.4;
      fpr = 0.3 - (threshold - 0.5) * 0.6;
    }
    
    // Ensure values stay within valid range [0,1]
    tpr = Math.max(0, Math.min(1, tpr));
    fpr = Math.max(0, Math.min(1, fpr));
    
    points.push({ threshold, tpr, fpr });
  }
  return points;
};

// Generate confusion matrix based on threshold
const generateConfusionMatrix = (threshold: number) => {
  const total = 1000; // Total sample size
  
  // Calculate values based on threshold to create realistic dependencies
  // As threshold increases: TP and FP decrease, TN and FN increase
  const tpRate = Math.max(0, 0.9 - threshold * 0.8); // TPR decreases as threshold increases
  const fpRate = Math.max(0, 0.8 - threshold * 0.76); // FPR decreases as threshold increases
  
  const positives = Math.round(total * 0.4); // 40% are actual positives
  const negatives = total - positives; // 60% are actual negatives
  
  const TP = Math.round(positives * tpRate);
  const FN = positives - TP;
  const FP = Math.round(negatives * fpRate);
  const TN = negatives - FP;
  
  return { TP, FP, TN, FN };
};

// Mock API response with threshold parameter
export const getMockMetrics = (threshold: number = 0.5): MetricsResponse => {
  return {
    threshold,
    roc_curve: generateROCCurve(),
    confusion_matrix: generateConfusionMatrix(threshold)
  };
};
