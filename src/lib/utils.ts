import { clsx, type ClassValue } from "clsx"
import { ConfusionMatrixData } from "@/types";
import { green, red } from "@mui/material/colors";

export function cn(...inputs: ClassValue[]) {
  return clsx(inputs)
}

/**
 * Format a numeric value as a percentage with the specified number of decimal places
 * @param value Value to format (0-1)
 * @param decimalPlaces Number of decimal places to show
 * @returns Formatted percentage string
 */
export const formatPercent = (value: number, decimalPlaces: number = 1): string => {
  return (value * 100).toFixed(decimalPlaces);
};

/**
 * Calculate percentage from a count relative to a total
 * @param count Count value
 * @param total Total value
 * @returns Percentage as a rounded integer
 */
export const calculatePercent = (count: number, total: number): number => {
  return Math.round((count / total) * 100);
};

/**
 * Get background color for confusion matrix cells based on type and percentage
 * @param type Cell type (TP, FP, TN, FN)
 * @param percent Percentage value (0-100)
 * @returns Color from Material UI palette
 */
export const getConfusionMatrixCellColor = (type: string, percent: number): string => {
  const shades = ['50', '100', '200', '300', '400', '500', '600', '700', '800', '900'];
  const index = Math.min(Math.floor(percent / 10), 9);
  const shade = shades[index] as keyof typeof green;
  return type === 'TP' || type === 'TN' ? green[shade] : red[shade];
};

/**
 * Determine text color based on background darkness
 * @param percent Percentage value (0-100)
 * @returns Text color
 */
export const getTextColorForCell = (percent: number): string => {
  const index = Math.min(Math.floor(percent / 10), 9);
  return index <= 4 ? 'text.primary' : 'common.white';
};

/**
 * Process confusion matrix data to include derived metrics
 * @param data Raw confusion matrix data
 * @returns Processed data with additional metrics
 */
export const processConfusionMatrixData = (data: ConfusionMatrixData) => {
  const TP = data.true_positives;
  const FP = data.false_positives;
  const TN = data.true_negatives;
  const FN = data.false_negatives;
  const total = TP + FP + TN + FN;

  return {
    ...data,
    total,
    tpPercent: calculatePercent(TP, total),
    fnPercent: calculatePercent(FN, total),
    fpPercent: calculatePercent(FP, total),
    tnPercent: calculatePercent(TN, total),
    accuracyFormatted: formatPercent(data.accuracy),
    precisionFormatted: formatPercent(data.precision),
    recallFormatted: formatPercent(data.recall),
    f1ScoreFormatted: formatPercent(data.f1_score)
  };
};
