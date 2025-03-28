import React from "react";
import {
  Card,
  CardContent,
  CardHeader,
  Typography,
  Box,
  Paper,
  Grid,
} from "@mui/material";
import { green, red } from "@mui/material/colors"; // Import Material-UI colors
import { ConfusionMatrixData } from "@/types";

interface ConfusionMatrixProps {
  data: ConfusionMatrixData;
  className?: string;
  isLoading?: boolean;
}

const ConfusionMatrix: React.FC<ConfusionMatrixProps> = ({
  data,
  className,
  isLoading = false,
}) => {
  // Map backend properties to frontend properties
  const TP = data.true_positives;
  const FP = data.false_positives;
  const TN = data.true_negatives;
  const FN = data.false_negatives;
  const total = TP + FP + TN + FN;

  // Calculate percentages for visual representation
  const tpPercent = Math.round((TP / total) * 100);
  const fnPercent = Math.round((FN / total) * 100);
  const fpPercent = Math.round((FP / total) * 100);
  const tnPercent = Math.round((TN / total) * 100);

  // Derived metrics (using values from the API response if available)
  const accuracy = data.accuracy ? (data.accuracy * 100).toFixed(1) : (((TP + TN) / total) * 100).toFixed(1);
  const precision = data.precision ? (data.precision * 100).toFixed(1) : ((TP / (TP + FP)) * 100).toFixed(1);
  const recall = data.recall ? (data.recall * 100).toFixed(1) : ((TP / (TP + FN)) * 100).toFixed(1);
  const f1Score = data.f1_score ? (data.f1_score * 100).toFixed(1) : ((2 * TP / (2 * TP + FP + FN)) * 100).toFixed(1);

  // Define color shades for mapping percentages
  const shades = ['50', '100', '200', '300', '400', '500', '600', '700', '800', '900'];

  // Function to determine background color based on cell type and percentage
  const getBackgroundColor = (type: string, percent: number) => {
    const index = Math.min(Math.floor(percent / 10), 9); // Map 0-100% to index 0-9
    const shade = shades[index];
    return type === 'TP' || type === 'TN' ? green[shade] : red[shade];
  };

  // Function to determine text color based on background shade
  const getTextColor = (percent: number) => {
    const index = Math.min(Math.floor(percent / 10), 9);
    return index <= 4 ? 'text.primary' : 'common.white'; // Dark text for light backgrounds, white for dark
  };

  return (
    <Card
      className={className}
      sx={{ overflow: "hidden", transition: "all 0.3s" }}
    >
      <CardHeader
        title={<Typography variant="h6">Confusion Matrix</Typography>}
        sx={{ pb: 1 }}
      />
      <CardContent>
        <Box sx={{ opacity: isLoading ? 0.5 : 1, transition: "opacity 0.3s" }}>
          <Paper variant="outlined" sx={{ p: 2, mb: 2 }}>
            {/* Predicted Labels */}
            <Box display="flex" alignItems="center" mb={2}>
              <Box width={100} /> {/* Placeholder for top-left empty space */}
              <Box flex={1} textAlign="center">
                <Typography variant="subtitle2">
                  Predicted Positive
                </Typography>
              </Box>
              <Box flex={1} textAlign="center">
                <Typography variant="subtitle2">
                  Predicted Negative
                </Typography>
              </Box>
            </Box>

            {/* Actual Positive Row */}
            <Box display="flex" width="100%" mb={2}>
              <Box
                width={100}
                display="flex"
                alignItems="center"
                justifyContent="center"
                sx={{ transform: "rotate(-90deg)" }}
              >
                <Typography variant="subtitle2">Actual Positive</Typography>
              </Box>
              {/* True Positive Cell */}
              <Box
                flex={1}
                border={1}
                borderColor="divider"
                p={2}
                textAlign="center"
                sx={{ backgroundColor: getBackgroundColor('TP', tpPercent) }}
              >
                <Typography
                  variant="body2"
                  sx={{ color: getTextColor(tpPercent) }}
                >
                  True Positive
                </Typography>
                <Typography
                  variant="h4"
                  sx={{ color: getTextColor(tpPercent) }}
                >
                  {TP}
                </Typography>
                <Typography
                  variant="caption"
                  sx={{ color: getTextColor(tpPercent) }}
                >
                  {tpPercent}%
                </Typography>
              </Box>
              {/* False Negative Cell */}
              <Box
                flex={1}
                border={1}
                borderColor="divider"
                p={2}
                textAlign="center"
                sx={{ backgroundColor: getBackgroundColor('FN', fnPercent) }}
              >
                <Typography
                  variant="body2"
                  sx={{ color: getTextColor(fnPercent) }}
                >
                  False Negative
                </Typography>
                <Typography
                  variant="h4"
                  sx={{ color: getTextColor(fnPercent) }}
                >
                  {FN}
                </Typography>
                <Typography
                  variant="caption"
                  sx={{ color: getTextColor(fnPercent) }}
                >
                  {fnPercent}%
                </Typography>
              </Box>
            </Box>

            {/* Actual Negative Row */}
            <Box display="flex" width="100%">
              <Box
                width={100}
                display="flex"
                alignItems="center"
                justifyContent="center"
                sx={{ transform: "rotate(-90deg)" }}
              >
                <Typography variant="subtitle2">Actual Negative</Typography>
              </Box>
              {/* False Positive Cell */}
              <Box
                flex={1}
                border={1}
                borderColor="divider"
                p={2}
                textAlign="center"
                sx={{ backgroundColor: getBackgroundColor('FP', fpPercent) }}
              >
                <Typography
                  variant="body2"
                  sx={{ color: getTextColor(fpPercent) }}
                >
                  False Positive
                </Typography>
                <Typography
                  variant="h4"
                  sx={{ color: getTextColor(fpPercent) }}
                >
                  {FP}
                </Typography>
                <Typography
                  variant="caption"
                  sx={{ color: getTextColor(fpPercent) }}
                >
                  {fpPercent}%
                </Typography>
              </Box>
              {/* True Negative Cell */}
              <Box
                flex={1}
                border={1}
                borderColor="divider"
                p={2}
                textAlign="center"
                sx={{ backgroundColor: getBackgroundColor('TN', tnPercent) }}
              >
                <Typography
                  variant="body2"
                  sx={{ color: getTextColor(tnPercent) }}
                >
                  True Negative
                </Typography>
                <Typography
                  variant="h4"
                  sx={{ color: getTextColor(tnPercent) }}
                >
                  {TN}
                </Typography>
                <Typography
                  variant="caption"
                  sx={{ color: getTextColor(tnPercent) }}
                >
                  {tnPercent}%
                </Typography>
              </Box>
            </Box>
          </Paper>

          {/* Derived Metrics on One Line */}
          <Grid container spacing={2}>
            <Box sx={{ flex: 1 }}>
              <Paper variant="outlined" sx={{ p: 1.5, textAlign: "center" }}>
                <Typography variant="caption" color="text.secondary">
                  Accuracy
                </Typography>
                <Typography variant="h6">{accuracy}%</Typography>
              </Paper>
            </Box>
            <Box sx={{ flex: 1 }}>
              <Paper variant="outlined" sx={{ p: 1.5, textAlign: "center" }}>
                <Typography variant="caption" color="text.secondary">
                  Precision
                </Typography>
                <Typography variant="h6">{precision}%</Typography>
              </Paper>
            </Box>
            <Box sx={{ flex: 1 }}>
              <Paper variant="outlined" sx={{ p: 1.5, textAlign: "center" }}>
                <Typography variant="caption" color="text.secondary">
                  Recall
                </Typography>
                <Typography variant="h6">{recall}%</Typography>
              </Paper>
            </Box>
            <Box sx={{ flex: 1 }}>
              <Paper variant="outlined" sx={{ p: 1.5, textAlign: "center" }}>
                <Typography variant="caption" color="text.secondary">
                  F1 Score
                </Typography>
                <Typography variant="h6">{f1Score}%</Typography>
              </Paper>
            </Box>
          </Grid>
        </Box>
      </CardContent>
    </Card>
  );
};

export default ConfusionMatrix;