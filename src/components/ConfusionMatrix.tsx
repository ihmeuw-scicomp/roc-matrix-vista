// import React from "react";
import React, { useEffect } from "react";
import {
  Card,
  CardContent,
  CardHeader,
  Typography,
  Box,
  Paper,
  Grid,
  Skeleton
} from "@mui/material";
import { green, red } from "@mui/material/colors";
import { ConfusionMatrixData } from "@/types";
import { 
  processConfusionMatrixData, 
  getConfusionMatrixCellColor, 
  getTextColorForCell 
} from "@/lib/utils";

interface ConfusionMatrixProps {
  data: ConfusionMatrixData;
  className?: string;
  isLoading?: boolean;
}

/**
 * Matrix cell component for confusion matrix
 */
const MatrixCell: React.FC<{
  label: string;
  count: number;
  percent: number;
  type: string;
}> = ({ label, count, percent, type }) => {
  return (
    <Box
      flex={1}
      border={1}
      borderColor="divider"
      p={2}
      textAlign="center"
      sx={{ backgroundColor: getConfusionMatrixCellColor(type, percent) }}
    >
      <Typography variant="body2" sx={{ color: getTextColorForCell(percent) }}>
        {label}
      </Typography>
      <Typography variant="h4" sx={{ color: getTextColorForCell(percent) }}>
        {count}
      </Typography>
      <Typography variant="caption" sx={{ color: getTextColorForCell(percent) }}>
        {percent}%
      </Typography>
    </Box>
  );
};

/**
 * Metric display component for showing evaluation metrics
 */
const MetricDisplay: React.FC<{
  label: string;
  value: string;
}> = ({ label, value }) => {
  return (
    <Box sx={{ flex: 1 }}>
      <Paper variant="outlined" sx={{ p: 1.5, textAlign: "center" }}>
        <Typography variant="caption" color="text.secondary">{label}</Typography>
        <Typography variant="h6">{value}%</Typography>
      </Paper>
    </Box>
  );
};

const ConfusionMatrix: React.FC<ConfusionMatrixProps> = ({
  data,
  className,
  isLoading = false,
}) => {
  // Process data once to get all derived values
  const processedData = processConfusionMatrixData(data);
  
  if (isLoading) {
    return (
      <Card className={className}>
        <CardHeader title={<Skeleton variant="text" width="60%" />} />
        <CardContent>
          <Skeleton variant="rectangular" height={300} />
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className={className} sx={{ overflow: "hidden", transition: "all 0.3s" }}>
      <CardHeader
        title={
          <>
            <Typography variant="h6">Confusion Matrix</Typography>
            <Typography variant="caption">Threshold: {processedData.threshold}</Typography>
          </>
        }
        sx={{ pb: 1 }}
      />
      <CardContent>
        <Paper variant="outlined" sx={{ p: 2, mb: 2 }}>
          {/* Predicted Labels */}
          <Box display="flex" alignItems="center" mb={2}>
            <Box width={100} />
            <Box flex={1} textAlign="center">
              <Typography variant="subtitle2">Predicted Positive</Typography>
            </Box>
            <Box flex={1} textAlign="center">
              <Typography variant="subtitle2">Predicted Negative</Typography>
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
            
            <MatrixCell 
              label="True Positive" 
              count={processedData.true_positives} 
              percent={processedData.tpPercent} 
              type="TP" 
            />
            
            <MatrixCell 
              label="False Negative" 
              count={processedData.false_negatives} 
              percent={processedData.fnPercent} 
              type="FN" 
            />
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
            
            <MatrixCell 
              label="False Positive" 
              count={processedData.false_positives} 
              percent={processedData.fpPercent} 
              type="FP" 
            />
            
            <MatrixCell 
              label="True Negative" 
              count={processedData.true_negatives} 
              percent={processedData.tnPercent} 
              type="TN" 
            />
          </Box>
        </Paper>
        
        <Grid container spacing={2}>
          <MetricDisplay label="Accuracy" value={processedData.accuracyFormatted} />
          <MetricDisplay label="Precision" value={processedData.precisionFormatted} />
          <MetricDisplay label="Recall" value={processedData.recallFormatted} />
          <MetricDisplay label="F1 Score" value={processedData.f1ScoreFormatted} />
        </Grid>
      </CardContent>
    </Card>
  );
};

export default ConfusionMatrix;
