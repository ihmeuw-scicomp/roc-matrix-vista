import React from 'react';
import { 
  Card, CardHeader, CardContent, Typography, CircularProgress, Box,
  Paper, Divider
} from '@mui/material';

export interface WorkloadEstimation {
  predicted_positives: number;
  predicted_negatives: number;
  expected_true_positives: number;
  expected_false_positives: number;
  expected_missed_relevant: number;
  total_articles: number;
}

interface WorkloadSummaryProps {
  workloadEstimation?: WorkloadEstimation;
  threshold: number;
  loading: boolean;
  error: boolean;
}

const WorkloadSummary: React.FC<WorkloadSummaryProps> = ({ 
  workloadEstimation, 
  threshold, 
  loading, 
  error 
}) => {
  if (error) {
    return (
      <Card>
        <CardHeader title="Workload Summary" />
        <CardContent>
          <Typography color="error">Failed to load workload estimation data</Typography>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader 
        title="Workload Summary" 
        subheader={`Threshold = ${threshold.toFixed(2)}`}
      />
      <CardContent>
        {loading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
            <CircularProgress />
          </Box>
        ) : workloadEstimation ? (
          <Paper variant="outlined" sx={{ p: 2 }}>
            {/* Predicted Positives Section */}
            <Typography variant="subtitle1" fontWeight="medium">
              Predicted Positive: ~{workloadEstimation.predicted_positives.toLocaleString()} articles
            </Typography>
            <Box sx={{ pl: 3, mb: 1.5 }}>
              <Typography variant="body2">
                • ~{workloadEstimation.expected_true_positives.toLocaleString()} are likely relevant
              </Typography>
              <Typography variant="body2">
                • ~{workloadEstimation.expected_false_positives.toLocaleString()} are likely not relevant
              </Typography>
            </Box>
            
            {/* Predicted Negatives Section */}
            <Typography variant="subtitle1" fontWeight="medium">
              Predicted Negative: ~{workloadEstimation.predicted_negatives.toLocaleString()} articles
            </Typography>
            <Box sx={{ pl: 3, mb: 2 }}>
              <Typography variant="body2">
                • ~{workloadEstimation.expected_missed_relevant.toLocaleString()} missed relevant articles (FN)
              </Typography>
              <Typography variant="body2">
                • ~{(workloadEstimation.predicted_negatives - workloadEstimation.expected_missed_relevant).toLocaleString()} true negatives
              </Typography>
            </Box>
            
            <Divider sx={{ my: 2 }} />
            
            {/* Interpretation Section */}
            <Typography variant="subtitle1" fontWeight="medium">
              Interpretation:
            </Typography>
            <Typography variant="body2" sx={{ mt: 1 }}>
              "At threshold={threshold.toFixed(2)}, you only need to screen {workloadEstimation.predicted_positives.toLocaleString()} articles 
              ({((workloadEstimation.predicted_positives / workloadEstimation.total_articles) * 100).toFixed(1)}% of total),
              catching ~{workloadEstimation.expected_true_positives.toLocaleString()} relevant ones but missing 
              ~{workloadEstimation.expected_missed_relevant.toLocaleString()} that are automatically excluded."
            </Typography>
          </Paper>
        ) : (
          <Typography>No workload estimation data available</Typography>
        )}
      </CardContent>
    </Card>
  );
};

export default WorkloadSummary;