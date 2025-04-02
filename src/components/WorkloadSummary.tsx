import React from 'react';
import { 
  Card, CardHeader, CardContent, Typography, CircularProgress, Box,
  Paper, Divider, Alert, Skeleton, Grid
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

/**
 * A component that displays workload estimation data including predicted positives, negatives,
 * and interpretation of the results based on the current threshold
 */
const WorkloadSummary: React.FC<WorkloadSummaryProps> = ({ 
  workloadEstimation, 
  threshold, 
  loading, 
  error 
}) => {
  const formatPercent = (value: number, total: number): string => {
    return `${((value / total) * 100).toFixed(1)}%`;
  };

  // Show error state with helpful message
  if (error) {
    return (
      <Card>
        <CardHeader title="Workload Summary" />
        <CardContent>
          <Alert 
            severity="error" 
            sx={{ mb: 2 }}
          >
            Failed to load workload estimation data
          </Alert>
          <Box display="flex" justifyContent="center">
            <Typography color="text.secondary">
              Try refreshing the page or check your connection.
            </Typography>
          </Box>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader 
        title="Workload Summary" 
        subheader={`Current threshold: ${threshold.toFixed(2)}`}
      />
      <CardContent>
        {loading ? (
          <>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
              <CircularProgress size={20} sx={{ mr: 2 }} />
              <Typography variant="body2">Loading workload estimation...</Typography>
            </Box>
            <Skeleton variant="rectangular" height={200} animation="wave" />
          </>
        ) : !workloadEstimation ? (
          <Alert severity="info">
            No workload estimation data available for this analysis
          </Alert>
        ) : (
          <>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2, mb: 2 }}>
              <Box sx={{ flex: 1, minWidth: '120px' }}>
                <Paper variant="outlined" sx={{ p: 2, height: '100%', bgcolor: 'rgba(46, 125, 50, 0.1)' }}>
                  <Typography variant="h6" color="text.primary" fontWeight="bold" gutterBottom>
                    {workloadEstimation.predicted_positives.toLocaleString()}
                  </Typography>
                  <Typography variant="subtitle2" color="text.primary">
                    Predicted Positives ({formatPercent(workloadEstimation.predicted_positives, workloadEstimation.total_articles)})
                  </Typography>
                </Paper>
              </Box>
              <Box sx={{ flex: 1, minWidth: '120px' }}>
                <Paper variant="outlined" sx={{ p: 2, height: '100%', bgcolor: 'rgba(25, 118, 210, 0.1)' }}>
                  <Typography variant="h6" color="text.primary" fontWeight="bold" gutterBottom>
                    {workloadEstimation.predicted_negatives.toLocaleString()}
                  </Typography>
                  <Typography variant="subtitle2" color="text.primary">
                    Predicted Negatives ({formatPercent(workloadEstimation.predicted_negatives, workloadEstimation.total_articles)})
                  </Typography>
                </Paper>
              </Box>
            </Box>
            
            <Paper variant="outlined" sx={{ p: 2 }}>
              {/* Predicted Positives Section */}
              <Typography variant="subtitle1" fontWeight="bold" color="text.primary" sx={{ color: 'rgba(46, 125, 50, 0.9)' }}>
                Predicted Positive: {workloadEstimation.predicted_positives.toLocaleString()} articles
              </Typography>
              <Box sx={{ pl: 3, mb: 1.5 }}>
                <Typography variant="body2">
                  • {workloadEstimation.expected_true_positives.toLocaleString()} are likely relevant (
                  {formatPercent(workloadEstimation.expected_true_positives, workloadEstimation.predicted_positives)})
                </Typography>
                <Typography variant="body2">
                  • {workloadEstimation.expected_false_positives.toLocaleString()} are likely not relevant (
                  {formatPercent(workloadEstimation.expected_false_positives, workloadEstimation.predicted_positives)})
                </Typography>
              </Box>
              
              {/* Predicted Negatives Section */}
              <Typography variant="subtitle1" fontWeight="bold" color="text.primary" sx={{ color: 'rgba(25, 118, 210, 0.9)' }}>
                Predicted Negative: {workloadEstimation.predicted_negatives.toLocaleString()} articles
              </Typography>
              <Box sx={{ pl: 3, mb: 2 }}>
                <Typography variant="body2">
                  • {workloadEstimation.expected_missed_relevant.toLocaleString()} missed relevant articles (
                  {formatPercent(workloadEstimation.expected_missed_relevant, workloadEstimation.predicted_negatives)})
                </Typography>
                <Typography variant="body2">
                  • {(workloadEstimation.predicted_negatives - workloadEstimation.expected_missed_relevant).toLocaleString()} true negatives (
                  {formatPercent(workloadEstimation.predicted_negatives - workloadEstimation.expected_missed_relevant, workloadEstimation.predicted_negatives)})
                </Typography>
              </Box>
              
              <Divider sx={{ my: 2 }} />
              
              {/* Interpretation Section */}
              <Typography variant="subtitle1" fontWeight="bold" color="text.primary">
                Interpretation:
              </Typography>
              <Typography variant="body2" sx={{ mt: 1 }}>
                At threshold={threshold.toFixed(2)}, you need to screen only {formatPercent(workloadEstimation.predicted_positives, workloadEstimation.total_articles)} of the total articles 
                ({workloadEstimation.predicted_positives.toLocaleString()} out of {workloadEstimation.total_articles.toLocaleString()}). 
                This would catch approximately {workloadEstimation.expected_true_positives.toLocaleString()} relevant articles 
                but would miss about {workloadEstimation.expected_missed_relevant.toLocaleString()} that are automatically excluded.
              </Typography>
            </Paper>
          </>
        )}
      </CardContent>
    </Card>
  );
};

export default WorkloadSummary;