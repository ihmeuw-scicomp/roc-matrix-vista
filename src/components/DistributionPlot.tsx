import React from 'react';
import { Card, CardHeader, CardContent, Skeleton, Typography, Box, Alert, CircularProgress } from '@mui/material';
import { ResponsiveContainer, ComposedChart, Area, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ReferenceLine } from 'recharts';

interface DataPoint {
    score: number;
    positives: number;
    negatives: number;
}

interface DistributionPlotProps {
  distributionData: DataPoint[];
  threshold: number;
  loading: boolean;
  error: Error | null | boolean;
}

/**
 * A component that displays the distribution of prediction scores with a threshold line
 */
const DistributionPlot: React.FC<DistributionPlotProps> = ({ 
  distributionData, 
  threshold, 
  loading, 
  error 
}) => {
  const formatTooltip = (value: number) => {
    return value.toFixed(0);
  };

  // Color constants for consistency
  const positiveColor = '#2e7d32'; // Green
  const negativeColor = '#1976d2'; // Blue

  // Show error state if an error occurred
  if (error) {
    return (
      <Card>
        <CardHeader title="Score Distribution" />
        <CardContent>
          <Alert 
            severity="error" 
            sx={{ mb: 2 }}
          >
            Failed to load distribution data
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
        title="Score Distribution" 
        subheader={`Classification threshold: ${threshold.toFixed(2)}`}
      />
      <CardContent>
        {loading ? (
          <>
            <Box display="flex" justifyContent="center" alignItems="center" height={50} mb={2}>
              <CircularProgress size={24} sx={{ mr: 2 }} />
              <Typography>Loading distribution data...</Typography>
            </Box>
            <Skeleton variant="rectangular" height={300} animation="wave" />
          </>
        ) : distributionData.length === 0 ? (
          <Alert severity="info">
            No distribution data available for this analysis
          </Alert>
        ) : (
          <Box sx={{ height: 370, width: '100%' }}>
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart
                data={distributionData}
                margin={{ top: 30, right: 30, left: 10, bottom: 65 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="score" 
                  type="number" 
                  domain={[0, 1]} 
                  tickCount={11}
                  label={{ 
                    value: 'Prediction Score', 
                    position: 'bottom', 
                    offset: 35,
                    style: { textAnchor: 'middle' }
                  }}
                />
                <YAxis 
                  label={{ value: 'Frequency', angle: -90, position: 'insideLeft', offset: 5 }}
                  tickFormatter={formatTooltip}
                />
                <Tooltip 
                  formatter={formatTooltip}
                  labelFormatter={(label) => `Score: ${Number(label).toFixed(2)}`}
                />
                <Legend 
                  verticalAlign="bottom" 
                  height={46}
                  wrapperStyle={{ 
                    paddingTop: '20px',
                    paddingBottom: '5px',
                    bottom: 0
                  }}
                  iconSize={14}
                  iconType="circle"
                />
                <Area 
                  type="monotone" 
                  dataKey="positives" 
                  fill={positiveColor}
                  stroke={positiveColor}
                  name="Positive Class"
                  isAnimationActive={true}
                  fillOpacity={0.6}
                />
                <Area 
                  type="monotone" 
                  dataKey="negatives" 
                  fill={negativeColor}
                  stroke={negativeColor}
                  name="Negative Class" 
                  isAnimationActive={true}
                  fillOpacity={0.6}
                />
                <ReferenceLine
                  x={threshold}
                  stroke="red"
                  strokeWidth={2}
                  label={{
                    value: `Threshold: ${threshold.toFixed(2)}`,
                    position: 'top',
                    fill: 'red',
                    fontSize: 12,
                    offset: 10,
                    dy: -10
                  }}
                  ifOverflow="extendDomain"
                />
              </ComposedChart>
            </ResponsiveContainer>
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

export default DistributionPlot;
