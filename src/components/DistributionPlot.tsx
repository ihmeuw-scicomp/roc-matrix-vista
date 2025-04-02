import React from 'react';
import { Card, CardHeader, CardContent, Skeleton, Typography, Box } from '@mui/material';
import { ResponsiveContainer, ComposedChart, Area, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';

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

const DistributionPlot: React.FC<DistributionPlotProps> = ({ distributionData, threshold, loading, error }) => {
  if (error) {
    return (
      <Card>
        <CardHeader title="Score Distribution" />
        <CardContent>
          <Typography color="error">Failed to load distribution data</Typography>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader title="Score Distribution" />
      <CardContent>
        {loading ? (
          <Skeleton variant="rectangular" height={300} animation="wave" />
        ) : (
          <Box sx={{ height: 300, width: '100%' }}>
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart
                data={distributionData}
                margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="score" 
                  type="number" 
                  domain={[0, 1]} 
                  tickCount={11}
                  label={{ value: 'Prediction Score', position: 'insideBottom', offset: -5 }}
                />
                <YAxis 
                  label={{ value: 'Frequency', angle: -90, position: 'insideLeft' }}
                />
                <Tooltip />
                <Legend />
                <Area 
                  type="monotone" 
                  dataKey="positives" 
                  fill="#8884d8" 
                  stroke="#8884d8" 
                  name="Positive Class"
                />
                <Area 
                  type="monotone" 
                  dataKey="negatives" 
                  fill="#82ca9d" 
                  stroke="#82ca9d"
                  name="Negative Class" 
                />
                <Line 
                  type="monotone" 
                  dataKey={() => threshold}
                  stroke="red"
                  strokeWidth={2}
                  dot={false}
                  activeDot={false}
                  name="Threshold"
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
