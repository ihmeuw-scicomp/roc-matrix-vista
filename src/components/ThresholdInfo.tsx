import React from 'react';
import { Card, CardContent, Typography } from '@mui/material';

interface ThresholdInfoProps {
  threshold: number;
  tpr?: number;
  fpr?: number;
}

const ThresholdInfo: React.FC<ThresholdInfoProps> = ({ threshold, tpr, fpr }) => {
  return (
    <Card>
      <CardContent>
        <Typography variant="h6">Threshold Impact</Typography>
        <Typography>Threshold:</Typography>
        <Typography>{threshold.toFixed(2)}</Typography>
        <Typography>Current Point:</Typography>
        <Typography>TPR</Typography>
        <Typography>{tpr !== undefined ? tpr.toFixed(3) : 'Loading...'}</Typography>
        <Typography>FPR</Typography>
        <Typography>{fpr !== undefined ? fpr.toFixed(3) : 'Loading...'}</Typography>
        <Typography>Understanding the threshold:</Typography>
        <Typography variant="body2" color="text.secondary">
          A higher threshold value means the model requires more confidence to classify a sample as positive, resulting in fewer false positives but more false negatives. A lower threshold will classify more samples as positive, increasing true positives but also false positives.
        </Typography>
      </CardContent>
    </Card>
  );
};

export default ThresholdInfo; 