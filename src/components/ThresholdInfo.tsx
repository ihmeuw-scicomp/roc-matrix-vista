import React from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box
} from '@mui/material';

interface ThresholdInfoProps {
  threshold: number;
  tpr?: number;
  fpr?: number;
}

const ThresholdInfo: React.FC<ThresholdInfoProps> = ({ threshold, tpr, fpr }) => {
  return (
    <Card>
      <CardContent>
        {/* Title */}
        <Typography variant="h6" gutterBottom>
          Threshold Impact
        </Typography>

        {/* Threshold & Current Point */}
        <Box
          display="flex"
          justifyContent="space-between"
          alignItems="flex-start"
          mb={2}
        >
          {/* Threshold Box */}
          <Box>
            <Typography variant="subtitle2" color="text.secondary">
              Threshold:
            </Typography>
            <Typography variant="h3">
              {threshold.toFixed(2)}
            </Typography>
          </Box>

          {/* Current Point Box */}
          <Box textAlign="right">
            <Typography variant="subtitle2" color="text.secondary">
              Current Point:
            </Typography>
            <Box display="flex" justifyContent="flex-end" mt={0.5}>
              <Box mr={3} textAlign="center">
                <Typography variant="subtitle2" color="text.secondary">
                  TPR
                </Typography>
                <Typography variant="h5">
                  {tpr !== undefined ? tpr.toFixed(3) : '...'}
                </Typography>
              </Box>
              <Box textAlign="center">
                <Typography variant="subtitle2" color="text.secondary">
                  FPR
                </Typography>
                <Typography variant="h5">
                  {fpr !== undefined ? fpr.toFixed(3) : '...'}
                </Typography>
              </Box>
            </Box>
          </Box>
        </Box>

        {/* Explanation */}
        <Typography variant="subtitle2" gutterBottom>
          Understanding the threshold:
        </Typography>
        <Typography variant="body2" color="text.secondary">
          A higher threshold value means the model requires more confidence to classify a sample as positive,
          resulting in fewer false positives but more false negatives. A lower threshold will classify more samples
          as positive, increasing true positives but also false positives.
        </Typography>
      </CardContent>
    </Card>
  );
};

export default ThresholdInfo;