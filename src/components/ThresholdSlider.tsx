import React from "react";
import { Slider, Box, Card, CardContent, Typography } from "@mui/material";

interface ThresholdSliderProps {
  value: number;
  onChange: (value: number) => void;
}

const ThresholdSlider: React.FC<ThresholdSliderProps> = ({ value, onChange }) => {
  const handleValueChange = (event: Event, newValue: number | number[]) => {
    onChange(newValue as number);
  };

  return (
    <Card sx={{ minHeight: 200 /* adjust to match ThresholdInfo's height */ }}>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Adjust Threshold
        </Typography>
        <Box
          sx={{
            width: '100%',
            maxWidth: 600,
            // You can increase the minHeight if you want even more space
            minHeight: 80,
            display: 'flex',
            alignItems: 'center',
          }}
        >
          <Slider
            value={value}
            min={0}
            max={1}
            step={0.01}
            onChange={handleValueChange}
            marks={[
              { value: 0, label: '0.00' },
              { value: 0.25, label: '0.25' },
              { value: 0.5, label: '0.50' },
              { value: 0.75, label: '0.75' },
              { value: 1, label: '1.00' },
            ]}
            valueLabelDisplay="auto"
          />
        </Box>
      </CardContent>
    </Card>
  );
};

export default ThresholdSlider;