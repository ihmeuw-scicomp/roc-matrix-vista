import React from "react";
import { Card, CardContent, Typography, Slider, Box } from "@mui/material";

interface ThresholdControlProps {
  threshold: number;
  onChange: (value: number) => void;
  className?: string;
}

const ThresholdControl: React.FC<ThresholdControlProps> = ({
  threshold,
  onChange,
  className,
}) => {
  const handleValueChange = (event: Event, value: number | number[]) => {
    onChange(value as number);
  };

  // Define marks for the slider
  const marks = [
    { value: 0, label: '0.00' },
    { value: 0.25, label: '0.25' },
    { value: 0.5, label: '0.50' },
    { value: 0.75, label: '0.75' },
    { value: 1, label: '1.00' },
  ];

  return (
    <Card className={className} sx={{ overflow: "hidden", transition: "all 0.3s" }}>
      <CardContent sx={{ pt: 2 }}>
        <Box sx={{ mb: 3 }}>
          <Box display="flex" justifyContent="space-between" alignItems="center">
            <Box>
              <Typography variant="caption" color="text.secondary" sx={{ 
                textTransform: "uppercase", 
                fontWeight: "medium",
                letterSpacing: "0.05em"
              }}>
                Threshold
              </Typography>
              <Typography variant="h5">
                {threshold.toFixed(2)}
              </Typography>
            </Box>
            <Typography variant="body2" color="text.secondary">
              Adjust to see results
            </Typography>
          </Box>
          
          <Box sx={{ mt: 3, mb: 1 }}>
            <Slider
              value={threshold}
              min={0}
              max={1}
              step={0.01}
              onChange={handleValueChange}
              marks={marks}
              valueLabelDisplay="auto"
            />
          </Box>
        </Box>
      </CardContent>
    </Card>
  );
};

export default ThresholdControl;
