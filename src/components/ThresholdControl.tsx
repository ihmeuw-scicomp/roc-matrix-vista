
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
            />
          </Box>
          
          <Box display="flex" justifyContent="space-between">
            <Typography variant="caption" color="text.secondary">0.00</Typography>
            <Typography variant="caption" color="text.secondary">0.50</Typography>
            <Typography variant="caption" color="text.secondary">1.00</Typography>
          </Box>
        </Box>
      </CardContent>
    </Card>
  );
};

export default ThresholdControl;
