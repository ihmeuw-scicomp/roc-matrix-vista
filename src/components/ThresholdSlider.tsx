import React from "react";
import { Slider, Box } from "@mui/material";

interface ThresholdSliderProps {
  value: number;
  onChange: (value: number) => void;
}

const ThresholdSlider: React.FC<ThresholdSliderProps> = ({ value, onChange }) => {
  const handleValueChange = (event: Event, newValue: number | number[]) => {
    onChange(newValue as number);
  };

  return (
    <Box sx={{ width: '100%', maxWidth: 600 }}>
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
  );
};

export default ThresholdSlider; 