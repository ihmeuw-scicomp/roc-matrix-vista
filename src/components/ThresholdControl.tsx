
import React from "react";
import { Slider } from "@/components/ui/slider";
import { Card, CardContent } from "@/components/ui/card";
import { cn } from "@/lib/utils";

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
  const handleValueChange = (values: number[]) => {
    onChange(values[0]);
  };

  return (
    <Card className={cn("overflow-hidden transition-custom animate-fade-up", className)}>
      <CardContent className="pt-6">
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-xs uppercase text-muted-foreground font-medium tracking-wide">
                Threshold
              </div>
              <div className="font-medium text-xl">
                {threshold.toFixed(2)}
              </div>
            </div>
            <div className="text-sm text-muted-foreground">
              Adjust to see results
            </div>
          </div>
          
          <Slider
            defaultValue={[threshold]}
            min={0}
            max={1}
            step={0.01}
            onValueChange={handleValueChange}
            className="cursor-pointer"
          />
          
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>0.00</span>
            <span>0.50</span>
            <span>1.00</span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default ThresholdControl;
