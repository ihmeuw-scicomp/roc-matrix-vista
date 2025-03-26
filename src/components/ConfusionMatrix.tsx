
import React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { cn } from "@/lib/utils";
import { ConfusionMatrixData } from "@/types";

interface ConfusionMatrixProps {
  data: ConfusionMatrixData;
  className?: string;
  isLoading?: boolean;
}

const ConfusionMatrix: React.FC<ConfusionMatrixProps> = ({
  data,
  className,
  isLoading = false,
}) => {
  const { TP, FP, TN, FN } = data;
  const total = TP + FP + TN + FN;
  
  // Calculate percentages for visual representation
  const tpPercent = Math.round((TP / total) * 100);
  const fpPercent = Math.round((FP / total) * 100);
  const tnPercent = Math.round((TN / total) * 100);
  const fnPercent = Math.round((FN / total) * 100);
  
  // Derived metrics
  const accuracy = ((TP + TN) / total * 100).toFixed(1);
  const precision = (TP / (TP + FP) * 100).toFixed(1);
  const recall = (TP / (TP + FN) * 100).toFixed(1);
  const f1Score = (2 * TP / (2 * TP + FP + FN) * 100).toFixed(1);

  return (
    <Card className={cn("overflow-hidden transition-custom animate-fade-up", className)}>
      <CardHeader className="pb-2">
        <CardTitle className="text-lg font-medium">Confusion Matrix</CardTitle>
      </CardHeader>
      <CardContent>
        <div className={cn("transition-opacity duration-300", {
          "opacity-50": isLoading
        })}>
          <div className="grid grid-cols-2 border border-border rounded-lg overflow-hidden">
            {/* Header Row */}
            <div className="col-span-2 grid grid-cols-2 text-xs text-center py-2 bg-muted/50">
              <div className="col-span-1"></div>
              <div className="grid grid-cols-2">
                <div>Predicted Positive</div>
                <div>Predicted Negative</div>
              </div>
            </div>
            
            {/* True Positive and False Negative */}
            <div className="grid grid-cols-3 border-t border-border">
              <div className="col-span-1 flex items-center justify-center bg-muted/50 text-xs py-3 px-2">
                <div className="-rotate-90 whitespace-nowrap">Actual Positive</div>
              </div>
              <div className="col-span-2 grid grid-cols-2">
                <div className="p-4 border-l border-r border-border flex flex-col justify-between items-center">
                  <div className="text-sm font-medium">True Positive</div>
                  <div className="text-2xl font-semibold">{TP}</div>
                  <div className="text-xs text-muted-foreground">{tpPercent}%</div>
                </div>
                <div className="p-4 flex flex-col justify-between items-center">
                  <div className="text-sm font-medium">False Negative</div>
                  <div className="text-2xl font-semibold">{FN}</div>
                  <div className="text-xs text-muted-foreground">{fnPercent}%</div>
                </div>
              </div>
            </div>
            
            {/* False Positive and True Negative */}
            <div className="grid grid-cols-3 border-t border-border">
              <div className="col-span-1 flex items-center justify-center bg-muted/50 text-xs py-3 px-2">
                <div className="-rotate-90 whitespace-nowrap">Actual Negative</div>
              </div>
              <div className="col-span-2 grid grid-cols-2">
                <div className="p-4 border-l border-r border-border flex flex-col justify-between items-center">
                  <div className="text-sm font-medium">False Positive</div>
                  <div className="text-2xl font-semibold">{FP}</div>
                  <div className="text-xs text-muted-foreground">{fpPercent}%</div>
                </div>
                <div className="p-4 flex flex-col justify-between items-center">
                  <div className="text-sm font-medium">True Negative</div>
                  <div className="text-2xl font-semibold">{TN}</div>
                  <div className="text-xs text-muted-foreground">{tnPercent}%</div>
                </div>
              </div>
            </div>
          </div>
          
          <div className="grid grid-cols-4 gap-4 mt-4">
            <div className="bg-card p-3 rounded-lg border">
              <div className="text-xs text-muted-foreground">Accuracy</div>
              <div className="text-lg font-medium">{accuracy}%</div>
            </div>
            <div className="bg-card p-3 rounded-lg border">
              <div className="text-xs text-muted-foreground">Precision</div>
              <div className="text-lg font-medium">{precision}%</div>
            </div>
            <div className="bg-card p-3 rounded-lg border">
              <div className="text-xs text-muted-foreground">Recall</div>
              <div className="text-lg font-medium">{recall}%</div>
            </div>
            <div className="bg-card p-3 rounded-lg border">
              <div className="text-xs text-muted-foreground">F1 Score</div>
              <div className="text-lg font-medium">{f1Score}%</div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default ConfusionMatrix;
