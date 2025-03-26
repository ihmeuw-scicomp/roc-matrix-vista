
import React, { useState, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import Layout from "@/components/Layout";
import ROCCurve from "@/components/ROCCurve";
import ConfusionMatrix from "@/components/ConfusionMatrix";
import ThresholdControl from "@/components/ThresholdControl";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { fetchMetrics } from "@/services/api";
import { ConfusionMatrixData } from "@/types";

const Index = () => {
  const [threshold, setThreshold] = useState(0.5);
  
  // Debounce threshold changes to avoid excessive API calls
  const [debouncedThreshold, setDebouncedThreshold] = useState(threshold);
  
  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedThreshold(threshold);
    }, 100);
    
    return () => clearTimeout(timer);
  }, [threshold]);
  
  // Fetch data using React Query
  const { data, isLoading, isError } = useQuery({
    queryKey: ['metrics', debouncedThreshold],
    queryFn: () => fetchMetrics(debouncedThreshold),
    staleTime: 60000, // 1 minute
    refetchOnWindowFocus: false,
  });
  
  // Default empty confusion matrix for initial render
  const emptyMatrix: ConfusionMatrixData = { TP: 0, FP: 0, TN: 0, FN: 0 };
  
  return (
    <Layout>
      <div className="max-w-6xl mx-auto pt-4 animate-fade-in">
        <Card className="mb-8 overflow-hidden">
          <CardHeader className="pb-2">
            <CardTitle className="text-2xl font-medium">ROC Matrix Vista</CardTitle>
            <CardDescription>
              Visualize and analyze classification performance with ROC curves and confusion matrices
            </CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-muted-foreground">
              Adjust the threshold below to see how it affects true positive rate, false positive rate, 
              and overall model performance metrics in real-time.
            </p>
          </CardContent>
        </Card>
        
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
          <ThresholdControl 
            threshold={threshold} 
            onChange={setThreshold}
            className="lg:col-span-1"
          />
          
          <Card className="lg:col-span-2 overflow-hidden transition-custom animate-fade-up">
            <CardHeader className="pb-2">
              <CardTitle className="text-lg font-medium">Threshold Impact</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-6">
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <div className="text-sm font-medium">Threshold:</div>
                    <div className="text-3xl font-semibold tracking-tight">{threshold.toFixed(2)}</div>
                  </div>
                  <div className="space-y-2">
                    <div className="text-sm font-medium">Current Point:</div>
                    <div className="flex space-x-4">
                      <div>
                        <div className="text-xs text-muted-foreground">TPR</div>
                        <div className="text-xl font-medium">
                          {data?.roc_curve.find(p => Math.abs(p.threshold - threshold) < 0.01)?.tpr.toFixed(3) || "—"}
                        </div>
                      </div>
                      <div>
                        <div className="text-xs text-muted-foreground">FPR</div>
                        <div className="text-xl font-medium">
                          {data?.roc_curve.find(p => Math.abs(p.threshold - threshold) < 0.01)?.fpr.toFixed(3) || "—"}
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
                
                <div className="space-y-2 pt-2">
                  <div className="text-sm font-medium">Understanding the threshold:</div>
                  <p className="text-sm text-muted-foreground">
                    A higher threshold value means the model requires more confidence to classify a sample as positive,
                    resulting in fewer false positives but more false negatives. A lower threshold will classify more
                    samples as positive, increasing true positives but also false positives.
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <ROCCurve 
            rocData={data?.roc_curve || []} 
            currentThreshold={threshold}
            isLoading={isLoading}
          />
          
          <ConfusionMatrix 
            data={data?.confusion_matrix || emptyMatrix}
            isLoading={isLoading}
          />
        </div>
      </div>
    </Layout>
  );
};

export default Index;
