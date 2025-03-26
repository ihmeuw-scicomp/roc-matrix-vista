
import React, { useState, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import MuiLayout from "@/components/MuiLayout";
import ROCCurve from "@/components/ROCCurve";
import ConfusionMatrix from "@/components/ConfusionMatrix";
import ThresholdControl from "@/components/ThresholdControl";
import { 
  Card, 
  CardContent, 
  CardHeader, 
  Typography, 
  Grid, 
  Box, 
  Paper, 
  Divider 
} from "@mui/material";
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
    <MuiLayout>
      <Box sx={{ pt: 2 }}>
        <Card sx={{ mb: 4 }}>
          <CardHeader 
            title="ROC Matrix Vista"
            titleTypographyProps={{ variant: "h4", fontWeight: "medium" }}
            subheader="Visualize and analyze classification performance with ROC curves and confusion matrices"
          />
          <CardContent>
            <Typography color="text.secondary">
              Adjust the threshold below to see how it affects true positive rate, false positive rate, 
              and overall model performance metrics in real-time.
            </Typography>
          </CardContent>
        </Card>
        
        <Grid container spacing={3} sx={{ mb: 3 }}>
          <Grid xs={12} md={4}>
            <ThresholdControl 
              threshold={threshold} 
              onChange={setThreshold}
            />
          </Grid>
          
          <Grid xs={12} md={8}>
            <Card>
              <CardHeader 
                title="Threshold Impact"
                titleTypographyProps={{ variant: "h6" }}
                sx={{ pb: 1 }}
              />
              <CardContent>
                <Grid container spacing={3}>
                  <Grid xs={12} sm={6}>
                    <Box>
                      <Typography variant="body2" fontWeight="medium">Threshold:</Typography>
                      <Typography variant="h4" fontWeight="medium">{threshold.toFixed(2)}</Typography>
                    </Box>
                  </Grid>
                  <Grid xs={12} sm={6}>
                    <Box>
                      <Typography variant="body2" fontWeight="medium">Current Point:</Typography>
                      <Box display="flex" gap={4} mt={1}>
                        <Box>
                          <Typography variant="caption" color="text.secondary">TPR</Typography>
                          <Typography variant="h6">
                            {data?.roc_curve.find(p => Math.abs(p.threshold - threshold) < 0.01)?.tpr.toFixed(3) || "—"}
                          </Typography>
                        </Box>
                        <Box>
                          <Typography variant="caption" color="text.secondary">FPR</Typography>
                          <Typography variant="h6">
                            {data?.roc_curve.find(p => Math.abs(p.threshold - threshold) < 0.01)?.fpr.toFixed(3) || "—"}
                          </Typography>
                        </Box>
                      </Box>
                    </Box>
                  </Grid>
                </Grid>
                
                <Divider sx={{ my: 2 }} />
                
                <Box>
                  <Typography variant="body2" fontWeight="medium">Understanding the threshold:</Typography>
                  <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                    A higher threshold value means the model requires more confidence to classify a sample as positive,
                    resulting in fewer false positives but more false negatives. A lower threshold will classify more
                    samples as positive, increasing true positives but also false positives.
                  </Typography>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
        
        <Grid container spacing={3}>
          <Grid xs={12} md={6}>
            <ROCCurve 
              rocData={data?.roc_curve || []} 
              currentThreshold={threshold}
              isLoading={isLoading}
            />
          </Grid>
          
          <Grid xs={12} md={6}>
            <ConfusionMatrix 
              data={data?.confusion_matrix || emptyMatrix}
              isLoading={isLoading}
            />
          </Grid>
        </Grid>
      </Box>
    </MuiLayout>
  );
};

export default Index;
