import React, { useState, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import MuiLayout from "@/components/MuiLayout";
import ROCCurve from "@/components/ROCCurve";
import ConfusionMatrix from "@/components/ConfusionMatrix";
import ThresholdSlider from "@/components/ThresholdSlider";
import ThresholdInfo from '@/components/ThresholdInfo';
import { 
  Card, 
  CardContent, 
  CardHeader, 
  Typography, 
  Box, 
  Paper, 
  Divider
} from "@mui/material";
import Grid from "@mui/material/Grid";
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
  
  // Show error message if data fetching fails
  if (isError) {
    return (
      <MuiLayout>
        <Box sx={{ p: 3, textAlign: 'center' }}>
          <Typography color="error">Error fetching data. Please try again later.</Typography>
        </Box>
      </MuiLayout>
    );
  }
  
  return (
    <MuiLayout>
      <Box sx={{ pt: 2 }}>
        {/* Main Title and Description */}
        <Card sx={{ mb: 4 }}>
          <CardHeader
            title="ROC Matrix Vista"
            titleTypographyProps={{ variant: "h4", fontWeight: "medium", textAlign: "center" }}
            subheader="Visualize and analyze classification performance with ROC curves and confusion matrices"
            subheaderTypographyProps={{ textAlign: "center" }}
          />
          <CardContent>
            <Typography color="text.secondary" sx={{ textAlign: "center" }}>
              Adjust the threshold below to see how it affects true positive rate, false positive rate,
              and overall model performance metrics in real-time.
            </Typography>
          </CardContent>
        </Card>

        {/* Threshold Adjustment Section */}
        <Box sx={{ display: 'flex', flexDirection: { xs: 'column', md: 'row' }, gap: 3, my: 4 }}>
          <Box sx={{ flex: 1 }}>
            <Card>
              <CardContent>
                <ThresholdSlider value={threshold} onChange={setThreshold} />
              </CardContent>
            </Card>
          </Box>
          <Box sx={{ flex: 1 }}>
            <ThresholdInfo
              threshold={threshold}
              tpr={data?.current_metrics.tpr}
              fpr={data?.current_metrics.fpr}
            />
          </Box>
        </Box>

        {/* ROC Curve and Confusion Matrix Section */}
        <Grid container spacing={3}>
          <Grid component="div" sx={{ width: { xs: "100%", md: "50%" }, padding: 1.5 }}>
            <ROCCurve
              rocData={data?.roc_curve || []}
              currentThreshold={threshold}
              currentPoint={data?.current_metrics}
              onThresholdSelect={setThreshold}
              isLoading={isLoading}
            />
          </Grid>
          <Grid component="div" sx={{ width: { xs: "100%", md: "50%" }, padding: 1.5 }}>
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
