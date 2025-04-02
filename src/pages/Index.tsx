import React, { useState, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import MuiLayout from "@/components/MuiLayout";
import ROCCurve from "@/components/ROCCurve";
import ConfusionMatrix from "@/components/ConfusionMatrix";
import ThresholdSlider from "@/components/ThresholdSlider";
import ThresholdInfo from "@/components/ThresholdInfo";
import DistributionPlot from "@/components/DistributionPlot";
import WorkloadSummary from "@/components/WorkloadSummary";
import { Card, CardHeader, CardContent, Typography, Box, Grid } from "@mui/material";
import { fetchMetrics, uploadData, getAnalysisStatus, fetchExtendedMetrics } from "@/services/api";
import { ConfusionMatrixData } from "@/types";

const Index = () => {
  const [threshold, setThreshold] = useState(0.5);
  const [analysisId, setAnalysisId] = useState<number | null>(null);
  const HARDCODED_ANALYSIS_ID = 1;
  
  // Debounce threshold changes
  const [debouncedThreshold, setDebouncedThreshold] = useState(threshold);
  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedThreshold(threshold);
    }, 100);
    return () => clearTimeout(timer);
  }, [threshold]);

  useEffect(() => {
    const initializeAnalysis = async () => {
      const id = HARDCODED_ANALYSIS_ID;
      setAnalysisId(id); // ✅ Set right away so UI can use it

      try {
        
        const status = await getAnalysisStatus(id);
        console.log(getAnalysisStatus(id))
        if (!status.has_roc_data) {
          console.log("ROC data missing — uploading dataset");
          await uploadData(id); // Should store data under the same analysisId
        } else {
          console.log("Analysis already initialized. Skipping upload.");
        }
      } catch (err) {
        console.error("Failed to get analysis status or upload data:", err);
      }
    };

    initializeAnalysis();
  }, []);
  
  // Fetch data
  const { data, isLoading, isError } = useQuery({
    queryKey: ["metrics", analysisId, debouncedThreshold],
    queryFn: () => analysisId ? fetchMetrics(analysisId, debouncedThreshold) : null,
    staleTime: 60000,
    refetchOnWindowFocus: false,
    enabled: !!analysisId // Only run query when analysisId exists
  });

  // Fetch extended metrics
  const { 
    data: extendedMetrics, 
    isLoading: isExtendedMetricsLoading, 
    isError: isExtendedMetricsError 
  } = useQuery({
    queryKey: ["extendedMetrics", analysisId, debouncedThreshold],
    queryFn: () => analysisId ? fetchExtendedMetrics(analysisId, debouncedThreshold) : null,
    staleTime: 60000,
    refetchOnWindowFocus: false,
    enabled: !!analysisId
  });

  // Default empty matrix
  const emptyMatrix: ConfusionMatrixData = { 
    true_positives: 0, 
    false_positives: 0, 
    true_negatives: 0, 
    false_negatives: 0,
    accuracy: 0,
    precision: 0,
    recall: 0,
    f1_score: 0,
    threshold: 0
  };

  // Error state
  if (isError || isExtendedMetricsError) {
    return (
      <MuiLayout>
        <Box sx={{ p: 3, textAlign: "center" }}>
          <Typography color="error">
            Error fetching data. Please try again later.
          </Typography>
        </Box>
      </MuiLayout>
    );
  }

  return (
    <MuiLayout>
      <Box sx={{ pt: 2 }}>
        {/* Main Title */}
        <Card sx={{ mb: 4 }}>
          <CardHeader
            title={<Typography variant="h4" fontWeight="medium" textAlign="center">ROC Matrix Vista</Typography>}
            subheader={<Typography textAlign="center">Visualize and analyze classification performance with ROC curves and confusion matrices</Typography>}
          />
          <CardContent>
            <Typography color="text.secondary" sx={{ textAlign: "center" }}>
              Adjust the threshold below to see how it affects true positive rate, false positive rate,
              and overall model performance metrics in real-time.
            </Typography>
          </CardContent>
        </Card>

        {/* Threshold Adjustment Section */}
        <Box
          sx={{
            display: "flex",
            flexDirection: { xs: "column", md: "row" },
            gap: 3,
            my: 4,
          }}
        >
          {/* Slider (occupies 2/5 of the width) */}
          <Box sx={{ flex: 2 }}>
            <ThresholdSlider value={threshold} onChange={setThreshold} />
          </Box>

          {/* Threshold Info (occupies 3/5 of the width) */}
          <Box sx={{ flex: 3 }}>
            <ThresholdInfo
              threshold={threshold}
              tpr={data?.current_metrics?.tpr}
              fpr={data?.current_metrics?.fpr}
            />
          </Box>
        </Box>

        {/* ROC Curve and Confusion Matrix Section */}
        <Box sx={{ display: 'flex', flexDirection: { xs: 'column', md: 'row' }, gap: 3, mb: 3 }}>
          <Box sx={{ flex: 1 }}>
            <ROCCurve
              rocData={data?.roc_curve || []}
              currentThreshold={threshold}
              currentPoint={data?.current_metrics}
              onThresholdSelect={setThreshold}
              isLoading={isLoading}
            />
          </Box>
          <Box sx={{ flex: 1 }}>
            <ConfusionMatrix
              data={data?.confusion_matrix || emptyMatrix}
              isLoading={isLoading}
            />
          </Box>
        </Box>

        {/* Distribution Plot - New */}
        <Box sx={{ mb: 3 }}>
          <DistributionPlot
            distributionData={extendedMetrics?.distribution_data || []}
            threshold={threshold}
            loading={isExtendedMetricsLoading}
            error={isExtendedMetricsError}
          />
        </Box>
        
        {/* Workload Summary - New */}
        <Box>
          <WorkloadSummary
            workloadEstimation={extendedMetrics?.workload_estimation}
            threshold={threshold}
            loading={isExtendedMetricsLoading}
            error={isExtendedMetricsError}
          />
        </Box>
      </Box>
    </MuiLayout>
  );
};

export default Index;