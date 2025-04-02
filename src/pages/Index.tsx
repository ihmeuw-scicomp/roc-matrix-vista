import React, { useState, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import MuiLayout from "@/components/MuiLayout";
import ROCCurve from "@/components/ROCCurve";
import ConfusionMatrix from "@/components/ConfusionMatrix";
import ThresholdSlider from "@/components/ThresholdSlider";
import ThresholdInfo from "@/components/ThresholdInfo";
import DistributionPlot from "@/components/DistributionPlot";
import WorkloadSummary from "@/components/WorkloadSummary";
import { Card, CardHeader, CardContent, Typography, Box, Grid, Alert } from "@mui/material";
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
  
  // Fetch data for ROC curve and confusion matrix (using labeled data)
  const { data: metricsData, isLoading: isMetricsLoading, isError: isMetricsError } = useQuery({
    queryKey: ["metrics", analysisId, debouncedThreshold],
    queryFn: () => analysisId ? fetchMetrics(analysisId, debouncedThreshold) : null,
    staleTime: 60000,
    refetchOnWindowFocus: false,
    enabled: !!analysisId // Only run query when analysisId exists
  });

  // Fetch extended metrics for distribution and workload (using unlabeled data)
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
  if (isMetricsError || isExtendedMetricsError) {
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

  const distributionData = extendedMetrics?.distribution_data 
  ? extendedMetrics.distribution_data.map(bin => {
      const midpoint = (bin.bin_start + bin.bin_end) / 2;
      return {
        score: midpoint,
        positives: midpoint >= threshold ? bin.count : 0,
        negatives: midpoint < threshold ? bin.count : 0
      };
    })
  : [];

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

        {/* Data Source Information */}
        <Box sx={{ mb: 4 }}>
          <Alert severity="info" sx={{ mb: 2 }}>
            <Typography variant="body2">
              <strong>ROC Curve & Confusion Matrix:</strong> Based on <strong>{extendedMetrics?.validation_metrics?.labeled_count || 0}</strong> labeled data points
            </Typography>
            <Typography variant="body2">
              <strong>Distribution & Workload Estimation:</strong> Applied to <strong>{extendedMetrics?.workload_estimation?.total_articles || 0}</strong> unlabeled data points
            </Typography>
          </Alert>
        </Box>

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
              tpr={metricsData?.current_metrics?.tpr}
              fpr={metricsData?.current_metrics?.fpr}
            />
          </Box>
        </Box>

        {/* ROC Curve and Confusion Matrix Section - Using labeled data */}
        <Box sx={{ display: 'flex', flexDirection: { xs: 'column', md: 'row' }, gap: 3, mb: 3 }}>
          <Box sx={{ flex: 1 }}>
            <ROCCurve
              rocData={metricsData?.roc_curve || []}
              currentThreshold={threshold}
              currentPoint={metricsData?.current_metrics}
              onThresholdSelect={setThreshold}
              isLoading={isMetricsLoading}
            />
          </Box>
          <Box sx={{ flex: 1 }}>
            <ConfusionMatrix
              data={metricsData?.confusion_matrix || emptyMatrix}
              isLoading={isMetricsLoading}
            />
          </Box>
        </Box>

        {/* Distribution Plot - Using unlabeled data */}
        <Box sx={{ mb: 3 }}>
          <DistributionPlot
            distributionData={distributionData}
            threshold={threshold}
            loading={isExtendedMetricsLoading}
            error={isExtendedMetricsError}
          />
        </Box>
        
        {/* Workload Summary - Using unlabeled data with validation metrics for estimation */}
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