
import React from "react";
import { Card, CardContent, CardHeader, Typography, Box, Grid, Paper } from "@mui/material";
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
    <Card className={className} sx={{ overflow: "hidden", transition: "all 0.3s" }}>
      <CardHeader 
        title={<Typography variant="h6">Confusion Matrix</Typography>}
        sx={{ pb: 1 }}
      />
      <CardContent>
        <Box sx={{ opacity: isLoading ? 0.5 : 1, transition: "opacity 0.3s" }}>
          <Paper variant="outlined" sx={{ overflow: "hidden", mb: 2 }}>
            {/* Header Row */}
            <Grid container>
              <Grid item xs={12} container>
                <Grid item xs={6}></Grid>
                <Grid item xs={6} container sx={{ bgcolor: "action.hover", py: 1 }}>
                  <Grid item xs={6}>
                    <Typography variant="caption" align="center">Predicted Positive</Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="caption" align="center">Predicted Negative</Typography>
                  </Grid>
                </Grid>
              </Grid>
              
              {/* True Positive and False Negative */}
              <Grid item xs={12} container sx={{ borderTop: 1, borderColor: "divider" }}>
                <Grid item xs={6} sx={{ 
                  bgcolor: "action.hover", 
                  display: "flex", 
                  alignItems: "center", 
                  justifyContent: "center",
                  position: "relative",
                  height: 100
                }}>
                  <Typography 
                    variant="caption" 
                    sx={{ 
                      transform: "rotate(-90deg)",
                      whiteSpace: "nowrap",
                      position: "absolute"
                    }}
                  >
                    Actual Positive
                  </Typography>
                </Grid>
                <Grid item xs={6} container>
                  <Grid item xs={6} sx={{ 
                    p: 2, 
                    borderLeft: 1, 
                    borderRight: 1, 
                    borderColor: "divider",
                    display: "flex",
                    flexDirection: "column",
                    justifyContent: "space-between",
                    alignItems: "center"
                  }}>
                    <Typography variant="body2">True Positive</Typography>
                    <Typography variant="h4">{TP}</Typography>
                    <Typography variant="caption" color="text.secondary">{tpPercent}%</Typography>
                  </Grid>
                  <Grid item xs={6} sx={{ 
                    p: 2,
                    display: "flex",
                    flexDirection: "column",
                    justifyContent: "space-between",
                    alignItems: "center"
                  }}>
                    <Typography variant="body2">False Negative</Typography>
                    <Typography variant="h4">{FN}</Typography>
                    <Typography variant="caption" color="text.secondary">{fnPercent}%</Typography>
                  </Grid>
                </Grid>
              </Grid>
              
              {/* False Positive and True Negative */}
              <Grid item xs={12} container sx={{ borderTop: 1, borderColor: "divider" }}>
                <Grid item xs={6} sx={{ 
                  bgcolor: "action.hover", 
                  display: "flex", 
                  alignItems: "center", 
                  justifyContent: "center",
                  position: "relative",
                  height: 100
                }}>
                  <Typography 
                    variant="caption" 
                    sx={{ 
                      transform: "rotate(-90deg)",
                      whiteSpace: "nowrap",
                      position: "absolute"
                    }}
                  >
                    Actual Negative
                  </Typography>
                </Grid>
                <Grid item xs={6} container>
                  <Grid item xs={6} sx={{ 
                    p: 2, 
                    borderLeft: 1, 
                    borderRight: 1, 
                    borderColor: "divider",
                    display: "flex",
                    flexDirection: "column",
                    justifyContent: "space-between",
                    alignItems: "center"
                  }}>
                    <Typography variant="body2">False Positive</Typography>
                    <Typography variant="h4">{FP}</Typography>
                    <Typography variant="caption" color="text.secondary">{fpPercent}%</Typography>
                  </Grid>
                  <Grid item xs={6} sx={{ 
                    p: 2,
                    display: "flex",
                    flexDirection: "column",
                    justifyContent: "space-between",
                    alignItems: "center"
                  }}>
                    <Typography variant="body2">True Negative</Typography>
                    <Typography variant="h4">{TN}</Typography>
                    <Typography variant="caption" color="text.secondary">{tnPercent}%</Typography>
                  </Grid>
                </Grid>
              </Grid>
            </Grid>
          </Paper>
          
          <Grid container spacing={2} sx={{ mt: 2 }}>
            <Grid item xs={3}>
              <Paper variant="outlined" sx={{ p: 1.5 }}>
                <Typography variant="caption" color="text.secondary">Accuracy</Typography>
                <Typography variant="h6">{accuracy}%</Typography>
              </Paper>
            </Grid>
            <Grid item xs={3}>
              <Paper variant="outlined" sx={{ p: 1.5 }}>
                <Typography variant="caption" color="text.secondary">Precision</Typography>
                <Typography variant="h6">{precision}%</Typography>
              </Paper>
            </Grid>
            <Grid item xs={3}>
              <Paper variant="outlined" sx={{ p: 1.5 }}>
                <Typography variant="caption" color="text.secondary">Recall</Typography>
                <Typography variant="h6">{recall}%</Typography>
              </Paper>
            </Grid>
            <Grid item xs={3}>
              <Paper variant="outlined" sx={{ p: 1.5 }}>
                <Typography variant="caption" color="text.secondary">F1 Score</Typography>
                <Typography variant="h6">{f1Score}%</Typography>
              </Paper>
            </Grid>
          </Grid>
        </Box>
      </CardContent>
    </Card>
  );
};

export default ConfusionMatrix;
