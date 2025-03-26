
import React, { useEffect, useRef } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import { ROCPoint } from "@/types";

interface ROCCurveProps {
  rocData: ROCPoint[];
  currentThreshold: number;
  className?: string;
  isLoading?: boolean;
}

const ROCCurve: React.FC<ROCCurveProps> = ({
  rocData,
  currentThreshold,
  className,
  isLoading = false,
}) => {
  const chartRef = useRef<HTMLDivElement>(null);
  
  useEffect(() => {
    // Dynamically import Plotly to avoid SSR issues
    const loadPlotly = async () => {
      if (chartRef.current) {
        try {
          const Plotly = await import('plotly.js-dist-min');
          
          // Find the closest point to current threshold
          const closestPoint = rocData.reduce((prev, curr) => {
            return Math.abs(curr.threshold - currentThreshold) < Math.abs(prev.threshold - currentThreshold) 
              ? curr 
              : prev;
          });
          
          // Prepare data for the ROC curve
          const trace1 = {
            x: rocData.map(point => point.fpr),
            y: rocData.map(point => point.tpr),
            mode: 'lines',
            type: 'scatter',
            name: 'ROC Curve',
            line: {
              color: 'rgba(0, 122, 255, 0.8)',
              width: 2.5
            },
            hoverinfo: 'text',
            text: rocData.map(point => 
              `Threshold: ${point.threshold.toFixed(2)}<br>` +
              `True Positive Rate: ${point.tpr.toFixed(3)}<br>` +
              `False Positive Rate: ${point.fpr.toFixed(3)}`
            )
          };
          
          // Diagonal reference line (random classifier)
          const trace2 = {
            x: [0, 1],
            y: [0, 1],
            mode: 'lines',
            type: 'scatter',
            name: 'Random',
            line: {
              color: 'rgba(180, 180, 180, 0.5)',
              width: 1.5,
              dash: 'dash'
            },
            hoverinfo: 'none'
          };
          
          // Current threshold point marker
          const trace3 = {
            x: [closestPoint.fpr],
            y: [closestPoint.tpr],
            mode: 'markers',
            type: 'scatter',
            name: `Threshold: ${currentThreshold.toFixed(2)}`,
            marker: {
              color: 'rgba(255, 59, 48, 1)',
              size: 10
            },
            hoverinfo: 'text',
            text: [
              `Threshold: ${currentThreshold.toFixed(2)}<br>` +
              `True Positive Rate: ${closestPoint.tpr.toFixed(3)}<br>` +
              `False Positive Rate: ${closestPoint.fpr.toFixed(3)}`
            ]
          };
          
          const data = [trace1, trace2, trace3];
          
          const layout = {
            title: '',
            xaxis: {
              title: 'False Positive Rate',
              range: [-0.02, 1.02],
              zeroline: false,
              gridcolor: 'rgba(180, 180, 180, 0.1)'
            },
            yaxis: {
              title: 'True Positive Rate',
              range: [-0.02, 1.02],
              zeroline: false,
              gridcolor: 'rgba(180, 180, 180, 0.1)'
            },
            margin: {
              l: 60,
              r: 30,
              b: 60,
              t: 10,
            },
            showlegend: true,
            legend: {
              x: 0.01,
              y: 0.02,
              bgcolor: 'rgba(255, 255, 255, 0.7)',
              bordercolor: 'rgba(0, 0, 0, 0.1)',
              borderwidth: 1
            },
            plot_bgcolor: 'rgba(0, 0, 0, 0)',
            paper_bgcolor: 'rgba(0, 0, 0, 0)',
            hovermode: 'closest',
            shapes: [
              // AUC fill
              {
                type: 'path',
                path: `M 0,0 ${trace1.x.map((x, i) => `L ${x},${trace1.y[i]}`).join(' ')} L 1,0 Z`,
                fillcolor: 'rgba(0, 122, 255, 0.1)',
                line: { width: 0 }
              }
            ],
            annotations: [
              {
                x: 0.95,
                y: 0.05,
                xref: 'paper',
                yref: 'paper',
                text: `AUC: ${calculateAUC(rocData).toFixed(3)}`,
                showarrow: false,
                font: {
                  family: 'Arial',
                  size: 14,
                  color: 'rgba(0, 0, 0, 0.7)'
                },
                bgcolor: 'rgba(255, 255, 255, 0.7)',
                bordercolor: 'rgba(0, 0, 0, 0.1)',
                borderwidth: 1,
                borderpad: 4,
                borderradius: 4
              }
            ]
          };
          
          const config = {
            responsive: true,
            displayModeBar: false
          };
          
          Plotly.newPlot(chartRef.current, data, layout, config);
          
          // Cleanup function
          return () => {
            if (chartRef.current) {
              Plotly.purge(chartRef.current);
            }
          };
        } catch (error) {
          console.error('Error loading Plotly:', error);
        }
      }
    };
    
    loadPlotly();
  }, [rocData, currentThreshold]);
  
  // Calculate Area Under Curve using trapezoidal rule
  function calculateAUC(rocPoints: ROCPoint[]): number {
    let auc = 0;
    const sortedPoints = [...rocPoints].sort((a, b) => a.fpr - b.fpr);
    
    for (let i = 1; i < sortedPoints.length; i++) {
      const width = sortedPoints[i].fpr - sortedPoints[i-1].fpr;
      const height = (sortedPoints[i].tpr + sortedPoints[i-1].tpr) / 2;
      auc += width * height;
    }
    
    return auc;
  }

  return (
    <Card className={cn("transition-custom animate-fade-up", className)}>
      <CardHeader className="pb-2">
        <CardTitle className="text-lg font-medium">ROC Curve</CardTitle>
      </CardHeader>
      <CardContent>
        <div 
          className={cn("transition-opacity duration-300 h-[400px]", {
            "opacity-50": isLoading
          })}
        >
          <div ref={chartRef} className="w-full h-full" />
        </div>
      </CardContent>
    </Card>
  );
};

export default ROCCurve;
