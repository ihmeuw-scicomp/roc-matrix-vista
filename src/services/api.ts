import { MetricsResponse, ExtendedMetricsResponse } from "@/types";
// ... existing code ...

// Use Axios to connect to the FastAPI backend
import axios from 'axios';

// Configure axios with base URL for the backend
const API_BASE_URL = 'http://localhost:8000'; // Update this to match your backend URL
const apiClient = axios.create({
  baseURL: API_BASE_URL,
});

const API_DELAY = 150; // Maintain delay for consistent UX

/**
 * Generic function to fetch data from the API with consistent error handling and delay
 * @param endpoint The API endpoint to call
 * @param params Optional query parameters
 * @returns Response data from the API
 */
const fetchData = async <T>(endpoint: string, params?: any): Promise<T> => {
  try {
    // Add artificial delay for consistent UX
    await new Promise(resolve => setTimeout(resolve, API_DELAY));
    
    const response = await apiClient.get(endpoint, { params });
    return response.data;
  } catch (error) {
    console.error(`Error fetching data from ${endpoint}:`, error);
    throw error;
  }
};

/**
 * Fetch metrics for a ROC analysis
 * @param analysisId ID of the ROC analysis
 * @param threshold Classification threshold (0-1)
 * @returns Metrics response including ROC curve and confusion matrix
 */
export const fetchMetrics = async (analysisId: number, threshold: number = 0.5): Promise<MetricsResponse> => {
  return fetchData<MetricsResponse>(`/api/v1/analyses/${analysisId}/metrics`, { threshold });
};

// export const fetchExtendedMetrics = async (analysisId: number, threshold: number) => {
//   try {
//     const response = await fetch(`/api/v1/analyses/${analysisId}/extended-metrics?threshold=${threshold}`);
    
//     if (!response.ok) {
//       throw new Error(`Error fetching extended metrics: ${response.status}`);
//     }
    
//     return await response.json();
//   } catch (error) {
//     console.error("Failed to fetch extended metrics:", error);
//     throw error;
//   }
// };

/**
 * Fetch extended metrics for a ROC analysis
 * @param analysisId ID of the ROC analysis
 * @param threshold Classification threshold (0-1)
 * @returns Extended metrics including distribution data and workload estimation
 */
export const fetchExtendedMetrics = async (analysisId: number, threshold: number): Promise<ExtendedMetricsResponse> => {
  return fetchData<ExtendedMetricsResponse>(`/api/v1/analyses/${analysisId}/extended-metrics`, { threshold });
};

/**
 * Upload data for a ROC analysis
 * @param analysisId ID of the ROC analysis
 * @returns Created or updated ROC analysis info
 */
export const uploadData = async (analysisId: number): Promise<{ id: number }> => {
  try {
    const formData = new FormData();

    // Load the CSV file from public directory
    const response = await fetch("/data/test_data.csv");
    const blob = await response.blob();

    formData.append("file", blob);
    formData.append("name", "Auto Analysis");
    formData.append("description", "Auto-uploaded from analysis page");
    formData.append("default_threshold", "0.5");

    // Add artificial delay for consistent UX
    await new Promise(resolve => setTimeout(resolve, API_DELAY));
    
    const res = await apiClient.post(`/api/v1/analyses/${analysisId}/upload-data`, formData, {
      headers: { "Content-Type": "multipart/form-data" }
    });

    if (res.status !== 200) throw new Error("Upload failed");
    return res.data;
  } catch (error) {
    console.error("Error uploading data:", error);
    throw error;
  }
};

/**
 * Get status of a ROC analysis
 * @param analysisId ID of the ROC analysis
 * @returns Analysis status information
 */
export const getAnalysisStatus = async (analysisId: number): Promise<{
  analysis_id: number;
  exists: boolean;
  has_roc_data: boolean;
  has_confusion_matrix: boolean;
}> => {
  return fetchData(`/api/v1/analyses-status/${analysisId}`);
};