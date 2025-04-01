import { MetricsResponse } from "@/types";
// ... existing code ...

// Use Axios to connect to the FastAPI backend
import axios from 'axios';

// Configure axios with base URL for the backend
const API_BASE_URL = 'http://localhost:8000'; // Update this to match your backend URL
const apiClient = axios.create({
  baseURL: API_BASE_URL,
});

const API_DELAY = 150; // Maintain delay for consistent UX

export const fetchMetrics = async (analysisId: number, threshold: number = 0.5): Promise<MetricsResponse> => {
  try {
    // Add artificial delay for consistent UX
    await new Promise(resolve => setTimeout(resolve, API_DELAY));
    
    // Fixed: Using backticks for proper string interpolation and including analysisId
    const response = await apiClient.get(`/api/v1/analyses/${analysisId}/metrics`, {
      params: { threshold }
    });
    
    return response.data;
  } catch (error) {
    console.error('Error fetching metrics:', error);
    throw error;
  }
};

export const uploadData = async (analysisId: number): Promise<{ id: number }> => {
  const formData = new FormData();

  // Load the CSV file from public directory
  const response = await fetch("/data/test_data.csv");
  const blob = await response.blob();

  formData.append("file", blob);
  formData.append("name", "Auto Analysis");
  formData.append("description", "Auto-uploaded from analysis page");
  formData.append("default_threshold", "0.5");

  const res = await apiClient.post(`/api/v1/analyses/${analysisId}/upload-data`, formData, {
    headers: { "Content-Type": "multipart/form-data" }
  });

  if (res.status !== 200) throw new Error("Upload failed");
  return res.data;
};

export const getAnalysisStatus = async (analysisId: number): Promise<{
  analysis_id: number;
  has_roc_data: boolean;
  has_confusion_matrix: boolean;
}> => {
  const response = await apiClient.get(`/api/v1/analyses-status/${analysisId}`);
  return response.data;
};