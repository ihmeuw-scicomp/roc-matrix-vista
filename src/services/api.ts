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

export const fetchMetrics = async (threshold: number = 0.5): Promise<MetricsResponse> => {
  try {
    // Add artificial delay for consistent UX
    await new Promise(resolve => setTimeout(resolve, API_DELAY));
    
    // Use the correct endpoint path that matches the backend route
    // The backend route is prefixed with API_V1_STR from settings
    const response = await apiClient.get('/api/v1/analyses/1/metrics', {
      params: { threshold }
    });
    
    return response.data;
  } catch (error) {
    console.error('Error fetching metrics:', error);
    throw error;
  }
};
