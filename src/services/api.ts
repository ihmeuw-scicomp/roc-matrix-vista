import { MetricsResponse } from "@/types";
// ... existing code ...

// Use Axios to connect to the FastAPI backend
import axios from 'axios';

const API_DELAY = 150; // Maintain delay for consistent UX

export const fetchMetrics = async (threshold: number = 0.5): Promise<MetricsResponse> => {
  try {
    // Add artificial delay for consistent UX
    await new Promise(resolve => setTimeout(resolve, API_DELAY));
    
    // Use relative path to go through the Vite proxy
    const response = await axios.get('/api/metrics', {
      params: { threshold }
    });
    
    return response.data;
  } catch (error) {
    console.error('Error fetching metrics:', error);
    throw error;
  }
};
