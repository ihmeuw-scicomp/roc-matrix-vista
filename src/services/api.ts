
import { MetricsResponse } from "@/types";
import { getMockMetrics } from "@/utils/mockData";

// In a real implementation, this would connect to a FastAPI backend
// For now, we'll simulate an API using our mock data utility

const API_DELAY = 150; // Simulate network delay of 150ms

export const fetchMetrics = async (threshold: number = 0.5): Promise<MetricsResponse> => {
  // Simulate API call
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve(getMockMetrics(threshold));
    }, API_DELAY);
  });
};
