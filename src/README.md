# Frontend Structure

This directory contains the frontend code for the ROC Matrix Vista application.

## Project Structure

```
src/
├── components/       # Reusable UI components
│   ├── ConfusionMatrix.tsx    # Confusion matrix visualization
│   ├── DistributionPlot.tsx   # Distribution plot for prediction scores
│   ├── Header.tsx             # Application header
│   ├── Layout.tsx             # Layout wrapper
│   ├── ROCCurve.tsx           # Interactive ROC curve component
│   ├── ThresholdInfo.tsx      # Threshold information display
│   └── ThresholdSlider.tsx    # Slider for adjusting threshold
│
├── pages/            # Page components that represent routes
│   └── Index.tsx     # Main dashboard page
│
├── services/         # API service functions
│   └── api.ts        # Backend API integration
│
├── types/            # TypeScript type definitions
│   └── index.ts      # Shared type definitions
│
├── lib/              # Utility functions and helpers
│   └── utils.ts      # Shared utility functions
│
├── App.tsx           # Main application component
├── main.tsx          # Application entry point
├── theme.ts          # UI theme configuration
└── index.css         # Global styles
```

## Key Features

- Interactive ROC curve with adjustable threshold
- Confusion matrix visualization
- Score distribution visualization
- Threshold adjustment with real-time updates

## Technology Stack

- React with TypeScript
- Recharts for data visualization
- Material-UI (MUI) for UI components
- React Query for data fetching and caching
- Vite for build tooling 