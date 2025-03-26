
import { createTheme } from '@mui/material/styles';

const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2', // This is equivalent to primary in your current theme
    },
    secondary: {
      main: '#f5f5f6', // This is similar to your secondary theme
    },
    error: {
      main: '#ff3b30', // Matches your destructive color
    },
    background: {
      default: '#f8f8f8', // Similar to your background color
      paper: '#ffffff', // Similar to your card background
    },
    text: {
      primary: '#292e36', // Similar to your foreground color
      secondary: '#71727a', // Similar to your muted foreground
    },
  },
  typography: {
    fontFamily: [
      'system-ui',
      '-apple-system',
      'BlinkMacSystemFont',
      'Segoe UI',
      'Roboto',
      'Oxygen',
      'Ubuntu',
      'Cantarell',
      'Fira Sans',
      'Droid Sans',
      'Helvetica Neue',
      'sans-serif',
    ].join(','),
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          borderRadius: '0.5rem',
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: '0.5rem',
          boxShadow: '0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24)',
        },
      },
    },
  },
});

export default theme;
