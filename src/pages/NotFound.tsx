
import { useLocation } from "react-router-dom";
import { useEffect } from "react";
import { Button, Typography, Box, Container } from "@mui/material";
import MuiLayout from "@/components/MuiLayout";

const NotFound = () => {
  const location = useLocation();

  useEffect(() => {
    console.error(
      "404 Error: User attempted to access non-existent route:",
      location.pathname
    );
  }, [location.pathname]);

  return (
    <MuiLayout>
      <Container maxWidth="sm">
        <Box 
          sx={{ 
            minHeight: "70vh", 
            display: "flex", 
            flexDirection: "column", 
            alignItems: "center", 
            justifyContent: "center",
            textAlign: "center" 
          }}
        >
          <Typography variant="h1" color="primary.light" sx={{ opacity: 0.2, mb: 2, fontWeight: "bold" }}>
            404
          </Typography>
          <Typography variant="h4" gutterBottom>
            Page not found
          </Typography>
          <Typography color="text.secondary" paragraph sx={{ mb: 4 }}>
            The page you are looking for doesn't exist or has been moved.
          </Typography>
          <Button variant="contained" color="primary" href="/">
            Return to Home
          </Button>
        </Box>
      </Container>
    </MuiLayout>
  );
};

export default NotFound;
