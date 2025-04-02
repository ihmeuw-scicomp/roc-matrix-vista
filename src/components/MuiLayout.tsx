import React, { ReactNode } from "react";
import { AppBar, Toolbar, Typography, Container, Box } from "@mui/material";

interface LayoutProps {
  children: ReactNode;
}

const MuiLayout: React.FC<LayoutProps> = ({ children }) => {
  return (
    <Box sx={{ display: "flex", flexDirection: "column", minHeight: "100vh" }}>
      <AppBar position="static" color="default" elevation={1}>
        <Toolbar sx={{ justifyContent: 'space-between' }}>
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            {/* Logo - Using placeholder.svg from public directory */}
            <Box sx={{ mr: 2 }}>
              <img src="/placeholder.svg" alt="Matrix Vista" width={32} height={32} />
            </Box>
            <Typography variant="h6" color="inherit" noWrap>
              Matrix Vista
            </Typography>
          </Box>
          <a
            href="https://en.wikipedia.org/wiki/Receiver_operating_characteristic"
            target="_blank"
            rel="noopener noreferrer"
            style={{ textDecoration: 'none', color: 'inherit' }}
          >
            Learn about ROC curves
          </a>
        </Toolbar>
      </AppBar>
      
      <Container maxWidth="lg" sx={{ mt: 4, mb: 4, flex: 1 }}>
        {children}
      </Container>
      
      <Box component="footer" sx={{ py: 2, bgcolor: "background.paper", borderTop: 1, borderColor: "divider" }}>
        <Container maxWidth="lg">
          <Typography variant="body2" color="text.secondary" align="center">
            © {new Date().getFullYear()} ROC Matrix Vista
          </Typography>
        </Container>
      </Box>
    </Box>
  );
};

export default MuiLayout;
