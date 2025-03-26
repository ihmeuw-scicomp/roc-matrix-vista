
import React from "react";
import Header from "./Header";
import { cn } from "@/lib/utils";

interface LayoutProps {
  children: React.ReactNode;
  className?: string;
}

const Layout: React.FC<LayoutProps> = ({ children, className }) => {
  return (
    <div className="min-h-screen flex flex-col bg-background">
      <Header />
      <main className={cn("flex-1 container mx-auto px-4 pb-12", className)}>
        {children}
      </main>
      <footer className="py-6 px-8 border-t border-border">
        <div className="container mx-auto">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <div className="text-sm text-muted-foreground mb-4 md:mb-0">
              © {new Date().getFullYear()} Matrix Vista. All rights reserved.
            </div>
            <div className="flex space-x-6">
              <span className="text-sm text-muted-foreground">Made with precision</span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Layout;
