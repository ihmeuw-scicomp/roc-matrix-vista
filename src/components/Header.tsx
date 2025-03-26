
import React from "react";
import { cn } from "@/lib/utils";

interface HeaderProps {
  className?: string;
}

const Header: React.FC<HeaderProps> = ({ className }) => {
  return (
    <header className={cn("w-full py-6 px-8 flex justify-between items-center", className)}>
      <div className="flex items-center space-x-2">
        <div className="w-8 h-8 rounded-lg bg-primary flex items-center justify-center">
          <svg 
            width="18" 
            height="18" 
            viewBox="0 0 24 24" 
            fill="none" 
            xmlns="http://www.w3.org/2000/svg"
            className="text-primary-foreground"
          >
            <path 
              d="M3 9H21M7 3V5M17 3V5M6 12H10V16H6V12ZM14 12H18V16H14V12Z" 
              stroke="currentColor" 
              strokeWidth="2" 
              strokeLinecap="round" 
              strokeLinejoin="round"
            />
          </svg>
        </div>
        <div>
          <h1 className="text-xl font-medium tracking-tight">Matrix Vista</h1>
          <p className="text-xs text-muted-foreground">Interactive ROC Analysis</p>
        </div>
      </div>
      
      <div className="hidden md:flex items-center">
        <a 
          href="https://en.wikipedia.org/wiki/Receiver_operating_characteristic" 
          target="_blank" 
          rel="noopener noreferrer"
          className="text-sm text-muted-foreground hover:text-foreground transition-colors"
        >
          Learn about ROC curves
        </a>
      </div>
    </header>
  );
};

export default Header;
