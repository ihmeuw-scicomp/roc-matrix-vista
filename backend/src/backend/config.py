from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    # Project settings
    PROJECT_NAME: str = "ROC-Matrix-Vista"
    VERSION: str = "0.1.0"
    API_V1_STR: str = "/api/v1"
    
    # Database settings
    DATABASE_URL: str = "sqlite:///./roc_data.db"
    
    # CORS settings
    CORS_ORIGINS: List[str] = ["http://localhost", "http://localhost:3000", "http://localhost:5173"]
    
    # API settings
    API_DELAY: int = 150  # Milliseconds
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Create global settings instance
settings = Settings()