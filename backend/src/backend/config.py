from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Project settings
    PROJECT_NAME: str = "ROC Matrix Vista"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Database settings
    DATABASE_URL: str = "sqlite:///test.db"
    
    # CORS settings
    CORS_ORIGINS: list[str] = ["*"]  # In production, replace with specific origins
    
    # API settings
    API_DELAY: int = 150  # Milliseconds
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Create global settings instance
settings = Settings() 