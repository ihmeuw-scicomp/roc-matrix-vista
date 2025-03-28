from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.core.config import settings
from backend.routers import roc_analysis

app = FastAPI(title=settings.PROJECT_NAME)

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(roc_analysis.router, prefix=settings.API_V1_STR)

@app.get("/")
def root():
    return {"message": "Welcome to ROC Matrix Vista API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)