"""
FastAPI main application entry point.
Serves insurance risk predictions, explanations, RAG queries, and model management.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import pandas as pd
from api.routes import predictions, explanations, rag, model_mgmt
from api.dependencies import init_models, init_rag
import uvicorn

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lifecycle management
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Insurance Risk Platform API...")
    await init_models()
    await init_rag()
    yield
    logger.info("Shutting down API...")

# Create FastAPI app
app = FastAPI(
    title="Insurance Risk Platform API",
    description="Production-grade ML/AI system for motor vehicle insurance risk assessment",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(predictions.router, prefix="/api/v1/predict", tags=["predictions"])
app.include_router(explanations.router, prefix="/api/v1/explain", tags=["explanations"])
app.include_router(rag.router, prefix="/api/v1/rag", tags=["rag"])
app.include_router(model_mgmt.router, prefix="/api/v1/models", tags=["model_management"])

# Root endpoint
@app.get("/", tags=["root"])
async def root():
    """Root endpoint - API status."""
    return {
        "status": "operational",
        "service": "Insurance Risk Platform API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health", tags=["health"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": str(pd.Timestamp.now())
    }

if __name__ == "__main__":
    import os
    host = os.getenv('API_HOST', '0.0.0.0')
    port = int(os.getenv('API_PORT', 8000))
    debug = os.getenv('API_DEBUG', 'false').lower() == 'true'
    
    uvicorn.run(app, host=host, port=port, log_level='info')
