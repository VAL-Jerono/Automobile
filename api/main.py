"""
============================================================================
INSURANCE ANALYTICS API - FASTAPI APPLICATION
============================================================================

Production-ready API serving ML model predictions for:
- Churn prediction (89.26% ROC-AUC)
- Claims frequency prediction (92.25% ROC-AUC)
- Claims severity estimation (R² = 0.352)
- Customer lifetime value calculation

Author: Valerie Jerono
Date: February 2026
============================================================================
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import logging
from pathlib import Path

from api.routes import predictions, health
from api.models import ModelManager
from api.schemas import HealthResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Insurance Analytics API",
    description="Production ML API for automobile insurance customer analytics",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model manager (lazy loading)
model_manager = None

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global model_manager
    logger.info("🚀 Starting Insurance Analytics API...")
    
    try:
        model_manager = ModelManager()
        logger.info("✅ Model Manager initialized successfully")
        logger.info(f"📊 Models loaded from: {model_manager.model_dir}")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Model Manager: {str(e)}")
        # Don't fail startup - allow lazy loading
        model_manager = None

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("👋 Shutting down Insurance Analytics API...")

# Health check endpoint
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint
    
    Returns:
        HealthResponse: API health status with model availability
    """
    models_status = {}
    
    if model_manager:
        try:
            models_status = {
                "churn_model": model_manager.churn_model is not None,
                "claims_frequency_model": model_manager.claims_frequency_model is not None,
                "claims_severity_model": model_manager.claims_severity_model is not None,
                "clv_model": model_manager.clv_model is not None
            }
        except Exception as e:
            logger.error(f"Error checking model status: {str(e)}")
    
    return HealthResponse(
        status="healthy" if model_manager else "degraded",
        timestamp=datetime.utcnow(),
        version="1.0.0",
        models_loaded=models_status
    )

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Insurance Analytics API",
        "version": "1.0.0",
        "status": "operational",
        "description": "Production ML API for automobile insurance customer analytics",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predictions": "/api/v1/predict/*"
        },
        "models": {
            "churn": {
                "endpoint": "/api/v1/predict/churn",
                "performance": "89.26% ROC-AUC"
            },
            "claims_frequency": {
                "endpoint": "/api/v1/predict/claims-frequency",
                "performance": "92.25% ROC-AUC"
            },
            "claims_severity": {
                "endpoint": "/api/v1/predict/claims-severity",
                "performance": "R² = 0.352"
            },
            "clv": {
                "endpoint": "/api/v1/predict/clv",
                "description": "Customer Lifetime Value"
            }
        }
    }

# Include routers
app.include_router(
    predictions.router,
    prefix="/api/v1/predict",
    tags=["Predictions"]
)

# Get model manager dependency
def get_model_manager() -> ModelManager:
    """Dependency to get model manager instance"""
    if model_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Model Manager not initialized. Please try again later."
        )
    return model_manager

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )
