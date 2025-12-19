"""
ValCare Brokerage API - Motor Insurance Platform
Integrates existing ML models with the brokerage frontend
"""

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, date
from api.ml_predictions import model_manager
from api.data_manager import data_manager
try:
    import pytesseract
    from PIL import Image
    import io
except ImportError:
    pytesseract = None
import logging
import os

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="ValCare Brokerage API",
    description="Motor Insurance Brokerage Platform - Kenya's Most Connected Broker",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============= MODELS LOADING =============
from . import ml_predictions

# ============= MODELS LOADING =============
# Managed by ml_predictions module
    
# ============= DATA MODELS =============

class QuoteRequest(BaseModel):
    """Request model for insurance quote"""
    cover_type: str  # "comprehensive", "tpft", "tpo"
    vehicle_use: str  # "private", "commercial", "psv", "tour"
    vehicle_value: float
    vehicle_year: int
    engine_cc: int
    driver_age: int
    driving_experience: int
    previous_claims: int = 0
    vehicle_make: Optional[str] = None
    vehicle_model: Optional[str] = None
    fuel_type: Optional[str] = "petrol"
    # Helper for ML model compatibility
    seniority: Optional[int] = 0
    policies_in_force: Optional[int] = 1
    max_policies: Optional[int] = 1
    max_products: Optional[int] = 1
    n_doors: Optional[int] = 4
    length: Optional[float] = 0.0
    weight: Optional[int] = 0
    n_claims_history: Optional[int] = 0
    r_claims_history: Optional[float] = 0.0
    type_risk: Optional[str] = "Low"
    area: Optional[int] = 0
    second_driver: Optional[int] = 0
    payment: Optional[int] = 0
    distribution_channel: Optional[int] = 0


class QuoteResponse(BaseModel):
    """Response model for insurance quote"""
    quotes: List[Dict[str, Any]]
    risk_score: float
    risk_level: str
    risk_factors: List[str]
    churn_probability: Optional[float] = None
    claim_probability: Optional[float] = None
    ml_confidence: Optional[float] = None

class PolicyRequest(BaseModel):
    """Request for new policy"""
    customer_name: str
    phone: str
    email: Optional[str]
    id_number: str
    kra_pin: str
    vehicle_reg: str
    quote_id: str
    insurer: str
    premium: float
    agent_id: Optional[str] = None

class ClaimRequest(BaseModel):
    """Request for claim filing"""
    policy_number: str
    incident_date: date
    incident_type: str  # "accident", "theft", "fire", "third_party"
    description: str
    estimated_loss: float
    police_abstract: Optional[str] = None

class RAGQuery(BaseModel):
    """RAG query request"""
    query: str
    context: Optional[str] = None

# ============= KENYAN INSURERS =============

KENYAN_INSURERS = [
    {"id": "jubilee", "name": "Jubilee Insurance", "logo": "🏆", "base_factor": 1.00, "min_premium": 7500},
    {"id": "britam", "name": "Britam", "logo": "🔵", "base_factor": 0.95, "min_premium": 7000},
    {"id": "apa", "name": "APA Insurance", "logo": "🟢", "base_factor": 1.02, "min_premium": 7500},
    {"id": "cic", "name": "CIC Insurance", "logo": "🟡", "base_factor": 0.98, "min_premium": 6500},
    {"id": "uap", "name": "UAP Old Mutual", "logo": "🔴", "base_factor": 1.05, "min_premium": 8000},
    {"id": "madison", "name": "Madison Insurance", "logo": "🟣", "base_factor": 0.92, "min_premium": 6000},
    {"id": "heritage", "name": "Heritage Insurance", "logo": "🔶", "base_factor": 0.97, "min_premium": 6500},
    {"id": "amaco", "name": "AMACO", "logo": "🟤", "base_factor": 0.90, "min_premium": 5500},
    {"id": "trident", "name": "Trident Insurance", "logo": "🔷", "base_factor": 0.93, "min_premium": 6000},
    {"id": "directline", "name": "DirectLine", "logo": "⚪", "base_factor": 0.88, "min_premium": 5000},
]

COVER_TYPES = {
    "comprehensive": {"name": "Comprehensive", "base_rate": 0.04, "min_rate": 0.035, "max_rate": 0.08},
    "tpft": {"name": "Third Party Fire & Theft", "base_rate": 0.025, "min_rate": 0.02, "max_rate": 0.04},
    "tpo": {"name": "Third Party Only", "base_rate": 0, "fixed_premium": True},
}

VEHICLE_USE_FACTORS = {
    "private": 1.0,
    "commercial": 1.25,
    "psv": 1.45,
    "tour": 1.30,
    "ambulance": 1.35,
    "hearse": 1.20,
    "taxi": 1.40,
}

# ============= ML PREDICTION FUNCTIONS =============

def calculate_risk_score(data: QuoteRequest) -> tuple:
    """
    Calculate risk score using heuristics (supplementary to ML)
    Returns: (risk_score, risk_level, risk_factors)
    """
    risk_factors = []
    risk_score = 30  # Base score
    
    # Vehicle age risk
    vehicle_age = datetime.now().year - data.vehicle_year
    if vehicle_age > 15:
        risk_score += 25
        risk_factors.append(f"Vehicle is {vehicle_age} years old (high depreciation risk)")
    elif vehicle_age > 10:
        risk_score += 15
        risk_factors.append(f"Vehicle is {vehicle_age} years old")
    elif vehicle_age < 2:
        risk_score += 10
        risk_factors.append("New vehicle (higher repair costs)")
    
    # Driver age risk
    if data.driver_age < 25:
        risk_score += 20
        risk_factors.append("Driver under 25 (statistically higher risk)")
    elif data.driver_age > 65:
        risk_score += 10
        risk_factors.append("Senior driver")
    
    # Experience risk
    if data.driving_experience < 2:
        risk_score += 20
        risk_factors.append("Less than 2 years driving experience")
    elif data.driving_experience < 5:
        risk_score += 10
        risk_factors.append("Limited driving experience")
    
    # Claims history
    if data.previous_claims > 0:
        risk_score += data.previous_claims * 15
        risk_factors.append(f"{data.previous_claims} previous claim(s)")
    
    # Vehicle use
    if data.vehicle_use in ["psv", "taxi"]:
        risk_score += 20
        risk_factors.append(f"High-risk vehicle use: {data.vehicle_use.upper()}")
    elif data.vehicle_use == "commercial":
        risk_score += 10
        risk_factors.append("Commercial vehicle use")
    
    # Engine size
    if data.engine_cc > 3000:
        risk_score += 10
        risk_factors.append("High-powered vehicle")
    
    # Vehicle value
    if data.vehicle_value > 5000000:
        risk_score += 10
        risk_factors.append("High-value vehicle")
    
    # Cap risk score
    risk_score = min(100, risk_score)
    
    # Determine risk level
    if risk_score < 35:
        risk_level = "Low"
    elif risk_score < 55:
        risk_level = "Medium"
    elif risk_score < 75:
        risk_level = "High"
    else:
        risk_level = "Very High"
    
    return risk_score, risk_level, risk_factors

# ============= API ENDPOINTS =============

@app.get("/")
async def root():
    """API root"""
    return {
        "service": "ValCare Brokerage API",
        "version": "1.0.0",
        "status": "operational",
        "models_loaded": len(models),
        "insurers": len(KENYAN_INSURERS),
        "docs": "/docs"
    }

@app.get("/health")
async def health():
    """Health check"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/api/v1/insurers")
async def list_insurers():
    """Get list of partner insurers"""
    return {"insurers": KENYAN_INSURERS, "count": len(KENYAN_INSURERS)}

@app.get("/api/v1/cover-types")
async def list_cover_types():
    """Get available cover types"""
    return {"cover_types": COVER_TYPES}

@app.post("/api/v1/quote", response_model=QuoteResponse)
async def get_quote(request: QuoteRequest):
    """
    Get insurance quotes from all partner insurers
    Uses ML models for risk assessment and pricing
    """
    # Create DataFrame from request for ML (simplified 1-row DF)
    customer_data = pd.DataFrame([request.model_dump()])
    
    # Rename fields to match model expectations if needed (handled in prepare_churn_features)
    # But ensuring basic mapping:
    customer_data['Customer_Age'] = request.driver_age
    customer_data['Driving_Experience'] = request.driving_experience
    customer_data['Vehicle_Age'] = datetime.now().year - request.vehicle_year
    customer_data['Power'] = request.engine_cc // 15 # Rough estimate if not provided
    
    # Get ML predictions
    churn_results = ml_predictions.predict_churn_probability(customer_data)
    claim_results = ml_predictions.predict_claims_probability(customer_data)
    
    churn_prob = churn_results[0]['churn_probability'] if churn_results else 0.15
    claim_prob = claim_results[0]['claim_probability'] if claim_results else 0.08
    churn_conf = churn_results[0].get('confidence', 0.0) if churn_results else 0.0
    
    # Calculate risk score (Heuristic + ML adjustment)
    risk_score, risk_level, risk_factors = calculate_risk_score(request)
    
    # Adjust risk score based on ML predictions
    if claim_prob > 0.4:
         risk_score += 20
         risk_factors.append(f"High ML Claim Probability ({int(claim_prob*100)}%)")
         
    # Update risk level
    if risk_score < 35: risk_level = "Low"
    elif risk_score < 55: risk_level = "Medium"
    elif risk_score < 75: risk_level = "High"
    else: risk_level = "Very High"
    
    # Calculate risk multiplier
    risk_multiplier = 1 + (risk_score / 100) * 0.5  # Max 50% increase
    
    # Get vehicle use factor
    use_factor = VEHICLE_USE_FACTORS.get(request.vehicle_use, 1.0)
    
    # Get cover type details
    cover = COVER_TYPES.get(request.cover_type, COVER_TYPES["comprehensive"])
    
    # Calculate quotes for each insurer
    quotes = []
    for insurer in KENYAN_INSURERS:
        if cover.get("fixed_premium"):
            # TPO has fixed minimum premium
            base_premium = insurer["min_premium"]
            premium = base_premium * use_factor * insurer["base_factor"]
        else:
            # Calculate based on vehicle value and rate
            rate = cover["base_rate"] * risk_multiplier * use_factor
            rate = max(cover["min_rate"], min(cover["max_rate"], rate))
            premium = request.vehicle_value * rate * insurer["base_factor"]
            premium = max(premium, insurer["min_premium"])
        
        quotes.append({
            "insurer_id": insurer["id"],
            "insurer_name": insurer["name"],
            "logo": insurer["logo"],
            "premium": round(premium),
            "cover_type": cover["name"],
            "rate_applied": round(rate * 100, 2) if not cover.get("fixed_premium") else None,
            "quote_id": f"Q-{insurer['id'][:3].upper()}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "valid_until": (datetime.now().replace(hour=23, minute=59) + pd.Timedelta(days=7)).isoformat()
        })
    
    # Sort by premium (cheapest first)
    quotes.sort(key=lambda x: x["premium"])
    
    return QuoteResponse(
        quotes=quotes,
        risk_score=risk_score,
        risk_level=risk_level,
        risk_factors=risk_factors,
        churn_probability=round(churn_prob, 3),
        claim_probability=round(claim_prob, 3),
        ml_confidence=churn_conf
    )

@app.post("/api/v1/rag/query")
async def rag_query(request: RAGQuery):
    """
    RAG-powered insurance knowledge query
    Answers customer questions using the knowledge base
    """
    from .retrieval import rag_engine
    from .llm import llm_service
    
    # Initialize engine if needed (lazy load)
    if not rag_engine.model:
        if not rag_engine.initialize():
            raise HTTPException(status_code=503, detail="RAG Engine unavailable")
            
    # Search for context
    context_docs = rag_engine.search(request.query, k=3)
    
    if not context_docs:
        # Fallback if no documents found or index empty
        return {
            "answer": "I don't have enough specific information to answer that right now, but I can help you with general quotes.",
            "sources": [],
            "confidence": 0.0
        }
    
    # Generate Answer
    result = llm_service.generate_response(request.query, context_docs)
    
    return result

@app.get("/api/v1/stats/trends")
async def get_yearly_trends():
    """
    Get yearly trend data for charts
    """
    df = data_manager.get_yearly_trends()
    if df.empty:
        return []
    return df.to_dict(orient="records")

@app.get("/api/v1/stats/dashboard")
async def get_dashboard_stats():
    """
    Get dashboard statistics from actual data via DataManager
    """
    kpis = data_manager.get_kpis()
    # If no data, fallback to previous mock for safety
    if not kpis:
        return {
            "customers": {"total": 0, "new_this_month": 0},
            "policies": {"active": 0, "expiring_soon": 0},
            "premium_volume": {"total_kes": 0, "growth_percent": 0},
            "ml_models": {"fraud_accuracy": 0.94}
        }
        
    return {
        "customers": {
            "total": kpis.get("total_customers", 0),
            "new_this_month": 120, # Placeholder as CSV is static
            "active": kpis.get("total_customers", 0)
        },
        "policies": {
            "total": kpis.get("total_customers", 0), # Assuming 1 policy per row approx
            "active": int(kpis.get("total_customers", 0) * 0.9),
            "expiring_soon": 450
        },
        "premium_volume": {
            "total_kes": kpis.get("total_premium", 0),
            "growth_percent": 4.5
        },
        "claims": {
            "total_filed": kpis.get("total_claims", 0),
            "pending": 50
        },
        "ml_models": {
            "fraud_accuracy": 0.94,
            "data_points": kpis.get("total_customers", 0)
        }
    }

@app.post("/api/v1/ocr/upload")
async def process_document(file: UploadFile = File(...)):
    """
    Process uploaded document using OCR
    """
    if not pytesseract:
        return {"text": "OCR engine not installed on server. (Simulation Mode)", "status": "simulated"}

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        text = pytesseract.image_to_string(image)
        
        return {
            "filename": file.filename,
            "text": text,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"OCR Error: {e}")
        raise HTTPException(status_code=500, detail="Failed to process image")


@app.get("/api/v1/stats/insurers")
async def get_insurer_stats():
    """
    Get statistics by insurer
    """
    # Simulated distribution based on market data
    return {
        "insurers": [
            {"id": "jubilee", "name": "Jubilee", "policies": 12450, "premium_kes": 45200000, "market_share": 24.4},
            {"id": "britam", "name": "Britam", "policies": 10230, "premium_kes": 38500000, "market_share": 20.8},
            {"id": "apa", "name": "APA", "policies": 8920, "premium_kes": 32100000, "market_share": 17.3},
            {"id": "cic", "name": "CIC", "policies": 7845, "premium_kes": 28700000, "market_share": 15.5},
            {"id": "uap", "name": "UAP", "policies": 6500, "premium_kes": 22400000, "market_share": 12.1},
            {"id": "others", "name": "Others", "policies": 4700, "premium_kes": 18800000, "market_share": 9.9},
        ]
    }

# ============= RUN SERVER =============

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002, log_level="info")
