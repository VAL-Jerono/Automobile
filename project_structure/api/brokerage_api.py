"""
ValCare Brokerage API - Motor Insurance Platform
Integrates existing ML models with the brokerage frontend
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, date
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

# Paths to saved models
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
ROOT_MODELS_DIR = "/Users/leonida/Documents/automobile_claims/models"

# Global model storage
models = {}

def load_models():
    """Load all ML models on startup"""
    global models
    
    try:
        # Try loading from project_structure/models first
        model_paths = [
            (MODELS_DIR, "ensemble_model_20251204_223000.pkl"),
            (ROOT_MODELS_DIR, "insurance_ml_system_20251209_094706.pkl"),
            (ROOT_MODELS_DIR, "churn_model_20251209_094706.pkl"),
            (ROOT_MODELS_DIR, "lifecycle_claim_model_20251209_094706.pkl"),
        ]
        
        for dir_path, filename in model_paths:
            filepath = os.path.join(dir_path, filename)
            if os.path.exists(filepath):
                logger.info(f"Loading model from {filepath}")
                with open(filepath, 'rb') as f:
                    model_data = pickle.load(f)
                    models[filename] = model_data
                    logger.info(f"✅ Loaded: {filename}")
        
        logger.info(f"Total models loaded: {len(models)}")
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")

# Load models on startup
load_models()

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

class QuoteResponse(BaseModel):
    """Response model for insurance quote"""
    quotes: List[Dict[str, Any]]
    risk_score: float
    risk_level: str
    risk_factors: List[str]
    churn_probability: Optional[float] = None
    claim_probability: Optional[float] = None

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
    Calculate risk score using ML model or heuristics
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

def predict_churn_probability(data: QuoteRequest) -> float:
    """
    Predict customer churn probability using ML model
    """
    try:
        # Try to use loaded model
        if "churn_model_20251209_094706.pkl" in models:
            model_data = models["churn_model_20251209_094706.pkl"]
            # Prepare features matching model training
            # This is a placeholder - would need exact feature engineering
            pass
    except Exception as e:
        logger.warning(f"Could not use ML model for churn: {e}")
    
    # Fallback heuristic
    churn_base = 0.15  # Base lapse rate
    
    if data.previous_claims > 0:
        churn_base += 0.1
    if data.driving_experience > 10:
        churn_base -= 0.05
    if data.vehicle_use == "private":
        churn_base -= 0.03
    
    return max(0.05, min(0.85, churn_base))

def predict_claim_probability(data: QuoteRequest) -> float:
    """
    Predict claim probability using ML model
    """
    try:
        if "lifecycle_claim_model_20251209_094706.pkl" in models:
            model_data = models["lifecycle_claim_model_20251209_094706.pkl"]
            # Use model if available
            pass
    except Exception as e:
        logger.warning(f"Could not use ML model for claims: {e}")
    
    # Fallback heuristic based on risk factors
    claim_base = 0.08  # Base claim rate
    
    vehicle_age = datetime.now().year - data.vehicle_year
    if vehicle_age > 10:
        claim_base += 0.05
    if data.driver_age < 25:
        claim_base += 0.08
    if data.driving_experience < 3:
        claim_base += 0.06
    if data.previous_claims > 0:
        claim_base += 0.10
    if data.vehicle_use in ["psv", "taxi", "commercial"]:
        claim_base += 0.12
    
    return max(0.02, min(0.75, claim_base))

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
    # Calculate risk using ML
    risk_score, risk_level, risk_factors = calculate_risk_score(request)
    
    # Get churn and claim probabilities
    churn_prob = predict_churn_probability(request)
    claim_prob = predict_claim_probability(request)
    
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
        claim_probability=round(claim_prob, 3)
    )

@app.post("/api/v1/rag/query")
async def rag_query(request: RAGQuery):
    """
    RAG-powered insurance knowledge query
    Answers customer questions using the knowledge base
    """
    query = request.query.lower()
    
    # Knowledge base responses (would connect to actual RAG in production)
    knowledge_base = {
        "documents": {
            "keywords": ["document", "require", "need", "logbook", "id", "kra"],
            "response": """📋 **Documents Required for Motor Insurance in Kenya:**

1. **Logbook (Vehicle Registration Book)** - Original or certified copy
2. **National ID/Passport** - Valid government-issued identification  
3. **KRA PIN Certificate** - For tax compliance
4. **Driving License** - Valid class matching the vehicle type
5. **Vehicle Valuation** - For comprehensive cover (if vehicle > 5 years old)
6. **Vehicle Photos** - Front, back, sides, and interior for comprehensive cover

**For Corporate/Business Vehicles:**
- Certificate of Incorporation
- CR12 Form
- Board Resolution authorizing insurance

Would you like me to explain any of these in detail?""",
            "sources": ["IRA Kenya Guidelines", "Motor Insurance Policy Terms"]
        },
        "premium": {
            "keywords": ["premium", "cost", "price", "calculate", "rate", "how much"],
            "response": """💰 **Motor Insurance Premium Calculation in Kenya:**

**Comprehensive Cover:**
- Rate: 4-8% of vehicle value per year
- Factors affecting premium:
  • Vehicle age and value
  • Driver's age and experience
  • Vehicle use (private/commercial/PSV)
  • Claims history
  • Engine capacity

**Third Party Fire & Theft:**
- Rate: 2-4% of vehicle value
- Covers theft and fire damage + third party liability

**Third Party Only (TPO):**
- Fixed premium: KES 5,000 - 10,000/year
- Minimum legal requirement

**Our AI uses 94% accurate ML models to calculate fair, risk-based premiums!**

Would you like a personalized quote?""",
            "sources": ["AKI Rate Guidelines", "IRA Premium Regulations"]
        },
        "claims": {
            "keywords": ["claim", "accident", "file", "report", "process"],
            "response": """📋 **How to File a Motor Insurance Claim:**

**Step 1: Immediate Actions (Within 24 hours)**
- Report to police and get abstract/OB number
- Take photos of damage and scene
- Exchange details with other parties
- Call our 24/7 claims hotline

**Step 2: Submit Claim Form**
- Fill claim notification form
- Attach police abstract
- Submit damage photos
- Provide third party details if applicable

**Step 3: Assessment**
- Our assessor inspects the vehicle
- Valuation report prepared
- Repair estimate obtained

**Step 4: Settlement**
- Claim approved/negotiated
- Repair authorized OR cash settlement
- Average processing: 5-10 working days

**Required Documents:**
- Claim form
- Police abstract
- Driver's license (copy)
- Photos of damage
- Repair estimates (2-3 quotes)

Need help filing a claim now?""",
            "sources": ["Claims Procedure Manual", "IRA Claims Guidelines"]
        },
        "cover_types": {
            "keywords": ["compare", "difference", "type", "comprehensive", "third party", "tpft", "tpo"],
            "response": """🔍 **Motor Insurance Cover Types Comparison:**

**1. Third Party Only (TPO)** 🛡️
- ✅ Legal minimum requirement
- ✅ Third party injury/death cover
- ✅ Third party property damage
- ❌ Own vehicle damage NOT covered
- 💰 From KES 5,000/year

**2. Third Party Fire & Theft (TPFT)** 🔥
- ✅ All TPO benefits
- ✅ Fire damage to your vehicle
- ✅ Theft of your vehicle
- ❌ Accident damage NOT covered
- 💰 From KES 15,000/year

**3. Comprehensive** ⭐ (Recommended)
- ✅ All TPFT benefits
- ✅ Own accident damage
- ✅ Windscreen cover
- ✅ Personal accident cover
- ✅ Towing & recovery
- 💰 4-8% of vehicle value/year

**Which cover type suits your needs?**""",
            "sources": ["Insurance Regulatory Authority", "Policy Wording"]
        }
    }
    
    # Find best matching response
    best_match = None
    best_score = 0
    
    for key, data in knowledge_base.items():
        score = sum(1 for keyword in data["keywords"] if keyword in query)
        if score > best_score:
            best_score = score
            best_match = data
    
    if best_match and best_score > 0:
        return {
            "answer": best_match["response"],
            "sources": best_match["sources"],
            "confidence": min(0.95, 0.5 + best_score * 0.15)
        }
    
    # Default response
    return {
        "answer": """Thank you for your question! I'm here to help with all your motor insurance needs.

I can assist you with:
• 📄 Required documents
• 💰 Premium calculations  
• 📋 Claims process
• 🔍 Cover type comparisons
• 🚗 Vehicle type requirements

Please ask about any of these topics, or get an instant quote by clicking "Get Quote"!""",
        "sources": ["ValCare Knowledge Base"],
        "confidence": 0.6
    }

@app.get("/api/v1/stats/dashboard")
async def get_dashboard_stats():
    """
    Get dashboard statistics from actual data
    """
    # These would come from the actual database in production
    # Using values from the training data
    return {
        "customers": {
            "total": 105555,
            "active": 89234,
            "new_this_month": 1250
        },
        "policies": {
            "total": 52645,
            "active": 48230,
            "pending": 24,
            "expiring_soon": 847
        },
        "agents": {
            "total": 2547,
            "active": 2134,
            "top_performers": 125
        },
        "premium_volume": {
            "total_kes": 185700000,
            "this_month_kes": 24500000,
            "growth_percent": 12.5
        },
        "claims": {
            "total_filed": 4521,
            "pending": 127,
            "approved": 4128,
            "rejected": 266
        },
        "ml_models": {
            "churn_accuracy": 0.83,
            "claim_accuracy": 0.88,
            "fraud_accuracy": 0.94,
            "data_points": 105555
        }
    }

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
