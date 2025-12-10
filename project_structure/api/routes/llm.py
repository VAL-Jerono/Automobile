"""
LLM endpoints for customer vehicle checks and underwriter assistance.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from ml.models.llm_fine_tune import OllamaFineTuner

router = APIRouter()

# Initialize LLM (singleton pattern)
_llm_instance = None

def get_llm() -> OllamaFineTuner:
    """Get or create LLM instance."""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = OllamaFineTuner(base_model='phi3:mini')
    return _llm_instance

# Request models
class VehicleCheckRequest(BaseModel):
    """Customer vehicle check request."""
    make: str
    model: str
    year: int
    fuel_type: Optional[str] = None
    power: Optional[int] = None
    usage: Optional[str] = "personal"
    customer_age: Optional[int] = None
    
class UnderwriterQueryRequest(BaseModel):
    """Underwriter AI assistant query."""
    query: str
    context: Optional[Dict[str, Any]] = None
    policy_id: Optional[int] = None
    
class RiskAssessmentRequest(BaseModel):
    """Detailed risk assessment request."""
    policy_data: Dict[str, Any]
    
class PolicyRecommendationRequest(BaseModel):
    """Policy recommendation request."""
    customer_profile: Dict[str, Any]


# Customer endpoints
@router.post("/check-vehicle")
async def check_vehicle(request: VehicleCheckRequest):
    """
    Customer-facing: AI-powered vehicle check before quotation.
    Returns insurability assessment and risk factors.
    """
    try:
        llm = get_llm()
        
        # Build vehicle info
        vehicle_info = {
            "make_model": f"{request.year} {request.make} {request.model}",
            "age": f"{2025 - request.year} years",
            "fuel_type": request.fuel_type or "Unknown",
            "power": f"{request.power} HP" if request.power else "Unknown",
            "usage": request.usage,
            "owner_age": request.customer_age
        }
        
        # Generate assessment
        prompt = f"""As an insurance AI assistant, analyze this vehicle for insurability:

Vehicle: {vehicle_info['make_model']}
Age: {vehicle_info['age']}
Fuel: {vehicle_info['fuel_type']}
Power: {vehicle_info['power']}
Usage: {vehicle_info['usage']}
Owner Age: {vehicle_info['owner_age']}

Provide a brief assessment in 3 parts:
1. Insurability (Good/Moderate/High Risk)
2. Key risk factors (1-2 sentences)
3. Recommendation (proceed to quote or considerations)

Keep response under 100 words:"""

        assessment = llm.generate_text(prompt, max_tokens=150)
        
        # Determine if customer can proceed to quotation
        can_proceed = "high risk" not in assessment.lower()
        
        return {
            "status": "success",
            "vehicle": f"{request.year} {request.make} {request.model}",
            "assessment": assessment,
            "can_proceed_to_quote": can_proceed,
            "message": "Vehicle assessment complete. You may proceed to get a quotation." if can_proceed 
                      else "Please review the risk factors before proceeding."
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vehicle check failed: {str(e)}")


@router.post("/quick-quote-advice")
async def quick_quote_advice(request: VehicleCheckRequest):
    """
    Customer-facing: Quick advice before getting formal quote.
    Simpler, faster response for initial guidance.
    """
    try:
        llm = get_llm()
        
        prompt = f"""A customer wants insurance for a {request.year} {request.make} {request.model}.
In 2 sentences, provide quick advice on what to expect:"""

        advice = llm.generate_text(prompt, max_tokens=80)
        
        return {
            "status": "success",
            "vehicle": f"{request.year} {request.make} {request.model}",
            "advice": advice,
            "next_step": "Get Detailed Quote"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quote advice failed: {str(e)}")


# Underwriter endpoints
@router.post("/underwriter-assist")
async def underwriter_assist(request: UnderwriterQueryRequest):
    """
    Admin-facing: AI assistant for underwriters.
    Answers questions about policies, risk factors, and underwriting decisions.
    """
    try:
        llm = get_llm()
        
        # Build context for the query
        context_str = ""
        if request.context:
            context_str = "\n".join([f"{k}: {v}" for k, v in request.context.items()])
        
        policy_ref = f"Policy #{request.policy_id}" if request.policy_id else "General inquiry"
        
        prompt = f"""As an insurance underwriting AI assistant, answer this query:

Query: {request.query}

Context:
{context_str if context_str else "No additional context"}

Reference: {policy_ref}

Provide a professional, concise answer (2-3 sentences):"""

        answer = llm.generate_text(prompt, max_tokens=150)
        
        return {
            "status": "success",
            "query": request.query,
            "answer": answer,
            "policy_id": request.policy_id,
            "timestamp": str(pd.Timestamp.now())
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Underwriter assist failed: {str(e)}")


@router.post("/assess-risk")
async def assess_risk(request: RiskAssessmentRequest):
    """
    Admin-facing: Detailed risk assessment for underwriting decisions.
    Provides comprehensive analysis of policy risk.
    """
    try:
        llm = get_llm()
        
        # Use existing risk assessment method
        assessment = llm.generate_risk_assessment(request.policy_data)
        
        # Determine risk level from assessment
        risk_level = "moderate"
        if "high risk" in assessment.lower():
            risk_level = "high"
        elif "low risk" in assessment.lower() or "good risk" in assessment.lower():
            risk_level = "low"
        
        return {
            "status": "success",
            "risk_level": risk_level,
            "assessment": assessment,
            "policy_data": request.policy_data,
            "timestamp": str(pd.Timestamp.now())
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Risk assessment failed: {str(e)}")


@router.post("/recommend-policy")
async def recommend_policy(request: PolicyRecommendationRequest):
    """
    Admin-facing: Generate policy recommendations for customers.
    Helps underwriters suggest appropriate coverage.
    """
    try:
        llm = get_llm()
        
        # Use existing recommendation method
        recommendation = llm.generate_policy_recommendation(request.customer_profile)
        
        return {
            "status": "success",
            "customer_profile": request.customer_profile,
            "recommendation": recommendation,
            "timestamp": str(pd.Timestamp.now())
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Policy recommendation failed: {str(e)}")


@router.post("/explain-decision")
async def explain_decision(policy_id: int, decision: str, reason: str):
    """
    Admin-facing: Generate explanation for underwriting decision.
    Creates clear explanations for approve/deny decisions.
    """
    try:
        llm = get_llm()
        
        # Use existing claim explanation method
        explanation = llm.generate_claim_explanation(policy_id, reason)
        
        return {
            "status": "success",
            "policy_id": policy_id,
            "decision": decision,
            "reason": reason,
            "explanation": explanation,
            "timestamp": str(pd.Timestamp.now())
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Decision explanation failed: {str(e)}")


@router.get("/health")
async def llm_health():
    """Check LLM service health."""
    try:
        llm = get_llm()
        is_available = llm.check_model_availability()
        
        return {
            "status": "healthy" if is_available else "unavailable",
            "ollama_host": llm.ollama_host,
            "model": llm.base_model,
            "available": is_available
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


# Import pandas for timestamps
import pandas as pd
