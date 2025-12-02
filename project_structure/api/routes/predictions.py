"""
Prediction endpoints for lapse, claims, and risk scoring.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import logging
import pandas as pd
import numpy as np
from api.dependencies import get_ensemble_model

logger = logging.getLogger(__name__)
router = APIRouter()

class PolicyData(BaseModel):
    policy_id: int
    age: int = Field(..., ge=18, le=100)
    vehicle_age: int = Field(..., ge=0, le=60)
    premium: float = Field(..., gt=0)
    claims_history: int = Field(default=0, ge=0)
    second_driver: int = Field(default=0, ge=0, le=1)
    type_fuel: str = Field(default="P")
    
    class Config:
        json_schema_extra = {
            "example": {
                "policy_id": 123,
                "age": 45,
                "vehicle_age": 3,
                "premium": 250.0,
                "claims_history": 1,
                "second_driver": 0,
                "type_fuel": "P"
            }
        }

class LapsePredictionResponse(BaseModel):
    policy_id: int
    lapse_probability: float
    lapse_risk: str  # Low, Medium, High
    confidence: float
    recommended_action: str

@router.post("/lapse", response_model=LapsePredictionResponse)
async def predict_lapse(policy: PolicyData):
    """
    Predict lapse probability for a policy.
    
    Returns:
    - lapse_probability: Probability between 0-1
    - lapse_risk: Risk level (Low <0.3, Medium 0.3-0.6, High >0.6)
    - recommended_action: Suggested action (e.g., retention offer, review)
    """
    try:
        model = get_ensemble_model()
        
        # Prepare data
        X = pd.DataFrame([policy.dict()])
        prediction = model.predict(X)[0]
        
        # Risk categorization
        if prediction < 0.3:
            risk_level = "Low"
            action = "Standard monitoring"
        elif prediction < 0.6:
            risk_level = "Medium"
            action = "Consider retention offer"
        else:
            risk_level = "High"
            action = "Immediate retention intervention"
        
        return LapsePredictionResponse(
            policy_id=policy.policy_id,
            lapse_probability=float(prediction),
            lapse_risk=risk_level,
            confidence=0.92,  # Model confidence (placeholder)
            recommended_action=action
        )
    
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class ClaimsData(BaseModel):
    policy_id: int
    premium: float
    vehicle_value: float
    claims_history: int = 0
    age: int
    
    class Config:
        json_schema_extra = {
            "example": {
                "policy_id": 456,
                "premium": 300.0,
                "vehicle_value": 15000.0,
                "claims_history": 2,
                "age": 35
            }
        }

class ClaimsAmountResponse(BaseModel):
    policy_id: int
    expected_claims_amount: float
    expected_claims_frequency: float  # Claims per year
    severity_level: str

@router.post("/claims_amount", response_model=ClaimsAmountResponse)
async def predict_claims_amount(policy: ClaimsData):
    """
    Predict expected claims amount for a policy.
    """
    try:
        model = get_ensemble_model()
        
        X = pd.DataFrame([policy.dict()])
        prediction = model.predict(X)[0]
        
        # Estimate claims amount (simplified)
        claims_amount = prediction * policy.premium * 2
        frequency = max(0.1, prediction)
        
        severity = "High" if prediction > 0.6 else "Medium" if prediction > 0.3 else "Low"
        
        return ClaimsAmountResponse(
            policy_id=policy.policy_id,
            expected_claims_amount=float(claims_amount),
            expected_claims_frequency=float(frequency),
            severity_level=severity
        )
    
    except Exception as e:
        logger.error(f"Claims prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class RiskScoreResponse(BaseModel):
    policy_id: int
    risk_score: float  # 0-100
    risk_category: str
    key_risk_factors: list
    mitigation_strategies: list

@router.post("/risk_score", response_model=RiskScoreResponse)
async def predict_risk_score(policy: PolicyData):
    """
    Generate comprehensive risk score for a policy.
    """
    try:
        model = get_ensemble_model()
        
        X = pd.DataFrame([policy.dict()])
        prediction = model.predict(X)[0]
        risk_score = prediction * 100
        
        # Key risk factors (simplified)
        risk_factors = []
        if policy.age > 60:
            risk_factors.append("High driver age")
        if policy.vehicle_age > 10:
            risk_factors.append("Older vehicle")
        if policy.claims_history > 0:
            risk_factors.append("Prior claims")
        
        # Mitigation strategies
        strategies = [
            "Enroll in telematics program",
            "Offer defensive driving course",
            "Increase deductible for savings"
        ]
        
        return RiskScoreResponse(
            policy_id=policy.policy_id,
            risk_score=float(risk_score),
            risk_category="High" if risk_score > 60 else "Medium" if risk_score > 30 else "Low",
            key_risk_factors=risk_factors,
            mitigation_strategies=strategies
        )
    
    except Exception as e:
        logger.error(f"Risk score failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
