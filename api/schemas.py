"""
Pydantic schemas for API request/response models
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, List
from datetime import datetime
from enum import Enum

# Enums
class RiskCategory(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class CustomerSegment(str, Enum):
    PROTECT = "PROTECT"
    RESCUE = "RESCUE"
    GROW = "GROW"
    MONITOR = "MONITOR"

# Health Check
class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str
    models_loaded: Dict[str, bool]

# Churn Prediction
class ChurnPredictionRequest(BaseModel):
    policy_id: Optional[int] = Field(None, description="Policy ID for tracking")
    age: float = Field(..., ge=18, le=100, description="Driver age")
    tenure: float = Field(..., ge=0, description="Customer tenure in years")
    premium: float = Field(..., gt=0, description="Annual premium amount")
    vehicle_age: float = Field(..., ge=0, le=50, description="Vehicle age in years")
    claims_history: int = Field(..., ge=0, description="Number of historical claims")
    channel: str = Field(..., description="Distribution channel (agent/broker)")
    payment_frequency: Optional[str] = Field("annual", description="Payment frequency")
    
    class Config:
        schema_extra = {
            "example": {
                "policy_id": 123456,
                "age": 45,
                "tenure": 2.5,
                "premium": 350.0,
                "vehicle_age": 3,
                "claims_history": 1,
                "channel": "broker",
                "payment_frequency": "annual"
            }
        }

class ChurnPredictionResponse(BaseModel):
    policy_id: Optional[int]
    churn_probability: float = Field(..., ge=0, le=1)
    risk_category: RiskCategory
    segment: CustomerSegment
    recommended_action: str
    confidence: float
    prediction_timestamp: datetime

# Claims Frequency Prediction
class ClaimsFrequencyRequest(BaseModel):
    policy_id: Optional[int]
    age: float = Field(..., ge=18, le=100)
    vehicle_age: float = Field(..., ge=0, le=50)
    vehicle_type: str
    area: str
    power_hp: float = Field(..., gt=0)
    
    class Config:
        schema_extra = {
            "example": {
                "policy_id": 123456,
                "age": 45,
                "vehicle_age": 3,
                "vehicle_type": "Passenger",
                "area": "Urban",
                "power_hp": 120
            }
        }

class ClaimsFrequencyResponse(BaseModel):
    policy_id: Optional[int]
    claims_probability: float = Field(..., ge=0, le=1)
    risk_category: RiskCategory
    expected_claims: float
    prediction_timestamp: datetime

# Claims Severity Prediction
class ClaimsSeverityRequest(BaseModel):
    policy_id: Optional[int]
    vehicle_value: float = Field(..., gt=0)
    vehicle_age: float = Field(..., ge=0)
    power_hp: float = Field(..., gt=0)
    area: str
    
    class Config:
        schema_extra = {
            "example": {
                "policy_id": 123456,
                "vehicle_value": 25000,
                "vehicle_age": 3,
                "power_hp": 120,
                "area": "Urban"
            }
        }

class ClaimsSeverityResponse(BaseModel):
    policy_id: Optional[int]
    expected_severity: float
    severity_range_low: float
    severity_range_high: float
    prediction_timestamp: datetime

# Customer Lifetime Value
class CLVRequest(BaseModel):
    policy_id: Optional[int] = None
    # Customer demographics
    age: float = Field(..., ge=18, le=100)
    gender: str
    married: int = Field(..., ge=0, le=1)
    children: int = Field(..., ge=0)
    
    # Policy details
    tenure: float = Field(..., ge=0)
    premium: float = Field(..., gt=0)
    channel: str
    
    # Financial
    income: float = Field(..., gt=0)
    credit_score: float = Field(..., ge=300, le=850)
    
    # Vehicle information
    vehicle_age: float = Field(..., ge=0)
    vehicle_type: str
    vehicle_ownership: int = Field(..., ge=0, le=1)
    annual_mileage: float = Field(..., gt=0)
    
    # Location
    postal_code: str
    
    # Driving record
    driving_experience: float = Field(..., ge=0)
    claims_history: int = Field(..., ge=0)
    speeding_violations: int = Field(..., ge=0)
    duis: int = Field(..., ge=0)
    past_accidents: int = Field(..., ge=0)
    
    class Config:
        schema_extra = {
            "example": {
                "policy_id": 123456,
                "age": 45,
                "gender": "M",
                "married": 1,
                "children": 2,
                "tenure": 2.5,
                "premium": 650.0,
                "channel": "agent",
                "income": 50000,
                "credit_score": 720,
                "vehicle_age": 5,
                "vehicle_type": "sedan",
                "vehicle_ownership": 1,
                "annual_mileage": 12000,
                "postal_code": "10013",
                "driving_experience": 10,
                "claims_history": 1,
                "speeding_violations": 0,
                "duis": 0,
                "past_accidents": 0
            }
        }

class CLVResponse(BaseModel):
    policy_id: Optional[int]
    predicted_clv: float
    value_tier: str
    expected_lifetime_years: float
    total_expected_revenue: float
    annual_premium: float
    churn_probability: float
    expected_claims_cost: float
    acquisition_cost: float
    npv_cash_flows: float
    annual_projections: List[Dict]
    confidence: float
    methodology: str
    prediction_timestamp: datetime

# Batch Prediction
class BatchPredictionRequest(BaseModel):
    policies: List[Dict]
    prediction_type: str = Field(..., description="churn, claims_frequency, claims_severity, or clv")
    
    @validator('prediction_type')
    def validate_prediction_type(cls, v):
        allowed = ['churn', 'claims_frequency', 'claims_severity', 'clv']
        if v not in allowed:
            raise ValueError(f'prediction_type must be one of {allowed}')
        return v

class BatchPredictionResponse(BaseModel):
    predictions: List[Dict]
    total_processed: int
    timestamp: datetime
