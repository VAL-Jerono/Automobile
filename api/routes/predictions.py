"""
Prediction routes for all ML models
"""

from fastapi import APIRouter, HTTPException, Depends
from datetime import datetime
from typing import List
import logging

from api.schemas import (
    ChurnPredictionRequest, ChurnPredictionResponse,
    ClaimsFrequencyRequest, ClaimsFrequencyResponse,
    ClaimsSeverityRequest, ClaimsSeverityResponse,
    CLVRequest, CLVResponse,
    BatchPredictionRequest, BatchPredictionResponse
)
from api.models import ModelManager

logger = logging.getLogger(__name__)
router = APIRouter()

# Dependency to get model manager
model_manager = None

def get_model_manager():
    """Get model manager instance"""
    global model_manager
    if model_manager is None:
        model_manager = ModelManager()
    return model_manager

@router.post("/churn", response_model=ChurnPredictionResponse)
async def predict_churn(
    request: ChurnPredictionRequest,
    manager: ModelManager = Depends(get_model_manager)
):
    """
    Predict customer churn probability
    
    - **policy_id**: Optional policy identifier
    - **age**: Driver age (18-100)
    - **tenure**: Customer tenure in years
    - **premium**: Annual premium amount
    - **vehicle_age**: Vehicle age in years
    - **claims_history**: Number of historical claims
    - **channel**: Distribution channel (agent/broker)
    
    Returns churn probability and risk categorization
    """
    try:
        # Prepare input data
        input_data = request.dict()
        
        # Get prediction from model
        prediction = manager.predict_churn(input_data)
        
        # Determine recommended action
        risk = prediction['risk_category']
        if risk == "CRITICAL":
            action = "Immediate retention call with premium discount offer (15-20%)"
        elif risk == "HIGH":
            action = "Proactive outreach within 7 days with value-added services"
        elif risk == "MEDIUM":
            action = "Include in next retention campaign, monitor closely"
        else:
            action = "Standard engagement, loyalty rewards program"
        
        # Build response
        response = ChurnPredictionResponse(
            policy_id=request.policy_id,
            churn_probability=prediction['churn_probability'],
            risk_category=prediction['risk_category'],
            segment=prediction['segment'],
            recommended_action=action,
            confidence=prediction['confidence'],
            prediction_timestamp=datetime.utcnow()
        )
        
        logger.info(f"Churn prediction for policy {request.policy_id}: {prediction['churn_probability']:.3f}")
        return response
        
    except Exception as e:
        logger.error(f"Error in churn prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@router.post("/claims-frequency", response_model=ClaimsFrequencyResponse)
async def predict_claims_frequency(
    request: ClaimsFrequencyRequest,
    manager: ModelManager = Depends(get_model_manager)
):
    """
    Predict claims frequency probability
    
    Returns probability of filing a claim in the next policy period
    """
    try:
        input_data = request.dict()
        prediction = manager.predict_claims_frequency(input_data)
        
        response = ClaimsFrequencyResponse(
            policy_id=request.policy_id,
            claims_probability=prediction['claims_probability'],
            risk_category=prediction['risk_category'],
            expected_claims=prediction['expected_claims'],
            prediction_timestamp=datetime.utcnow()
        )
        
        logger.info(f"Claims frequency prediction for policy {request.policy_id}: {prediction['claims_probability']:.3f}")
        return response
        
    except Exception as e:
        logger.error(f"Error in claims frequency prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@router.post("/claims-severity", response_model=ClaimsSeverityResponse)
async def predict_claims_severity(
    request: ClaimsSeverityRequest,
    manager: ModelManager = Depends(get_model_manager)
):
    """
    Predict expected claims severity
    
    Returns estimated cost of claims if one occurs
    """
    try:
        input_data = request.dict()
        prediction = manager.predict_claims_severity(input_data)
        
        response = ClaimsSeverityResponse(
            policy_id=request.policy_id,
            expected_severity=prediction['expected_severity'],
            severity_range_low=prediction['severity_range_low'],
            severity_range_high=prediction['severity_range_high'],
            prediction_timestamp=datetime.utcnow()
        )
        
        logger.info(f"Claims severity prediction for policy {request.policy_id}: ${prediction['expected_severity']:.2f}")
        return response
        
    except Exception as e:
        logger.error(f"Error in claims severity prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@router.post("/clv", response_model=CLVResponse)
async def predict_clv(
    request: CLVRequest,
    manager: ModelManager = Depends(get_model_manager)
):
    """
    Predict customer lifetime value using integrated churn-claims DCF model
    
    Calculates 10-year discounted cash flows incorporating:
    - Survival probability from churn model
    - Expected claims costs from frequency and severity models
    - Net present value with 5% discount rate
    
    Returns comprehensive CLV breakdown with annual projections
    """
    try:
        input_data = request.dict()
        prediction = manager.predict_clv(input_data)
        
        response = CLVResponse(
            policy_id=request.policy_id,
            predicted_clv=prediction['predicted_clv'],
            value_tier=prediction['value_tier'],
            expected_lifetime_years=prediction['expected_lifetime_years'],
            total_expected_revenue=prediction['total_expected_revenue'],
            annual_premium=prediction['annual_premium'],
            churn_probability=prediction['churn_probability'],
            expected_claims_cost=prediction['expected_claims_cost'],
            acquisition_cost=prediction['acquisition_cost'],
            npv_cash_flows=prediction['npv_cash_flows'],
            annual_projections=prediction['annual_projections'],
            confidence=prediction['confidence'],
            methodology=prediction['methodology'],
            prediction_timestamp=datetime.utcnow()
        )
        
        logger.info(f"CLV prediction for policy {request.policy_id}: €{prediction['predicted_clv']:.2f} ({prediction['value_tier']})")
        return response
        
    except Exception as e:
        logger.error(f"Error in CLV prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@router.post("/batch", response_model=BatchPredictionResponse)
async def batch_predictions(
    request: BatchPredictionRequest,
    manager: ModelManager = Depends(get_model_manager)
):
    """
    Batch predictions for multiple policies
    
    - **policies**: List of policy data dictionaries
    - **prediction_type**: Type of prediction (churn, claims_frequency, claims_severity, clv)
    
    Returns predictions for all policies
    """
    try:
        predictions = []
        
        for policy_data in request.policies:
            if request.prediction_type == "churn":
                pred = manager.predict_churn(policy_data)
            elif request.prediction_type == "claims_frequency":
                pred = manager.predict_claims_frequency(policy_data)
            elif request.prediction_type == "claims_severity":
                pred = manager.predict_claims_severity(policy_data)
            elif request.prediction_type == "clv":
                pred = manager.predict_clv(policy_data)
            
            predictions.append({
                "policy_id": policy_data.get("policy_id"),
                **pred
            })
        
        response = BatchPredictionResponse(
            predictions=predictions,
            total_processed=len(predictions),
            timestamp=datetime.utcnow()
        )
        
        logger.info(f"Batch prediction completed: {len(predictions)} policies processed")
        return response
        
    except Exception as e:
        logger.error(f"Error in batch prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")
