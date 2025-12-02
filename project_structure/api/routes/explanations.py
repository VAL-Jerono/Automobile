"""
Model explanation endpoints using SHAP and LLM.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import logging
from api.dependencies import get_ensemble_model, get_llm_engine

logger = logging.getLogger(__name__)
router = APIRouter()

class ExplanationRequest(BaseModel):
    prediction_id: str
    model_type: str = "ensemble"  # ensemble, xgb, lgb, nn
    include_llm_narrative: bool = True

class FeatureImportance(BaseModel):
    feature: str
    importance: float
    direction: str  # positive, negative

class ExplanationResponse(BaseModel):
    prediction_id: str
    top_features: List[FeatureImportance]
    shap_summary: str
    llm_narrative: str = None

@router.post("/prediction", response_model=ExplanationResponse)
async def explain_prediction(req: ExplanationRequest):
    """
    Generate SHAP-based explanation for a model prediction.
    Optionally includes LLM-generated narrative explanation.
    """
    try:
        model = get_ensemble_model()
        
        # Get SHAP explanation (placeholder)
        top_features = [
            FeatureImportance(feature="vehicle_age", importance=0.35, direction="positive"),
            FeatureImportance(feature="claims_history", importance=0.28, direction="positive"),
            FeatureImportance(feature="premium", importance=0.22, direction="negative"),
            FeatureImportance(feature="age", importance=0.15, direction="positive"),
        ]
        
        shap_summary = f"""
        The model primarily considers {top_features[0].feature} (35% importance) as the strongest 
        predictor, followed by {top_features[1].feature} (28% importance). 
        Premium and driver age also play significant roles.
        """
        
        llm_narrative = None
        if req.include_llm_narrative:
            llm = get_llm_engine()
            llm_narrative = llm.generate_text(
                f"Explain why a prediction for policy {req.prediction_id} shows high lapse risk "
                f"based on vehicle age and claims history.",
                max_tokens=200
            )
        
        return ExplanationResponse(
            prediction_id=req.prediction_id,
            top_features=top_features,
            shap_summary=shap_summary,
            llm_narrative=llm_narrative
        )
    
    except Exception as e:
        logger.error(f"Explanation generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class PredictionNarrativeRequest(BaseModel):
    policy_id: int
    prediction_type: str  # lapse, claims, risk
    predicted_value: float

class NarrativeResponse(BaseModel):
    narrative: str
    key_insights: List[str]
    next_steps: List[str]

@router.post("/narrative", response_model=NarrativeResponse)
async def generate_narrative_explanation(req: PredictionNarrativeRequest):
    """
    Generate natural language explanation using fine-tuned LLM.
    """
    try:
        llm = get_llm_engine()
        
        prompt = f"""
        Generate a concise explanation for a motor vehicle insurance {req.prediction_type} prediction.
        Policy ID: {req.policy_id}
        Predicted Value: {req.predicted_value}
        
        Explanation should be clear, actionable, and suitable for customer communication.
        """
        
        narrative = llm.generate_text(prompt, max_tokens=300)
        
        key_insights = [
            "Policy shows elevated lapse risk",
            "Claims history is a significant factor",
            "Consider retention intervention"
        ]
        
        next_steps = [
            "Review policy usage patterns",
            "Contact customer for feedback",
            "Offer competitive rate adjustment"
        ]
        
        return NarrativeResponse(
            narrative=narrative or "Unable to generate narrative at this time",
            key_insights=key_insights,
            next_steps=next_steps
        )
    
    except Exception as e:
        logger.error(f"Narrative generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
