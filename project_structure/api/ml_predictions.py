"""
ML Predictions Module for Karbima Agency Agent Portal.
Loads trained models and provides prediction functions.
"""


import joblib
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import os
from .config import settings

logger = logging.getLogger(__name__)

class ModelManager:
    _instance = None
    _models = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance
    
    def load_models(self) -> bool:
        """Load all ML models into memory."""
        try:
            logger.info(f"Loading ML models from {settings.MODELS_DIR}...")
            
            # Helper to safely load model
            def load_safe(filename: str, key: str):
                path = settings.MODELS_DIR / filename
                if path.exists():
                    self._models[key] = joblib.load(path)
                    features = self._models[key].get('features', [])
                    logger.info(f"✓ Loaded {key} model (features: {len(features)})")
                    return True
                else:
                    logger.warning(f"⚠️ Model file not found: {path}")
                    return False

            load_safe(settings.CHURN_MODEL_NAME, 'churn')
            load_safe(settings.CLAIMS_MODEL_NAME, 'claims')
            load_safe(settings.SURVIVAL_MODEL_NAME, 'survival')
            
            logger.info(f"Total models loaded: {len(self._models)}")
            return len(self._models) > 0
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            return False
            
    def get_model(self, model_name: str) -> Any:
        return self._models.get(model_name)

# Singleton instance
model_manager = ModelManager()

# Auto-load on import, but don't crash if it fails (allows for offline testing)
try:
    model_manager.load_models()
except Exception as e:
    logger.error(f"Failed to auto-load models: {e}")

def prepare_churn_features(customer_data: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare features for churn model prediction.
    Maps database columns to model features with proper naming and engineering.
    """
    churn_model = model_manager.get_model('churn')
    if not churn_model:
        raise ValueError("Churn model not loaded")
    
    required_features = churn_model['features']
    
    # Feature engineering and mapping
    features = pd.DataFrame()
    
    # Map database columns to model features
    feature_mapping = {
        'Customer_Age': 'age',
        'Driving_Experience': 'driving_experience',
        'Seniority': 'seniority',
        'Policies_in_force': 'policies_in_force',
        'Max_policies': 'max_policies', 
        'Max_products': 'max_products',
        'Premium': 'premium',
        'Vehicle_Age': 'vehicle_age',
        'Power': 'power',
        'Cylinder_capacity': 'cylinder_capacity',
        'Value_vehicle': 'value_vehicle',
        'N_doors': 'n_doors',
        'Length': 'length',
        'Weight': 'weight',
        'N_claims_history': 'n_claims_history',
        'R_Claims_history': 'r_claims_history',
    }
    
    # Direct mappings
    for model_feat, db_col in feature_mapping.items():
        if db_col in customer_data.columns:
            features[model_feat] = customer_data[db_col]
        elif model_feat in required_features:
            features[model_feat] = 0.0  # Default value
            
    # Calculated features
    if 'Policy_Age' in required_features:
        features['Policy_Age'] = customer_data.get('seniority', 0)
    
    if 'Premium_per_HP' in required_features:
        premium = customer_data.get('premium', 0)
        power = customer_data.get('power', 0)
        features['Premium_per_HP'] = premium / (power + 1)
    
    if 'Value_per_HP' in required_features:
        value = customer_data.get('value_vehicle', 0)
        power = customer_data.get('power', 0)
        features['Value_per_HP'] = value / (power + 1)
    
    if 'Claims_per_Year' in required_features:
        claims = customer_data.get('n_claims_history', 0)
        seniority = customer_data.get('seniority', 0)
        features['Claims_per_Year'] = claims / (seniority + 1)
    
    # Provide defaults to ensure DataFrame shape matches model expectation
    for feat in required_features:
        if feat not in features.columns:
            features[feat] = 0.0
            
    return features[required_features]

def predict_churn_probability(customer_data: pd.DataFrame) -> List[Dict[str, Any]]:
    """Predict churn probability."""
    churn_artifact = model_manager.get_model('churn')
    if not churn_artifact:
        logger.warning("Churn model not loaded. Returning defaults.")
        # Return graceful defaults if model is missing
        return [{
            'customer_id': idx,
            'churn_probability': 0.15, # Industry average default
            'risk_category': 'Unknown',
            'recommended_action': 'Check model configuration',
            'confidence': 0.0
        } for idx in customer_data.index]
    
    try:
        model = churn_artifact['model']
        scaler = churn_artifact['scaler']
        
        # Prepare features
        X = prepare_churn_features(customer_data)
        X_scaled = scaler.transform(X)
        
        # Predict
        churn_probs = model.predict_proba(X_scaled)[:, 1]
        
        results = []
        for idx, prob in enumerate(churn_probs):
            if prob >= 0.7:
                risk_category = "Critical"
                action = "Immediate personal call required"
            elif prob >= 0.5:
                risk_category = "High"
                action = "Schedule call within 48 hours"
            elif prob >= 0.3:
                risk_category = "Medium"
                action = "Email outreach + monitor"
            else:
                risk_category = "Low"
                action = "Standard quarterly check-in"
            
            results.append({
                'customer_id': int(customer_data.iloc[idx].get('customer_id', idx)),
                'churn_probability': round(float(prob), 4),
                'risk_category': risk_category,
                'recommended_action': action,
                'confidence': round(abs(prob - 0.5) * 2, 2)
            })
        return results
        
    except Exception as e:
        logger.error(f"Churn prediction failed: {e}")
        # Return empty list or raise depends on API needs; empty list is safer
        return []

def predict_claims_probability(customer_data: pd.DataFrame) -> List[Dict[str, Any]]:
    """Predict claims lifecycle probability."""
    claims_artifact = model_manager.get_model('claims')
    if not claims_artifact:
        logger.warning("Claims model not loaded")
        return []
    
    try:
        model = claims_artifact['model']
        scaler = claims_artifact.get('scaler')
        features_list = claims_artifact.get('features', [])
        
        # Simple feature alignment (assumes DataFrame has correct columns for now)
        # In production, use a dedicated prepare function similar to churn
        X = pd.DataFrame()
        for f in features_list:
            X[f] = customer_data.get(f, 0.0)
            
        if scaler:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X
            
        probs = model.predict_proba(X_scaled)[:, 1]
        
        results = []
        for idx, prob in enumerate(probs):
             results.append({
                'customer_id': int(customer_data.iloc[idx].get('customer_id', idx)),
                'claim_probability': round(float(prob), 4),
                'risk_level': 'High' if prob > 0.3 else 'Low'
            })
        return results
    except Exception as e:
        logger.error(f"Claims prediction failed: {e}")
        return []

def get_model_info() -> Dict[str, Any]:
    """Get loaded model info."""
    return {
        'models_loaded': len(model_manager._models),
        'available_models': list(model_manager._models.keys())
    }

