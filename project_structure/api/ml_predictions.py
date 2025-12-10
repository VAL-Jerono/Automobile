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

logger = logging.getLogger(__name__)

# Model paths
MODELS_DIR = "/Users/leonida/Documents/automobile_claims/models"
CHURN_MODEL_PATH = os.path.join(MODELS_DIR, "churn_model_20251209_094706.pkl")
CLAIMS_MODEL_PATH = os.path.join(MODELS_DIR, "lifecycle_claim_model_20251209_094706.pkl")
SURVIVAL_MODEL_PATH = os.path.join(MODELS_DIR, "survival_model_20251209_094706.pkl")

# Global model storage
_models_cache = {}

def load_models():
    """Load all ML models into memory."""
    global _models_cache
    
    try:
        logger.info("Loading ML models...")
        
        # Load churn model
        if os.path.exists(CHURN_MODEL_PATH):
            _models_cache['churn'] = joblib.load(CHURN_MODEL_PATH)
            logger.info(f"✓ Loaded churn model (features: {len(_models_cache['churn']['features'])})")
        
        # Load claims model
        if os.path.exists(CLAIMS_MODEL_PATH):
            _models_cache['claims'] = joblib.load(CLAIMS_MODEL_PATH)
            logger.info(f"✓ Loaded lifecycle claims model")
        
        # Load survival model
        if os.path.exists(SURVIVAL_MODEL_PATH):
            _models_cache['survival'] = joblib.load(SURVIVAL_MODEL_PATH)
            logger.info(f"✓ Loaded survival model")
        
        logger.info(f"Total models loaded: {len(_models_cache)}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        return False

def prepare_churn_features(customer_data: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare features for churn model prediction.
    
    Maps database columns to model features with proper naming and engineering.
    """
    if 'churn' not in _models_cache:
        raise ValueError("Churn model not loaded")
    
    required_features = _models_cache['churn']['features']
    
    # Feature engineering and mapping
    features = pd.DataFrame()
    
    # Map database columns to model features
    feature_mapping = {
        'Customer_Age': 'age',
        'Driving_Experience': 'driving_experience',
        'Seniority': 'seniority',
        'Policies_in_force': 'policies_in_force',
        'Max_policies': 'max_policies' if 'max_policies' in customer_data.columns else None,
        'Max_products': 'max_products' if 'max_products' in customer_data.columns else None,
        'Premium': 'premium',
        'Vehicle_Age': 'vehicle_age',
        'Power': 'power',
        'Cylinder_capacity': 'cylinder_capacity',
        'Value_vehicle': 'value_vehicle',
        'N_doors': 'n_doors' if 'n_doors' in customer_data.columns else None,
        'Length': 'length' if 'length' in customer_data.columns else None,
        'Weight': 'weight' if 'weight' in customer_data.columns else None,
        'N_claims_history': 'n_claims_history',
        'R_Claims_history': 'r_claims_history' if 'r_claims_history' in customer_data.columns else None,
    }
    
    # Direct mappings
    for model_feat, db_col in feature_mapping.items():
        if db_col and db_col in customer_data.columns:
            features[model_feat] = customer_data[db_col]
        elif model_feat in required_features:
            features[model_feat] = 0  # Default value
    
    # Calculated features
    if 'Policy_Age' in required_features:
        # Approximate policy age from seniority
        features['Policy_Age'] = customer_data['seniority'] if 'seniority' in customer_data.columns else 0
    
    if 'Premium_per_HP' in required_features:
        if 'premium' in customer_data.columns and 'power' in customer_data.columns:
            features['Premium_per_HP'] = customer_data['premium'] / (customer_data['power'] + 1)  # +1 to avoid division by zero
        else:
            features['Premium_per_HP'] = 0
    
    if 'Value_per_HP' in required_features:
        if 'value_vehicle' in customer_data.columns and 'power' in customer_data.columns:
            features['Value_per_HP'] = customer_data['value_vehicle'] / (customer_data['power'] + 1)
        else:
            features['Value_per_HP'] = 0
    
    if 'Claims_per_Year' in required_features:
        if 'n_claims_history' in customer_data.columns and 'seniority' in customer_data.columns:
            features['Claims_per_Year'] = customer_data['n_claims_history'] / (customer_data['seniority'] + 1)
        else:
            features['Claims_per_Year'] = 0
    
    # Encoded features (simplified - using numeric values from DB)
    encoded_features = {
        'Distribution_channel_encoded': 'distribution_channel',
        'Payment_encoded': 'payment',
        'Area_encoded': 'area',
        'Second_driver_encoded': 'second_driver',
        'Type_fuel_encoded': 'type_risk',  # Using type_risk as proxy
    }
    
    for model_feat, db_col in encoded_features.items():
        if model_feat in required_features:
            if db_col in customer_data.columns:
                features[model_feat] = customer_data[db_col]
            else:
                features[model_feat] = 0
    
    # Ensure all required features are present
    for feat in required_features:
        if feat not in features.columns:
            features[feat] = 0
            logger.warning(f"Feature '{feat}' not found, using default value 0")
    
    return features[required_features]

def predict_churn_probability(customer_data: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Predict churn probability for given customers.
    
    Args:
        customer_data: DataFrame with customer features
        
    Returns:
        List of predictions with customer_id, churn_probability, risk_category
    """
    if 'churn' not in _models_cache:
        raise ValueError("Churn model not loaded. Call load_models() first.")
    
    try:
        churn_artifact = _models_cache['churn']
        model = churn_artifact['model']
        scaler = churn_artifact['scaler']
        threshold = churn_artifact.get('optimal_threshold', 0.35)
        
        # Prepare features
        X = prepare_churn_features(customer_data)
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Predict probabilities
        churn_probs = model.predict_proba(X_scaled)[:, 1]
        
        # Create results
        results = []
        for idx, prob in enumerate(churn_probs):
            # Determine risk category
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
            
            result = {
                'customer_id': int(customer_data.iloc[idx].get('customer_id', idx)),
                'policy_id': int(customer_data.iloc[idx].get('policy_id', 0)),
                'churn_probability': round(float(prob), 4),
                'risk_category': risk_category,
                'recommended_action': action,
                'confidence': round(abs(prob - 0.5) * 2, 2),  # 0-1 confidence
                'model_version': '2.0_20251209'
            }
            
            # Add premium value if available
            if 'premium' in customer_data.columns:
                result['premium'] = float(customer_data.iloc[idx]['premium'])
                result['risk_weighted_value'] = round(float(prob * customer_data.iloc[idx]['premium']), 2)
            
            results.append(result)
        
        return results
        
    except Exception as e:
        logger.error(f"Churn prediction failed: {e}")
        raise

def predict_claims_probability(customer_data: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Predict claims probability using lifecycle claims model.
    
    Returns:
        List of predictions with customer_id, claim_probability, expected_cost
    """
    if 'claims' not in _models_cache:
        logger.warning("Claims model not loaded, returning empty predictions")
        return []
    
    try:
        claims_artifact = _models_cache['claims']
        model = claims_artifact['model']
        scaler = claims_artifact.get('scaler')
        
        # Prepare features (similar process to churn)
        features = _models_cache['claims'].get('features', [])
        X = customer_data[features] if all(f in customer_data.columns for f in features) else customer_data
        
        if scaler:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X
        
        # Predict
        claim_probs = model.predict_proba(X_scaled)[:, 1]
        
        results = []
        for idx, prob in enumerate(claim_probs):
            # Estimate claim cost based on probability and average claim amount
            avg_claim_cost = 5000  # Average from data
            expected_cost = prob * avg_claim_cost
            
            results.append({
                'customer_id': int(customer_data.iloc[idx].get('customer_id', idx)),
                'claim_probability': round(float(prob), 4),
                'expected_claim_cost': round(expected_cost, 2),
                'risk_level': 'High' if prob > 0.3 else 'Medium' if prob > 0.15 else 'Low'
            })
        
        return results
        
    except Exception as e:
        logger.error(f"Claims prediction failed: {e}")
        return []

def calculate_customer_segments(churn_probs: List[float], premiums: List[float]) -> List[str]:
    """
    Segment customers based on churn probability and premium value.
    
    Segments:
    - Gold Tier: Low churn (<30%), high premium (>$500)
    - At-Risk High-Value: High churn (>50%), high premium (>$500)
    - Stable Base: Low churn (<30%), medium premium ($200-$500)
    - Immediate Attention: High churn (>50%), any premium
    """
    segments = []
    
    for churn, premium in zip(churn_probs, premiums):
        if churn < 0.3 and premium > 500:
            segment = "Gold Tier"
        elif churn > 0.5 and premium > 500:
            segment = "At-Risk High-Value"
        elif churn > 0.5:
            segment = "Immediate Attention"
        elif churn < 0.3:
            segment = "Stable Base"
        else:
            segment = "Standard"
        
        segments.append(segment)
    
    return segments

def get_model_info() -> Dict[str, Any]:
    """Get information about loaded models."""
    info = {
        'models_loaded': len(_models_cache),
        'available_models': list(_models_cache.keys()),
        'model_details': {}
    }
    
    for name, artifact in _models_cache.items():
        if isinstance(artifact, dict):
            info['model_details'][name] = {
                'features_count': len(artifact.get('features', [])),
                'created': artifact.get('created', 'Unknown'),
                'description': artifact.get('description', 'No description'),
                'metrics': artifact.get('metrics', {})
            }
    
    return info

# Initialize models on module import
try:
    load_models()
except Exception as e:
    logger.error(f"Failed to auto-load models: {e}")
