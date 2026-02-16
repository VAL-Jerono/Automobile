"""
Model Manager - Handles loading and inference for production ML models
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Any, Optional
import pickle
import glob

logger = logging.getLogger(__name__)

class ModelManager:
    """Manages all ML models with lazy loading from production_models folder"""
    
    def __init__(self, model_dir: Optional[Path] = None):
        """Initialize Model Manager with production models"""
        if model_dir is None:
            self.model_dir = Path(__file__).parent.parent / "production_models"
        else:
            self.model_dir = Path(model_dir)
        
        self._churn_model = None
        self._claims_frequency_model = None
        self._claims_severity_model = None
        self._clv_model = None
        
        self.churn_model_path = self._find_latest_model('churn_model')
        self.claims_frequency_model_path = self._find_latest_model('claims_frequency_model')
        self.claims_severity_model_path = self._find_latest_model('claims_severity_model')
        
        logger.info(f"Model Manager initialized. Model directory: {self.model_dir}")
    
    def _find_latest_model(self, model_name: str) -> Optional[Path]:
        """Find the latest model file by timestamp"""
        pattern = str(self.model_dir / f"{model_name}_*.pkl")
        files = glob.glob(pattern)
        return Path(sorted(files)[-1]) if files else None
    
    @property
    def churn_model(self):
        if self._churn_model is None:
            self._churn_model = self._load_model(self.churn_model_path)
        return self._churn_model
    
    @property
    def claims_frequency_model(self):
        if self._claims_frequency_model is None:
            self._claims_frequency_model = self._load_model(self.claims_frequency_model_path)
        return self._claims_frequency_model
    
    @property
    def claims_severity_model(self):
        if self._claims_severity_model is None:
            self._claims_severity_model = self._load_model(self.claims_severity_model_path)
        return self._claims_severity_model
    
    @property
    def clv_model(self):
        return None
    
    def _load_model(self, model_path: Optional[Path]):
        """Load model from pickle file"""
        try:
            if model_path is None or not model_path.exists():
                logger.error(f"Model file not found: {model_path}")
                return None
            
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"Model loaded: {model_path.name}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return None
    
    def predict_churn(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        if self.churn_model is None:
            raise ValueError("Churn model not loaded")
        
        try:
            # Extract the actual model from the dictionary
            model_dict = self.churn_model
            model = model_dict.get('model')
            
            features_df = pd.DataFrame([input_data])
            churn_prob = float(model.predict_proba(features_df)[0][1])
            
            if churn_prob >= 0.75:
                risk = "CRITICAL"
            elif churn_prob >= 0.50:
                risk = "HIGH"
            elif churn_prob >= 0.25:
                risk = "MEDIUM"
            else:
                risk = "LOW"
            
            segment = "PREMIUM" if input_data.get('premium', 0) > 800 else "STANDARD"
            
            return {
                'churn_probability': churn_prob,
                'risk_category': risk,
                'segment': segment,
                'confidence': 0.85
            }
        except Exception as e:
            logger.error(f"Error in churn prediction: {str(e)}")
            raise
    
    def predict_claims_frequency(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        if self.claims_frequency_model is None:
            raise ValueError("Claims frequency model not loaded")
        
        try:
            # Extract the actual model from the dictionary
            model_dict = self.claims_frequency_model
            model = model_dict.get('model')
            scaler = model_dict.get('scaler')
            
            features_df = pd.DataFrame([input_data])
            
            # Scale if available
            if scaler is not None:
                feature_names = model_dict.get('feature_names', [])
                available_features = [f for f in feature_names if f in features_df.columns]
                if available_features:
                    features_df[available_features] = scaler.transform(features_df[available_features])
            
            claims_prob = float(model.predict_proba(features_df)[0][1])
            
            risk = "HIGH" if claims_prob >= 0.30 else ("MEDIUM" if claims_prob >= 0.15 else "LOW")
            
            return {
                'claims_probability': claims_prob,
                'expected_claims_per_year': claims_prob * 1.5,
                'risk_level': risk,
                'confidence': 0.88
            }
        except Exception as e:
            logger.error(f"Error in claims frequency prediction: {str(e)}")
            raise
    
    def predict_claims_severity(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        if self.claims_severity_model is None:
            raise ValueError("Claims severity model not loaded")
        
        try:
            # Extract the actual model from the dictionary
            model_dict = self.claims_severity_model
            model = model_dict.get('model')
            scaler = model_dict.get('scaler')
            
            features_df = pd.DataFrame([input_data])
            
            # Scale if available
            if scaler is not None:
                feature_names = model_dict.get('feature_names', [])
                available_features = [f for f in feature_names if f in features_df.columns]
                if available_features:
                    features_df[available_features] = scaler.transform(features_df[available_features])
            
            severity = float(model.predict(features_df)[0])
            
            category = "SEVERE" if severity >= 5000 else ("MODERATE" if severity >= 2000 else "MINOR")
            
            return {
                'predicted_severity': severity,
                'severity_category': category,
                'confidence': 0.82
            }
        except Exception as e:
            logger.error(f"Error in severity prediction: {str(e)}")
            raise
    
    def predict_clv(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate Customer Lifetime Value using churn and claims models
        
        Formula:
        1. Survival probability: P(Survive_t) = (1 - P(Churn))^t
        2. Expected claims: E(Claims) = P(Claim) × E(Severity|Claim)
        3. Net cash flow: NetCF_t = Premium × 0.75 - E(Claims) - €20
        4. Present value: PV_t = NetCF_t × P(Survive_t) × (1.05)^-t
        5. Total CLV: Σ(PV_t) - €100 acquisition cost
        """
        try:
            # Get churn probability
            churn_prediction = self.predict_churn(input_data)
            churn_prob = churn_prediction['churn_probability']
            
            # Get claims predictions
            claims_freq = self.predict_claims_frequency(input_data)
            claims_severity = self.predict_claims_severity(input_data)
            
            claim_probability = claims_freq['claims_probability']
            expected_severity = claims_severity['predicted_severity']
            
            # Parameters
            premium = input_data.get('premium', 500)
            time_horizon = 10  # 10 years
            discount_rate = 0.05  # 5% discount rate
            profit_margin = 0.75  # 75% of premium retained
            admin_cost = 20  # €20 annual admin cost
            acquisition_cost = 100  # €100 acquisition cost
            
            # Calculate CLV
            clv_total = 0
            annual_cash_flows = []
            
            for year in range(1, time_horizon + 1):
                # 1. Survival probability for year t
                survival_prob = (1 - churn_prob) ** year
                
                # 2. Expected annual claims cost
                expected_claims_cost = claim_probability * expected_severity
                
                # 3. Net cash flow for the year
                net_cash_flow = (premium * profit_margin) - expected_claims_cost - admin_cost
                
                # 4. Present value of cash flow
                discount_factor = (1 + discount_rate) ** (-year)
                present_value = net_cash_flow * survival_prob * discount_factor
                
                clv_total += present_value
                annual_cash_flows.append({
                    'year': year,
                    'survival_prob': round(survival_prob, 4),
                    'net_cash_flow': round(net_cash_flow, 2),
                    'present_value': round(present_value, 2)
                })
            
            # 5. Subtract acquisition cost
            clv_final = clv_total - acquisition_cost
            
            # Determine value tier
            if clv_final >= 5000:
                tier = "PLATINUM"
            elif clv_final >= 2000:
                tier = "GOLD"
            elif clv_final >= 500:
                tier = "SILVER"
            else:
                tier = "BRONZE"
            
            # Calculate total expected revenue and lifetime
            total_revenue = premium * sum([cf['survival_prob'] for cf in annual_cash_flows])
            expected_lifetime = sum([cf['survival_prob'] for cf in annual_cash_flows])
            
            return {
                'predicted_clv': round(clv_final, 2),
                'value_tier': tier,
                'expected_lifetime_years': round(expected_lifetime, 2),
                'total_expected_revenue': round(total_revenue, 2),
                'annual_premium': premium,
                'churn_probability': churn_prob,
                'expected_claims_cost': round(claim_probability * expected_severity, 2),
                'acquisition_cost': acquisition_cost,
                'npv_cash_flows': round(clv_total, 2),
                'annual_projections': annual_cash_flows[:5],  # First 5 years
                'confidence': 0.82,
                'methodology': 'Integrated churn-claims model with 10-year DCF'
            }
            
        except Exception as e:
            logger.error(f"Error in CLV calculation: {str(e)}")
            # Fallback to simple calculation if models fail
            premium = input_data.get('premium', 500)
            tenure = input_data.get('tenure', 3)
            simple_clv = premium * tenure * 0.6 - 100
            
            return {
                'predicted_clv': round(simple_clv, 2),
                'value_tier': "SILVER",
                'confidence': 0.50,
                'methodology': 'Fallback simple calculation',
                'error': str(e)
            }
