"""
Ensemble ML model combining RandomForest, GradientBoosting, and Neural Networks.
Provides predictions and SHAP-based explanations.
Fallback to scikit-learn when XGBoost/LightGBM unavailable (libomp dependency).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except Exception as e:
    XGB_AVAILABLE = False
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except Exception as e:
    LGB_AVAILABLE = False
    print(f"⚠️  LightGBM unavailable, using sklearn alternatives")

import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score
try:
    import tensorflow as tf
    from tensorflow.keras import Sequential, layers
    TF_AVAILABLE = True
except Exception as e:
    TF_AVAILABLE = False
    print(f"⚠️  TensorFlow unavailable, using sklearn ensemble only")
from typing import Tuple, Dict, Any
try:
    import shap
    SHAP_AVAILABLE = True
except Exception as e:
    SHAP_AVAILABLE = False
import logging

logger = logging.getLogger(__name__)

class InsuranceEnsembleModel:
    """
    Ensemble model for insurance risk prediction.
    Combines tree models and Neural Networks with equal weighting.
    Falls back to sklearn when XGBoost/LightGBM unavailable.
    """
    
    def __init__(self, config_path: str = 'config.yaml'):
        self.xgb_model = None
        self.lgb_model = None
        self.rf_model = None
        self.gb_model = None
        self.nn_model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        self.model_dir = 'models'
        
    def preprocess(self, X: pd.DataFrame, y: pd.Series = None, fit: bool = True):
        """Preprocess features: encode categoricals, scale numerics."""
        X_proc = X.copy()
        
        # Encode categorical columns
        categorical_cols = X_proc.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if fit:
                self.label_encoders[col] = LabelEncoder()
                X_proc[col] = self.label_encoders[col].fit_transform(X_proc[col].astype(str))
            else:
                X_proc[col] = self.label_encoders[col].transform(X_proc[col].astype(str))
        
        # Scale numeric columns
        numeric_cols = X_proc.select_dtypes(include=[np.number]).columns
        if fit:
            X_proc[numeric_cols] = self.scaler.fit_transform(X_proc[numeric_cols])
        else:
            X_proc[numeric_cols] = self.scaler.transform(X_proc[numeric_cols])
        
        return X_proc
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: pd.DataFrame = None, y_val: pd.Series = None):
        """Train ensemble model components."""
        logger.info("Training ensemble model...")
        
        # Preprocess
        X_train_proc = self.preprocess(X_train, y_train, fit=True)
        self.feature_names = X_train_proc.columns.tolist()
        
        if X_val is not None:
            X_val_proc = self.preprocess(X_val, fit=False)
        
        # XGBoost
        if XGB_AVAILABLE:
            logger.info("Training XGBoost...")
            self.xgb_model = xgb.XGBClassifier(
                n_estimators=10, learning_rate=0.1, max_depth=6, subsample=0.8,
                colsample_bytree=0.8, random_state=42, n_jobs=-1
            )
            self.xgb_model.fit(X_train_proc, y_train, eval_set=[(X_val_proc, y_val)] if X_val is not None else None)
        else:
            logger.warning("XGBoost not available, using RandomForest instead")
            self.rf_model = RandomForestClassifier(
                n_estimators=10, max_depth=10, random_state=42, n_jobs=-1
            )
            self.rf_model.fit(X_train_proc, y_train)
        
        # LightGBM or GradientBoosting
        if LGB_AVAILABLE:
            logger.info("Training LightGBM...")
            self.lgb_model = lgb.LGBMClassifier(
                n_estimators=10, learning_rate=0.1, num_leaves=31, random_state=42, n_jobs=-1
            )
            self.lgb_model.fit(X_train_proc, y_train, eval_set=[(X_val_proc, y_val)] if X_val is not None else None)
        else:
            logger.warning("LightGBM not available, using GradientBoosting instead")
            self.gb_model = GradientBoostingClassifier(
                n_estimators=10, learning_rate=0.1, max_depth=5, random_state=42
            )
            self.gb_model.fit(X_train_proc, y_train)
        
        # Neural Network (optional)
        if TF_AVAILABLE:
            logger.info("Training Neural Network...")
            self.nn_model = Sequential([
                layers.Dense(128, activation='relu', input_shape=(len(self.feature_names),)),
                layers.Dropout(0.3),
                layers.Dense(64, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(32, activation='relu'),
                layers.Dropout(0.2),
                layers.Dense(1, activation='sigmoid')
            ])
            self.nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['auc'])
            self.nn_model.fit(X_train_proc, y_train, epochs=50, batch_size=32, 
                             validation_data=(X_val_proc, y_val) if X_val is not None else None,
                             verbose=0)
        else:
            logger.warning("TensorFlow not available, skipping Neural Network training")
        
        logger.info("✓ Ensemble training complete")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using ensemble average."""
        X_proc = self.preprocess(X, fit=False)
        
        predictions = []
        
        if XGB_AVAILABLE and self.xgb_model is not None:
            xgb_pred = self.xgb_model.predict_proba(X_proc)[:, 1]
            predictions.append(xgb_pred)
        elif self.rf_model is not None:
            rf_pred = self.rf_model.predict_proba(X_proc)[:, 1]
            predictions.append(rf_pred)
        
        if LGB_AVAILABLE and self.lgb_model is not None:
            lgb_pred = self.lgb_model.predict_proba(X_proc)[:, 1]
            predictions.append(lgb_pred)
        elif self.gb_model is not None:
            gb_pred = self.gb_model.predict_proba(X_proc)[:, 1]
            predictions.append(gb_pred)
        
        if self.nn_model is not None:
            nn_pred = self.nn_model.predict(X_proc, verbose=0).flatten()
            predictions.append(nn_pred)
        
        # Ensemble: average of available models
        ensemble_pred = np.mean(predictions, axis=0)
        return ensemble_pred
    
    def explain(self, X: pd.DataFrame, instance_idx: int = 0) -> Dict[str, Any]:
        """Generate SHAP explanation for a prediction."""
        X_proc = self.preprocess(X, fit=False)
        
        # SHAP for available models (prefer XGBoost/LightGBM if available)
        explainer = None
        shap_array = None
        
        if SHAP_AVAILABLE:
            if XGB_AVAILABLE and self.xgb_model is not None:
                explainer = shap.TreeExplainer(self.xgb_model)
            elif LGB_AVAILABLE and self.lgb_model is not None:
                explainer = shap.TreeExplainer(self.lgb_model)
            elif self.rf_model is not None:
                explainer = shap.TreeExplainer(self.rf_model)
            elif self.gb_model is not None:
                explainer = shap.TreeExplainer(self.gb_model)
            
            if explainer:
                shap_values = explainer.shap_values(X_proc.iloc[[instance_idx]])
                shap_array = shap_values[0] if isinstance(shap_values, (list, tuple)) else shap_values[0] if len(shap_values.shape) > 1 else shap_values
        
        if shap_array is None:
            # Fallback: use feature importance from available model
            if self.rf_model:
                shap_array = self.rf_model.feature_importances_
            elif self.gb_model:
                shap_array = self.gb_model.feature_importances_
            else:
                shap_array = np.ones(len(self.feature_names))
        
        top_features = pd.DataFrame({
            'feature': self.feature_names,
            'shap_value': shap_array
        }).abs().nlargest(5, 'shap_value')
        
        return {
            'prediction': float(self.predict(X.iloc[[instance_idx]])[0]),
            'top_features': top_features.to_dict('records'),
            'shap_values': shap_values[0].tolist()
        }
    
    def save(self, path: str = None):
        """Save models."""
        path = path or self.model_dir
        joblib.dump(self.xgb_model, f'{path}/xgb_model.pkl')
        joblib.dump(self.lgb_model, f'{path}/lgb_model.pkl')
        self.nn_model.save(f'{path}/nn_model.h5')
        joblib.dump(self.scaler, f'{path}/scaler.pkl')
        joblib.dump(self.label_encoders, f'{path}/label_encoders.pkl')
        logger.info(f"✓ Models saved to {path}")
    
    def load(self, path: str = None):
        """Load models."""
        path = path or self.model_dir
        self.xgb_model = joblib.load(f'{path}/xgb_model.pkl')
        self.lgb_model = joblib.load(f'{path}/lgb_model.pkl')
        self.nn_model = tf.keras.models.load_model(f'{path}/nn_model.h5')
        self.scaler = joblib.load(f'{path}/scaler.pkl')
        self.label_encoders = joblib.load(f'{path}/label_encoders.pkl')
        logger.info(f"✓ Models loaded from {path}")
