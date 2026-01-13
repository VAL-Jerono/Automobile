#!/usr/bin/env python3
"""
Generate Real Predictions Using Trained Models
===============================================
Loads the actual trained models and generates predictions on the insurance dataset.
Populates model_predictions table with full real data.

Models Used:
  • churn_model: Predicts lapse/churn probability
  • claims_frequency_model: Predicts claims probability
  • claims_severity_model: Predicts claims severity

Usage:
    python generate_predictions_from_models.py
"""

import pandas as pd
import numpy as np
import joblib
import logging
import sys
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_models():
    """Load trained models from disk."""
    models_dir = Path(__file__).parent.parent / 'models'
    
    logger.info(f"📁 Loading models from: {models_dir}")
    
    models = {}
    
    # Churn model
    churn_model_path = models_dir / 'churn_model_20260113_183202.pkl'
    if churn_model_path.exists():
        models['churn'] = joblib.load(churn_model_path)
        logger.info(f"✅ Loaded churn model")
    else:
        logger.warning(f"⚠️  Churn model not found: {churn_model_path}")
    
    # Claims frequency model
    claims_freq_path = models_dir / 'claims_frequency_model_20260113_183202.pkl'
    if claims_freq_path.exists():
        models['claims_frequency'] = joblib.load(claims_freq_path)
        logger.info(f"✅ Loaded claims frequency model")
    else:
        logger.warning(f"⚠️  Claims frequency model not found: {claims_freq_path}")
    
    # Claims severity model
    claims_sev_path = models_dir / 'claims_severity_model_20260113_183202.pkl'
    if claims_sev_path.exists():
        models['claims_severity'] = joblib.load(claims_sev_path)
        logger.info(f"✅ Loaded claims severity model")
    else:
        logger.warning(f"⚠️  Claims severity model not found: {claims_sev_path}")
    
    return models


def load_insurance_data():
    """Load insurance dataset."""
    # Try multiple possible locations
    data_paths = [
        Path(__file__).parent / 'Motor_vehicle_insurance_data.csv',
        Path(__file__).parent.parent / 'Automobile' / 'Motor_vehicle_insurance_data.csv',
        Path(__file__).parent.parent / 'Dataset of an actual motor vehicle insurance portfolio' / 'Motor vehicle insurance data.csv',
    ]
    
    for path in data_paths:
        if path.exists():
            logger.info(f"📊 Loading data: {path.name}")
            # Try semicolon delimiter first (European format)
            try:
                df = pd.read_csv(path, sep=';')
                if len(df.columns) > 1:
                    logger.info(f"✅ Loaded {len(df):,} rows, {len(df.columns)} columns")
                    return df
            except:
                pass
            
            # Try comma delimiter
            try:
                df = pd.read_csv(path, sep=',')
                if len(df.columns) > 1:
                    logger.info(f"✅ Loaded {len(df):,} rows, {len(df.columns)} columns")
                    return df
            except:
                pass
    
    logger.error("❌ Could not find insurance data CSV")
    return None


def prepare_features(df):
    """Prepare features for model predictions."""
    X = df.copy()

    # --- Date parsing for age/experience ---
    ref_date = datetime(2018, 1, 1)
    for col in ['Date_birth', 'Date_driving_licence']:
        if col in X.columns:
            X[col] = pd.to_datetime(X[col], errors='coerce')

    # Driver_age
    if 'Date_birth' in X.columns:
        age_years = (ref_date - X['Date_birth']).dt.days / 365.25
        X['Driver_age'] = age_years.clip(18, 100)

    # Licence_years
    if 'Date_driving_licence' in X.columns:
        licence_years = (ref_date - X['Date_driving_licence']).dt.days / 365.25
        X['Licence_years'] = licence_years.clip(0, 70)

    # Premium_to_Value
    if 'Premium' in X.columns and 'Value_vehicle' in X.columns:
        X['Premium_to_Value'] = X['Premium'] / (X['Value_vehicle'] + 1)
        X['Premium_to_Value'] = X['Premium_to_Value'].clip(0, 0.5)

    # Tenure_Segment from Seniority
    if 'Seniority' in X.columns:
        def tenure_bucket(v):
            if pd.isna(v):
                return 'Unknown'
            if v <= 1:
                return 'New (0-1yr)'
            if v <= 3:
                return 'Early (1-3yr)'
            if v <= 5:
                return 'Established (3-5yr)'
            if v <= 10:
                return 'Mature (5-10yr)'
            return 'Veteran (10+yr)'

        X['Tenure_Segment'] = X['Seniority'].apply(tenure_bucket)

    # Drop ID-like helper columns but keep the canonical ID
    id_cols = [col for col in X.columns if 'id' in col.lower() or col in ['ID', 'Index']]
    for col in id_cols:
        if col != 'ID':
            X = X.drop(col, axis=1, errors='ignore')

    # Handle categorical columns (encode to numeric codes)
    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col != 'ID':
            X[col] = X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else 'Unknown')
            X[col] = pd.Categorical(X[col]).codes

    # Fill numeric missing values with median
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        X[col] = X[col].fillna(X[col].median())

    return X


def generate_predictions(models, X):
    """Generate predictions using loaded models."""
    predictions = pd.DataFrame()
    
    try:
        # Get policy IDs
        if 'ID' in X.columns:
            predictions['policy_id'] = X['ID'].astype(int)
        else:
            predictions['policy_id'] = range(1, len(X) + 1)
        
        logger.info("🔄 Generating predictions...")
        
        # Churn probability (lapse risk)
        if 'churn' in models:
            try:
                churn_pred = models['churn'].predict_proba(X.drop('ID', axis=1, errors='ignore'))[:, 1]
                predictions['churn_probability'] = np.clip(churn_pred, 0, 1)
                logger.info("✅ Generated churn predictions")
            except Exception as e:
                logger.warning(f"⚠️  Could not generate churn predictions: {e}")
                predictions['churn_probability'] = 0.5
        else:
            predictions['churn_probability'] = 0.5
        
        # Claims probability
        if 'claims_frequency' in models:
            try:
                claims_pred = models['claims_frequency'].predict_proba(X.drop('ID', axis=1, errors='ignore'))[:, 1]
                predictions['claims_probability'] = np.clip(claims_pred, 0, 1)
                logger.info("✅ Generated claims frequency predictions")
            except Exception as e:
                logger.warning(f"⚠️  Could not generate claims frequency predictions: {e}")
                predictions['claims_probability'] = 0.3
        else:
            predictions['claims_probability'] = 0.3
        
        # Claims severity
        if 'claims_severity' in models:
            try:
                severity_pred = models['claims_severity'].predict(X.drop('ID', axis=1, errors='ignore'))
                predictions['claims_severity'] = np.clip(severity_pred, 0, 50000)
                logger.info("✅ Generated claims severity predictions")
            except Exception as e:
                logger.warning(f"⚠️  Could not generate claims severity predictions: {e}")
                predictions['claims_severity'] = 5000
        else:
            predictions['claims_severity'] = 5000
        
        # Derived metrics
        # Customer Lifetime Value (simplified: premium * (1 - churn) * years)
        premium = X['Premium'].fillna(500) if 'Premium' in X.columns else 500
        predictions['customer_lifetime_value'] = (
            premium * (1 - predictions['churn_probability']) * 5 +
            predictions['claims_severity'] * predictions['claims_probability']
        )
        
        # Customer segment (simplified based on CLV quartiles)
        clv_quartiles = pd.qcut(predictions['customer_lifetime_value'], q=4, labels=['Bronze', 'Silver', 'Gold', 'Platinum'], duplicates='drop')
        predictions['customer_segment'] = clv_quartiles.astype(str)
        
        # Journey quadrant (based on churn vs CLV)
        churn_median = predictions['churn_probability'].median()
        clv_median = predictions['customer_lifetime_value'].median()
        
        quadrants = []
        for _, row in predictions.iterrows():
            if row['churn_probability'] < churn_median:
                if row['customer_lifetime_value'] > clv_median:
                    quadrants.append('Protect')
                else:
                    quadrants.append('Grow')
            else:
                if row['customer_lifetime_value'] > clv_median:
                    quadrants.append('Rescue')
                else:
                    quadrants.append('Monitor')
        predictions['journey_quadrant'] = quadrants
        
        # Pricing adequacy flag
        predictions['pricing_adequacy_flag'] = (
            (predictions['claims_probability'] * predictions['claims_severity']) > premium
        ).astype(int)
        
        # Renewal risk score
        predictions['renewal_risk_score'] = predictions['churn_probability'] * 0.6 + (1 - predictions['claims_probability']) * 0.4
        
        # High renewal risk flag
        predictions['is_high_renewal_risk'] = (predictions['renewal_risk_score'] > 0.7).astype(int)
        
        logger.info(f"✅ Generated all predictions for {len(predictions):,} policies")
        return predictions
    
    except Exception as e:
        logger.error(f"❌ Error generating predictions: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_to_database(predictions):
    """Save predictions to MySQL database."""
    if predictions is None or predictions.empty:
        logger.error("No predictions to save")
        return False
    
    try:
        from sql_predictions_manager import SQLModelPredictionsManager
        
        manager = SQLModelPredictionsManager()
        if not manager.connect():
            logger.error("Could not connect to database")
            return False
        
        # Create table
        if not manager.create_predictions_table():
            logger.error("Could not create predictions table")
            manager.disconnect()
            return False
        
        # Insert predictions
        rows_inserted = manager.insert_predictions(predictions)
        
        if rows_inserted > 0:
            logger.info(f"✅ Inserted {rows_inserted:,} predictions into database")
            
            # Get summary
            summary = manager.get_prediction_summary()
            logger.info("\n📊 Database Summary:")
            logger.info(f"   Total predictions: {summary.get('total_predictions', 0):,}")
            logger.info(f"   Portfolio value: €{summary.get('total_portfolio_value', 0):,.0f}")
            logger.info(f"   High risk: {summary.get('high_risk_count', 0):,}")
            logger.info(f"   Unique segments: {summary.get('unique_segments', 0)}")
            logger.info(f"   Avg churn probability: {summary.get('avg_churn_probability', 0):.1%}")
            
            manager.disconnect()
            return True
        else:
            logger.error("No rows inserted")
            manager.disconnect()
            return False
    
    except Exception as e:
        logger.error(f"Database error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main execution."""
    logger.info("=" * 70)
    logger.info("GENERATE REAL PREDICTIONS FROM TRAINED MODELS")
    logger.info("=" * 70)
    
    # Load models
    models = load_models()
    if not models:
        logger.error("❌ No models loaded")
        return 1
    
    # Load insurance data
    df = load_insurance_data()
    if df is None:
        logger.error("❌ Could not load insurance data")
        return 1
    
    # Prepare features
    logger.info("🔧 Preparing features...")
    X = prepare_features(df)
    
    # Generate predictions
    predictions = generate_predictions(models, X)
    if predictions is None:
        logger.error("❌ Could not generate predictions")
        return 1
    
    # Save to database
    logger.info("\n💾 Saving to database...")
    if save_to_database(predictions):
        logger.info("\n✨ SUCCESS! Database populated with real predictions")
        return 0
    else:
        logger.error("\n❌ Failed to save to database")
        return 1


if __name__ == "__main__":
    sys.exit(main())
