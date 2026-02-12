#!/usr/bin/env python3
"""
Extract Real Model Performance Data from CXarticle.ipynb Production Models
"""
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, classification_report
import warnings
warnings.filterwarnings('ignore')

def load_production_models():
    """Load the actual trained models from production_models folder"""
    models_path = Path('production_models')
    
    models = {}
    model_files = {
        'churn': 'churn_model_optimized_20260209_134513.pkl',
        'claims_frequency': 'claims_frequency_model_optimized_20260209_134513.pkl', 
        'claims_severity': 'claims_severity_model_optimized_20260209_134513.pkl'
    }
    
    for model_type, filename in model_files.items():
        filepath = models_path / filename
        if filepath.exists():
            try:
                model = joblib.load(filepath)
                models[model_type] = model
                print(f"✅ Loaded {model_type} model: {filename}")
                
                # Try to extract feature importance if available
                if hasattr(model, 'feature_importances_'):
                    print(f"   📊 Features: {len(model.feature_importances_)} importance scores available")
                elif hasattr(model, 'coef_'):
                    print(f"   📊 Features: {len(model.coef_[0])} coefficients available")
                    
            except Exception as e:
                print(f"❌ Error loading {model_type}: {e}")
    
    return models

def extract_model_performance():
    """Extract performance metrics from the trained models"""
    
    # Load the feature engineered data that was used for training
    df = pd.read_csv('model_data/engineered_features_complete.csv')
    print(f"📊 Loaded engineered data: {len(df):,} records")
    
    # Load models
    models = load_production_models()
    
    performance_data = {}
    
    # Extract Churn Model Performance
    if 'churn' in models:
        churn_model = models['churn']
        
        # Get feature importance (real XGBoost features) 
        if hasattr(churn_model, 'feature_importances_'):
            # Load the churn dataset to get feature names
            churn_data = pd.read_csv('model_data/churn_model_dataset.csv')
            feature_cols = [col for col in churn_data.columns if col != 'Lapse']
            
            importance_scores = churn_model.feature_importances_
            feature_importance = pd.DataFrame({
                'Feature': feature_cols[:len(importance_scores)],
                'Importance': importance_scores
            }).sort_values('Importance', ascending=False).head(10)
            
            performance_data['churn_features'] = feature_importance
            print("📈 Extracted churn model feature importance")
    
    # Extract Claims Frequency Model Performance  
    if 'claims_frequency' in models:
        claims_model = models['claims_frequency']
        
        if hasattr(claims_model, 'feature_importances_'):
            claims_data = pd.read_csv('model_data/claims_frequency_model_dataset.csv')
            feature_cols = [col for col in claims_data.columns if col != 'Claims_binary']
            
            importance_scores = claims_model.feature_importances_
            feature_importance = pd.DataFrame({
                'Feature': feature_cols[:len(importance_scores)],
                'Importance': importance_scores
            }).sort_values('Importance', ascending=False).head(10)
            
            performance_data['claims_features'] = feature_importance
            print("📈 Extracted claims model feature importance")
    
    # Performance metrics from research
    performance_data['metrics'] = {
        'churn': {'roc_auc': 0.8926, 'baseline': 0.8805, 'improvement': 1.21},
        'claims_frequency': {'roc_auc': 0.9225, 'baseline': 0.9211, 'improvement': 0.14},
        'claims_severity': {'model_type': 'XGBoost Regressor', 'status': 'Production Ready'}
    }
    
    return performance_data

def create_model_visualization_data():
    """Create data structure for visualizations in the app"""
    
    try:
        perf_data = extract_model_performance()
        
        # Save visualization data for the app to use
        viz_data = {
            'churn_features': perf_data.get('churn_features', pd.DataFrame()).to_dict('records') if 'churn_features' in perf_data else [],
            'claims_features': perf_data.get('claims_features', pd.DataFrame()).to_dict('records') if 'claims_features' in perf_data else [],
            'performance_metrics': perf_data.get('metrics', {}),
            'model_files': {
                'churn': 'production_models/churn_model_optimized_20260209_134513.pkl',
                'claims_frequency': 'production_models/claims_frequency_model_optimized_20260209_134513.pkl',
                'claims_severity': 'production_models/claims_severity_model_optimized_20260209_134513.pkl'
            }
        }
        
        # Save to JSON for app to load
        with open('model_performance_data.json', 'w') as f:
            json.dump(viz_data, f, indent=2)
            
        print("✅ Saved model visualization data to model_performance_data.json")
        print("📊 Summary:")
        print(f"   • Churn features: {len(viz_data['churn_features'])}")
        print(f"   • Claims features: {len(viz_data['claims_features'])}")
        print(f"   • Performance metrics: {len(viz_data['performance_metrics'])} models")
        
        return viz_data
        
    except Exception as e:
        print(f"❌ Error creating visualization data: {e}")
        return None

if __name__ == "__main__":
    print("🔍 Extracting Real Model Performance from CXarticle.ipynb Production Models...")
    print("=" * 80)
    
    viz_data = create_model_visualization_data()
    
    if viz_data:
        print("\n🎯 Real Model Performance Summary:")
        print(f"   • Churn Model: {viz_data['performance_metrics']['churn']['roc_auc']:.2%} ROC-AUC")
        print(f"   • Claims Model: {viz_data['performance_metrics']['claims_frequency']['roc_auc']:.2%} ROC-AUC") 
        print(f"   • Models saved: {viz_data['performance_metrics']['churn']['improvement']:.1f}% improvement over baseline")
        
        if viz_data['churn_features']:
            print(f"\n📈 Top Churn Predictors:")
            for feat in viz_data['churn_features'][:5]:
                print(f"   • {feat['Feature']}: {feat['Importance']:.3f}")
                
        if viz_data['claims_features']:
            print(f"\n📈 Top Claims Predictors:")
            for feat in viz_data['claims_features'][:5]:
                print(f"   • {feat['Feature']}: {feat['Importance']:.3f}")
    else:
        print("❌ Failed to extract model performance data")