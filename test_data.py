#!/usr/bin/env python3

import pandas as pd
from pathlib import Path
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def test_real_data_loading():
    """Test if we can load and process the real data correctly"""
    
    # Load real data
    real_data_path = Path('model_data/engineered_features_complete.csv')
    
    if not real_data_path.exists():
        print("❌ Real data file not found")
        return False
        
    df = pd.read_csv(real_data_path)
    print(f"✅ Raw data loaded: {len(df):,} records")
    
    # Sample for testing (like the app does)
    if len(df) > 8000:
        df = df.sample(n=8000, random_state=42).reset_index(drop=True)
    
    # Test key column mapping
    key_mappings = {
        'Churn_target': 'Churn_Prob',
        'Claims_binary': 'Claims_Prob', 
        'Claims_severity': 'Claims_Severity',
        'Distribution_channel': 'Channel',
        'Premium': 'Annual_Premium',
        'Driver_age': 'Age'
    }
    
    print(f"🎯 Testing column mappings:")
    for old_col, new_col in key_mappings.items():
        if old_col in df.columns:
            print(f"   ✅ {old_col} -> {new_col}: {df[old_col].iloc[0]}")
        else:
            print(f"   ❌ Missing: {old_col}")
    
    # Apply mappings
    df = df.rename(columns=key_mappings)
    
    # Calculate CLV like the app
    expected_claims_cost = df['Claims_Prob'] * df['Claims_Severity']
    retention_years = 1 / (df['Churn_Prob'] + 0.001)
    df['CLV'] = (df['Annual_Premium'] - expected_claims_cost) * retention_years
    df['CLV'] = np.clip(df['CLV'], 50, 3000)
    
    # Apply channel multiplier
    channel_multiplier = df['Channel'].map({'Agent': 1.8, 'Broker': 1.0}).fillna(1.2)
    df['CLV'] *= channel_multiplier
    
    # Test Value Tiers creation (the problematic part)
    print(f"🔍 Testing Value Tiers creation...")
    print(f"   CLV range: €{df['CLV'].min():.0f} to €{df['CLV'].max():.0f}")
    print(f"   CLV unique values: {df['CLV'].nunique()}")
    
    try:
        df['Value_Tier'] = pd.qcut(df['CLV'], q=4, labels=['Bronze', 'Silver', 'Gold', 'Platinum'], duplicates='drop')
        print(f"   ✅ pd.qcut worked successfully")
    except ValueError as e:
        print(f"   ⚠️  pd.qcut failed: {e}")
        print(f"   🔧 Using smart binning...")
        clv_percentiles = df['CLV'].quantile([0, 0.25, 0.5, 0.75, 1.0]).unique()
        if len(clv_percentiles) < 2:
            df['Value_Tier'] = 'Silver'
            print(f"   📊 All values identical, using single tier")
        elif len(clv_percentiles) == 2:
            df['Value_Tier'] = pd.cut(df['CLV'], bins=2, labels=['Bronze', 'Gold'], include_lowest=True)
            print(f"   📊 Using 2 tiers")
        elif len(clv_percentiles) == 3:
            df['Value_Tier'] = pd.cut(df['CLV'], bins=clv_percentiles, labels=['Bronze', 'Gold'], include_lowest=True)
            print(f"   📊 Using 2 tiers from 3 percentiles")
        else:
            bins_to_use = clv_percentiles[:5] if len(clv_percentiles) >= 5 else clv_percentiles
            n_labels = len(bins_to_use) - 1
            labels = ['Bronze', 'Silver', 'Gold', 'Platinum'][:n_labels]
            df['Value_Tier'] = pd.cut(df['CLV'], bins=bins_to_use, labels=labels, include_lowest=True)
            print(f"   📊 Using {n_labels} tiers")
        print(f"   ✅ Smart binning worked")
    
    # Test segment creation
    clv_median = df['CLV'].median()
    churn_median = df['Churn_Prob'].median()
    
    conditions = [
        (df['CLV'] > clv_median) & (df['Churn_Prob'] <= churn_median),
        (df['CLV'] <= clv_median) & (df['Churn_Prob'] <= churn_median),
        (df['CLV'] > clv_median) & (df['Churn_Prob'] > churn_median),
        (df['CLV'] <= clv_median) & (df['Churn_Prob'] > churn_median)
    ]
    choices = ['PROTECT', 'DEVELOP', 'MANAGE', 'EXIT']
    df['Segment'] = np.select(conditions, choices, default='MANAGE')
    
    # Show results
    print(f"\n📈 Results:")
    print(f"   Sample size: {len(df):,}")
    print(f"   Churn rate: {df['Churn_Prob'].mean():.1%}")
    print(f"   Avg CLV: €{df['CLV'].mean():.0f}")
    print(f"   Channels: {df['Channel'].value_counts().to_dict()}")
    print(f"   Segments: {df['Segment'].value_counts().to_dict()}")
    print(f"   Value Tiers: {df['Value_Tier'].value_counts().to_dict()}")
    
    # Check for real variation (not flat like dummy data)
    churn_std = df['Churn_Prob'].std()
    clv_std = df['CLV'].std()
    print(f"\n🔍 Data Variation Check:")
    print(f"   Churn std dev: {churn_std:.3f} {'✅ Good variation' if churn_std > 0.05 else '⚠️  Low variation'}")
    print(f"   CLV std dev: €{clv_std:.0f} {'✅ Good variation' if clv_std > 100 else '⚠️  Low variation'}")
    
    return True

if __name__ == "__main__":
    test_real_data_loading()