#!/usr/bin/env python3
"""
Data Access Fix Script for Insurance Analytics App
Helps resolve the "No data available" error by checking and generating required data files.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os

def check_data_files():
    """Check which data files are available"""
    project_path = Path(__file__).parent
    
    print("🔍 Checking Data File Availability")
    print("=" * 50)
    
    # Core data files
    files_to_check = {
        "Primary Data": [
            "Motor_vehicle_insurance_data.csv",
            "sample_type_claim.csv"
        ],
        "Model Data": [
            "model_data/engineered_features_complete.csv",
            "model_data/churn_model_dataset.csv", 
            "model_data/claims_frequency_model_dataset.csv",
            "model_data/claims_severity_model_dataset.csv",
            "model_data/clv_model_dataset.csv"
        ],
        "App Data": [
            "model_outputs/rag_model_predictions.csv",
            "rag_model_predictions.csv",
            "phase5_1_results/rag_model_predictions.csv"
        ]
    }
    
    available_files = []
    missing_files = []
    
    for category, files in files_to_check.items():
        print(f"\n📂 {category}:")
        for file_path in files:
            full_path = project_path / file_path
            if full_path.exists():
                size_mb = full_path.stat().st_size / (1024 * 1024)
                print(f"  ✅ {file_path} ({size_mb:.1f} MB)")
                available_files.append(full_path)
            else:
                print(f"  ❌ {file_path} (Missing)")
                missing_files.append(file_path)
    
    # Check for iCloud files
    print(f"\n☁️ iCloud Files:")
    icloud_files = list(project_path.glob("**/.*.icloud"))
    for icloud_file in icloud_files:
        print(f"  ☁️ {icloud_file.name} (In iCloud - not downloaded)")
    
    return available_files, missing_files, icloud_files

def generate_predictions_data():
    """Generate rag_model_predictions.csv from available data"""
    project_path = Path(__file__).parent
    
    print("\n🔧 Generating Predictions Data")
    print("=" * 50)
    
    # Try to use engineered features first
    engineered_path = project_path / "model_data" / "engineered_features_complete.csv"
    motor_path = project_path / "Motor_vehicle_insurance_data.csv"
    
    if engineered_path.exists():
        print(f"📖 Loading: {engineered_path}")
        df = pd.read_csv(engineered_path)
        source = "engineered_features"
    elif motor_path.exists():
        print(f"📖 Loading: {motor_path}")
        df = pd.read_csv(motor_path, sep=';')
        source = "motor_vehicle_data"
        # Take a sample for performance
        df = df.sample(n=min(15000, len(df)), random_state=42).copy()
    else:
        print("❌ No source data available!")
        return False
    
    print(f"✅ Loaded {len(df):,} records")
    
    # Generate synthetic predictions based on research findings
    np.random.seed(42)  # For reproducibility
    
    # Ensure policy_id exists
    if 'policy_id' not in df.columns:
        if 'ID' in df.columns:
            df['policy_id'] = df['ID']
        else:
            df['policy_id'] = range(1, len(df) + 1)
    
    print("🎯 Generating predictions...")
    
    # Churn probability (research finding: 20.4% average)
    df['churn_probability'] = np.random.beta(2, 8, len(df))  # Realistic distribution
    
    # Claims probability (research finding: 18.6% frequency)
    df['claims_probability'] = np.random.beta(1.5, 8, len(df))
    
    # Claims severity (research finding: €825 average)
    df['claims_severity'] = np.random.lognormal(np.log(825), 0.8, len(df))
    
    # Customer Lifetime Value
    if 'Premium' in df.columns:
        # Use actual premium data to generate realistic CLV
        base_multiplier = np.random.normal(3.2, 1.1, len(df))  # ~€1,247 average CLV
        df['customer_lifetime_value'] = df['Premium'] * base_multiplier
    else:
        # Generate CLV with realistic distribution (research finding: €1,247 mean)
        df['customer_lifetime_value'] = np.random.lognormal(np.log(1247), 0.6, len(df))
    
    df['customer_lifetime_value'] = np.maximum(df['customer_lifetime_value'], 0)
    
    print("📊 Creating strategic segments...")
    
    # Strategic segmentation (research framework)
    clv_threshold = df['customer_lifetime_value'].median()
    risk_threshold = df['churn_probability'].median()
    
    conditions = [
        (df['customer_lifetime_value'] >= clv_threshold) & (df['churn_probability'] < risk_threshold),  # High value, low risk
        (df['customer_lifetime_value'] >= clv_threshold) & (df['churn_probability'] >= risk_threshold),  # High value, high risk  
        (df['customer_lifetime_value'] < clv_threshold) & (df['churn_probability'] < risk_threshold),   # Low value, low risk
        (df['customer_lifetime_value'] < clv_threshold) & (df['churn_probability'] >= risk_threshold)    # Low value, high risk
    ]
    
    choices = ['Protect', 'Rescue', 'Develop', 'Monitor']
    df['journey_quadrant'] = np.select(conditions, choices, default='Unknown')
    
    # Customer segments (simplified)
    df['customer_segment'] = np.random.choice(
        ['Premium', 'Standard', 'Budget', 'New'], 
        len(df), 
        p=[0.15, 0.5, 0.25, 0.1]
    )
    
    # Pricing adequacy (research finding: 14.8% underpriced)
    df['pricing_adequacy_flag'] = (np.random.random(len(df)) < 0.148).astype(int)
    
    # Renewal risk score
    df['renewal_risk_score'] = np.clip(
        df['churn_probability'] * 0.8 + np.random.normal(0, 0.05, len(df)), 
        0, 1
    )
    df['is_high_renewal_risk'] = (df['renewal_risk_score'] > 0.6).astype(int)
    
    # Save the predictions
    output_path = project_path / "model_outputs" / "rag_model_predictions.csv"
    output_path.parent.mkdir(exist_ok=True)
    
    df.to_csv(output_path, index=False)
    
    print(f"✅ Generated: {output_path}")
    print(f"📊 Shape: {df.shape}")
    print(f"🎯 Segments: {df['journey_quadrant'].value_counts().to_dict()}")
    print(f"💰 CLV Range: €{df['customer_lifetime_value'].min():.0f} - €{df['customer_lifetime_value'].max():.0f}")
    print(f"⚠️ Avg Churn Risk: {df['churn_probability'].mean():.1%}")
    
    return True

def download_icloud_instructions():
    """Provide instructions for downloading iCloud files"""
    print("\n☁️ iCloud File Download Instructions")
    print("=" * 50)
    print("If you have .rag_model_predictions.csv.icloud file:")
    print("1. Double-click the .icloud file to download it")
    print("2. Wait for download to complete")
    print("3. The file will appear as rag_model_predictions.csv")
    print("4. Move it to the model_outputs/ directory")
    print("\nAlternatively, run this script to generate synthetic data!")

def main():
    """Main function to check and fix data access"""
    print("🚗 Insurance Analytics - Data Access Fixer")
    print("=" * 50)
    
    # Check current data availability
    available, missing, icloud = check_data_files()
    
    # Check if app data is available
    has_predictions = any("rag_model_predictions.csv" in str(path) for path in available)
    
    if has_predictions:
        print("\n✅ App data is available! The app should work.")
        return
    
    print(f"\n❌ App predictions data is missing!")
    
    if icloud:
        print(f"☁️ Found {len(icloud)} iCloud files. Consider downloading them.")
        download_icloud_instructions()
    
    # Offer to generate synthetic data
    print("\n🔧 Would you like to generate synthetic predictions data?")
    print("This will create rag_model_predictions.csv for the app to use.")
    
    choice = input("Generate data? (y/n): ").lower().strip()
    
    if choice == 'y':
        success = generate_predictions_data()
        if success:
            print("\n🎉 Data generated successfully!")
            print("You can now run the Streamlit app:")
            print("streamlit run app.py")
        else:
            print("\n❌ Failed to generate data. Check the error messages above.")
    else:
        print("\n💡 To fix manually:")
        print("1. Download the iCloud files, or")
        print("2. Run this script again with 'y' to generate data")

if __name__ == "__main__":
    main()