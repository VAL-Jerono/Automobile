#!/usr/bin/env python3
"""
Quick Script to Populate Database with Sample Predictions
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'project_structure'))

from sql_predictions_manager import SQLModelPredictionsManager
import pandas as pd
import numpy as np

# Create sample data
print("📊 Creating sample prediction data...")
np.random.seed(42)

num_policies = 1000

data = {
    'policy_id': range(1, num_policies + 1),
    'Churn_Probability': np.random.beta(2, 5, num_policies),
    'Claims_Probability': np.random.beta(2, 8, num_policies),
    'Claims_Severity': np.random.gamma(2, 500, num_policies),
    'Customer_Lifetime_Value': np.random.gamma(2, 300, num_policies),
    'Customer_Segment': np.random.choice(['PROTECT', 'DEVELOP', 'MANAGE', 'EXIT'], num_policies, p=[0.15, 0.35, 0.35, 0.15]),
    'Journey_Quadrant': np.random.choice(['PROTECT', 'DEVELOP', 'MANAGE', 'EXIT'], num_policies, p=[0.15, 0.35, 0.35, 0.15]),
    'Pricing_Adequacy_Flag': np.random.choice([0, 1], num_policies, p=[0.86, 0.14]),
    'Renewal_Risk_Score': np.random.uniform(0, 100, num_policies),
    'Is_High_Renewal_Risk': np.random.choice([0, 1], num_policies, p=[0.75, 0.25]),
}

# Add churn risk levels
df = pd.DataFrame(data)
df['Churn_Risk_Level'] = pd.cut(
    df['Churn_Probability'],
    bins=[0, 0.3, 0.6, 0.85, 1.0],
    labels=['Low', 'Medium', 'High', 'Critical']
).astype(str)

print(f"✅ Created {len(df)} sample policies")

# Connect to database
print("🔌 Connecting to database...")
manager = SQLModelPredictionsManager()

if not manager.connect():
    print("❌ Could not connect to MySQL")
    print("💡 Make sure XAMPP is running!")
    sys.exit(1)

# Create table
print("📋 Creating table structure...")
manager.create_predictions_table()

# Insert data
print("💾 Inserting predictions...")
if manager.insert_predictions(df):
    print(f"✅ Successfully inserted {len(df)} predictions!")
else:
    print("❌ Failed to insert predictions")

# Verify
summary = manager.get_prediction_summary()
print("\n📊 Database Summary:")
print(f"   Total predictions: {summary.get('total_predictions', 0)}")
print(f"   Unique policies: {summary.get('unique_policies', 0)}")
print(f"   Avg churn prob: {summary.get('avg_churn_probability', 0):.2%}")
print(f"   Total portfolio: €{summary.get('total_portfolio_value', 0)/1e6:.2f}M")

manager.disconnect()
print("\n✅ Done! You can now run: streamlit run app.py")
