#!/usr/bin/env python3
"""
Load Real Model Predictions from Notebook to SQL Database
"""
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path

# Add project_structure to path
sys.path.insert(0, str(Path(__file__).parent / 'project_structure'))

from sql_predictions_manager import SQLModelPredictionsManager

print("=" * 70)
print("📊 Loading REAL predictions from Customer_Success_222331.ipynb")
print("=" * 70)

# Read the notebook
notebook_path = Path(__file__).parent / 'Customer_Success_222331.ipynb'

if not notebook_path.exists():
    print(f"❌ Notebook not found: {notebook_path}")
    sys.exit(1)

print(f"✅ Reading notebook: {notebook_path.name}")

with open(notebook_path, 'r') as f:
    nb = json.load(f)

# Find the cell that creates df_predictions
print("🔍 Looking for predictions data in notebook outputs...")

predictions_df = None

# Search through cells for stored predictions data
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        # Check if this cell has output with the predictions dataframe
        outputs = cell.get('outputs', [])
        for output in outputs:
            # Check if output contains data
            if output.get('output_type') == 'execute_result':
                data = output.get('data', {})
                if 'text/html' in data:
                    # This might be a dataframe display
                    continue
            
            # Check for stored variables
            if output.get('output_type') == 'display_data':
                data = output.get('data', {})
                # Try to find CSV text output or predictions
                if 'text/plain' in data:
                    text = data['text/plain']
                    if 'predictions' in text.lower() and 'shape' in text.lower():
                        print(f"   Found predictions reference in cell")

# Alternative: Check if CSV was generated
csv_path = Path(__file__).parent / 'model_outputs' / 'rag_model_predictions.csv'

if csv_path.exists():
    print(f"✅ Found predictions CSV: {csv_path}")
    predictions_df = pd.read_csv(csv_path)
    print(f"   Loaded {len(predictions_df)} rows")
else:
    print("⚠️  No predictions CSV found. Need to execute the notebook to generate predictions.")
    print("   Running notebook cell extraction...")
    
    # Execute the _temp_extract.py that was created
    temp_script = Path(__file__).parent / '_temp_extract.py'
    if temp_script.exists():
        print(f"   Executing: {temp_script.name}")
        import subprocess
        result = subprocess.run([sys.executable, str(temp_script)], 
                              capture_output=True, text=True, cwd=str(temp_script.parent))
        
        # Check if CSV was created
        if csv_path.exists():
            print(f"✅ Predictions generated!")
            predictions_df = pd.read_csv(csv_path)
            print(f"   Loaded {len(predictions_df)} rows")
        else:
            print("❌ Failed to generate predictions CSV")
            print("   You may need to run the notebook manually")
            sys.exit(1)
    else:
        print("❌ No temp script found. Cannot extract predictions.")
        sys.exit(1)

if predictions_df is None or len(predictions_df) == 0:
    print("❌ No predictions data available")
    sys.exit(1)

print("\n" + "=" * 70)
print("📊 Predictions Data Summary")
print("=" * 70)
print(f"Total records: {len(predictions_df)}")
print(f"Columns: {list(predictions_df.columns)}")
print(f"\nFirst few rows:")
print(predictions_df.head(3))

# Connect to database
print("\n" + "=" * 70)
print("🔌 Connecting to MySQL Database")
print("=" * 70)

manager = SQLModelPredictionsManager()

if not manager.connect():
    print("❌ Could not connect to MySQL")
    print("💡 Make sure XAMPP is running!")
    sys.exit(1)

print("✅ Connected to database")

# Clear existing data
print("\n🗑️  Clearing sample data...")
try:
    cursor = manager.connection.cursor()
    cursor.execute("TRUNCATE TABLE model_predictions;")
    manager.connection.commit()
    cursor.close()
    print("✅ Old data cleared")
except Exception as e:
    print(f"⚠️  Warning: {e}")

# Prepare data for insertion
print("\n💾 Preparing data for insertion...")

# Ensure all required columns exist
required_mapping = {
    'ID': 'policy_id',
    'Churn_Probability': 'Churn_Probability',
    'Claims_Probability': 'Claims_Probability', 
    'Claims_Severity': 'Claims_Severity',
    'Customer_Lifetime_Value': 'Customer_Lifetime_Value',
    'Customer_Segment': 'Customer_Segment',
    'Journey_Quadrant': 'Journey_Quadrant',
    'Pricing_Adequacy_Flag': 'Pricing_Adequacy_Flag',
    'Renewal_Risk_Score': 'Renewal_Risk_Score',
    'Is_High_Renewal_Risk': 'Is_High_Renewal_Risk'
}

# Create a clean dataframe with required columns
clean_df = pd.DataFrame()

for app_col, db_col in required_mapping.items():
    if app_col in predictions_df.columns:
        clean_df[db_col] = predictions_df[app_col]
    elif app_col == 'Is_High_Renewal_Risk' and 'High_Renewal_Risk' in predictions_df.columns:
        clean_df[db_col] = predictions_df['High_Renewal_Risk']
    elif app_col == 'Pricing_Adequacy_Flag' and 'Is_Underpriced' in predictions_df.columns:
        clean_df[db_col] = predictions_df['Is_Underpriced']
    elif app_col == 'Journey_Quadrant' and 'Journey_Segment' in predictions_df.columns:
        clean_df[db_col] = predictions_df['Journey_Segment']
    else:
        print(f"   ⚠️  Column {app_col} not found, using defaults")
        if 'Flag' in app_col or 'Risk' in app_col:
            clean_df[db_col] = 0
        elif 'Score' in app_col:
            clean_df[db_col] = 0.0
        elif 'Segment' in app_col or 'Quadrant' in app_col:
            clean_df[db_col] = 'MANAGE'
        else:
            clean_df[db_col] = 0.0

print(f"✅ Prepared {len(clean_df)} records")

# Insert data
print("\n📤 Inserting data into database...")
if manager.insert_predictions(clean_df):
    print(f"✅ Successfully inserted {len(clean_df)} REAL predictions!")
else:
    print("❌ Failed to insert predictions")
    manager.disconnect()
    sys.exit(1)

# Verify
print("\n" + "=" * 70)
print("✅ Verification")
print("=" * 70)

summary = manager.get_prediction_summary()
print(f"Total predictions: {summary.get('total_predictions', 0):,}")
print(f"Unique policies: {summary.get('unique_policies', 0):,}")
print(f"Avg churn prob: {summary.get('avg_churn_probability', 0):.2%}")
print(f"Avg claims prob: {summary.get('avg_claims_probability', 0):.2%}")
print(f"Total portfolio: €{summary.get('total_portfolio_value', 0)/1e6:.2f}M")
print(f"High risk count: {summary.get('high_risk_count', 0):,}")

manager.disconnect()

print("\n" + "=" * 70)
print("✅ SUCCESS! Real data loaded into database")
print("=" * 70)
print("🚀 Now run: streamlit run app.py")
print("=" * 70)
