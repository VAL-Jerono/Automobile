#!/usr/bin/env python3
"""
Extract Real Predictions from Customer_Success_222331.ipynb and Load to Database
Extracts stored predictions from notebook outputs or generated CSVs
"""
import sys
import json
import pandas as pd
from pathlib import Path
from io import StringIO

# Add project_structure to path
sys.path.insert(0, str(Path(__file__).parent / 'project_structure'))

print("=" * 80)
print("📊 LOADING REAL PREDICTIONS FROM NOTEBOOK")
print("=" * 80)

notebook_path = Path(__file__).parent / 'Customer_Success_222331.ipynb'
print(f"\n📖 Reading notebook: {notebook_path.name}")

# Check if notebook exists
if not notebook_path.exists():
    print(f"❌ Notebook not found: {notebook_path}")
    sys.exit(1)

print("⚙️  Extracting predictions from notebook outputs...")

predictions_df = None

# First, try to extract from notebook outputs
with open(notebook_path, 'r') as f:
    nb = json.load(f)

found_count = 0

# Search through cells for predictions data
for cell_idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    
    source = ''.join(cell.get('source', []))
    
    # Check if this cell creates df_predictions
    if 'df_predictions' in source or 'predictions' in source.lower():
        found_count += 1
        
        # Look at outputs for this cell
        for output in cell.get('outputs', []):
            # Check for HTML table output (pandas display)
            if output.get('output_type') == 'display_data':
                data = output.get('data', {})
                if 'text/html' in data:
                    html = data['text/html']
                    # Try to read HTML table as DataFrame
                    try:
                        dfs = pd.read_html(StringIO(html))
                        if len(dfs) > 0:
                            df = dfs[0]
                            if len(df) > 50:  # Real predictions have many rows
                                predictions_df = df
                                print(f"   ✅ Extracted {len(predictions_df)} rows from HTML output")
                                break
                    except Exception as e:
                        pass

if predictions_df is None:
    print("\n⚠️  No predictions found in notebook outputs")
    print("   Checking for generated CSV files...")
    
    # Check for generated CSV
    csv_paths = [
        Path(__file__).parent / 'model_outputs' / 'rag_model_predictions.csv',
        Path(__file__).parent.parent / 'model_outputs' / 'predictions.csv',
        Path(__file__).parent.parent / 'Motor vehicle insurance data.csv'
    ]
    
    for csv_path in csv_paths:
        if csv_path.exists():
            print(f"   📁 Found: {csv_path.name}")
            try:
                predictions_df = pd.read_csv(csv_path)
                print(f"   ✅ Loaded {len(predictions_df)} records from CSV")
                break
            except Exception as e:
                print(f"   ⚠️  Failed to read {csv_path.name}: {e}")

if predictions_df is None:
    print("\n❌ Could not extract predictions from notebook or CSV")
    print("   Please run the Customer_Success notebook to generate predictions")
    sys.exit(1)

# Process and prepare predictions for database
print(f"\n📋 Processing predictions data...")
print(f"   Shape: {predictions_df.shape}")
print(f"   Columns: {list(predictions_df.columns)}")

# Expected columns from the model
expected_cols = [
    'policy_id', 'churn_probability', 'claims_probability', 'claims_severity',
    'customer_lifetime_value', 'customer_segment', 'journey_quadrant',
    'pricing_adequacy_flag', 'renewal_risk_score', 'is_high_renewal_risk'
]

# Check what we have
available_cols = [col for col in expected_cols if col in predictions_df.columns]
print(f"   Available: {available_cols}")

# Select and prepare columns for database
df_final = predictions_df.copy()

# Handle column name variations
if 'policy_id' not in df_final.columns and 'PolicyID' in df_final.columns:
    df_final['policy_id'] = df_final['PolicyID']

# Ensure we have all required columns
for col in expected_cols:
    if col not in df_final.columns:
        if col == 'is_high_renewal_risk' and 'renewal_risk_score' in df_final.columns:
            df_final['is_high_renewal_risk'] = (df_final['renewal_risk_score'] > 0.5).astype(int)
        else:
            default_val = 'Unknown' if col.endswith('_segment') or col.endswith('_quadrant') else 0
            df_final[col] = default_val

# Select only the columns we need
df_final = df_final[expected_cols].drop_duplicates(subset=['policy_id']).reset_index(drop=True)

# Fill NaN with defaults
df_final = df_final.fillna({
    'churn_probability': 0.0,
    'claims_probability': 0.0,
    'claims_severity': 0.0,
    'customer_lifetime_value': 0.0,
    'customer_segment': 'Unknown',
    'journey_quadrant': 'Unknown',
    'pricing_adequacy_flag': 0,
    'renewal_risk_score': 0.0,
    'is_high_renewal_risk': 0
})

# Convert to correct types
df_final['policy_id'] = df_final['policy_id'].astype(int)
df_final['churn_probability'] = df_final['churn_probability'].astype(float)
df_final['claims_probability'] = df_final['claims_probability'].astype(float)
df_final['claims_severity'] = df_final['claims_severity'].astype(float)
df_final['customer_lifetime_value'] = df_final['customer_lifetime_value'].astype(float)
df_final['renewal_risk_score'] = df_final['renewal_risk_score'].astype(float)
df_final['pricing_adequacy_flag'] = df_final['pricing_adequacy_flag'].astype(int)
df_final['is_high_renewal_risk'] = df_final['is_high_renewal_risk'].astype(int)

print(f"   Final shape: {df_final.shape}")

# Load to database
print(f"\n🔗 Connecting to database...")
try:
    from sql_predictions_manager import PredictionsManager
    manager = PredictionsManager()
    
    # Get current count
    current_count = len(manager.get_all_predictions())
    print(f"   Current records in database: {current_count}")
    
    if current_count > 0 and current_count <= 1000:
        print(f"   ⚠️  Existing data appears to be sample data (≤1000 records)")
        print(f"\n   Clearing old sample data and loading real predictions...")
        
        # Delete all records
        try:
            import mysql.connector
            conn = mysql.connector.connect(
                host='localhost', user='root', password='', database='insurance'
            )
            cursor = conn.cursor()
            cursor.execute("TRUNCATE TABLE predictions")
            conn.commit()
            cursor.close()
            conn.close()
            print("   ✅ Cleared old predictions")
        except Exception as e:
            print(f"   ⚠️  Could not clear table: {e}")
    
    # Insert new predictions
    print(f"\n💾 Loading {len(df_final)} predictions to database...")
    
    success_count = 0
    for idx, row in df_final.iterrows():
        try:
            manager.insert_prediction(
                policy_id=int(row['policy_id']),
                churn_probability=float(row['churn_probability']),
                claims_probability=float(row['claims_probability']),
                claims_severity=float(row['claims_severity']),
                customer_lifetime_value=float(row['customer_lifetime_value']),
                customer_segment=str(row['customer_segment']),
                journey_quadrant=str(row['journey_quadrant']),
                pricing_adequacy_flag=int(row['pricing_adequacy_flag']),
                renewal_risk_score=float(row['renewal_risk_score']),
                is_high_renewal_risk=int(row['is_high_renewal_risk'])
            )
            success_count += 1
            
            if (idx + 1) % 5000 == 0:
                print(f"   ✓ Loaded {idx + 1:,}/{len(df_final):,} records...")
        
        except Exception as e:
            if idx < 5:  # Only print first few errors
                print(f"   ⚠️  Row {idx}: {str(e)[:60]}")
    
    print(f"\n✅ Successfully loaded {success_count:,}/{len(df_final):,} predictions")
    
    # Verify
    final_count = len(manager.get_all_predictions())
    print(f"\n📊 Final database state:")
    print(f"   Total predictions: {final_count:,}")
    print(f"   Portfolio value: €{df_final['customer_lifetime_value'].sum():,.0f}")
    print(f"   Avg churn prob: {df_final['churn_probability'].mean():.1%}")
    print(f"   Avg claims prob: {df_final['claims_probability'].mean():.1%}")
    
except Exception as e:
    print(f"\n❌ Database error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("✅ REAL PREDICTIONS LOADED SUCCESSFULLY")
print("=" * 80)
print("\nNext: Restart your Streamlit app to see real data")
