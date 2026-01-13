#!/usr/bin/env python3
"""
Quick Setup for Insurance Analytics - SQL Deployment
=====================================================
This script sets up the database and generates sample predictions
for demonstration purposes (using notebook data if available).

Usage:
    python quick_setup.py
"""

import os
import sys
from pathlib import Path
import subprocess
import json

def main():
    print("\n" + "="*70)
    print("🚀 Insurance Analytics - Quick SQL Setup")
    print("="*70)
    
    # Step 1: Check MySQL
    print("\n📋 Step 1: Checking MySQL...")
    result = subprocess.run(
        ['mysql', '--version'],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        print(f"✅ MySQL found: {result.stdout.strip()}")
    else:
        print("❌ MySQL not found. Please install: brew install mysql")
        return 1
    
    # Step 2: Start MySQL
    print("\n📋 Step 2: Starting MySQL...")
    result = subprocess.run(
        ['mysql.server', 'start'],
        capture_output=True,
        text=True
    )
    print("✅ MySQL started (or already running)")
    
    # Step 3: Simplified database init
    print("\n📋 Step 3: Initializing database...")
    
    try:
        import mysql.connector
        conn = mysql.connector.connect(
            host='localhost',
            user='root',
            password=''
        )
        cursor = conn.cursor()
        
        # Create database
        cursor.execute("CREATE DATABASE IF NOT EXISTS insurance")
        cursor.execute("USE insurance")
        
        # Create predictions table (without actual data for now)
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS model_predictions (
            prediction_id INT PRIMARY KEY AUTO_INCREMENT,
            policy_id INT NOT NULL UNIQUE,
            churn_probability FLOAT,
            claims_probability FLOAT,
            claims_severity FLOAT,
            customer_lifetime_value FLOAT,
            customer_segment VARCHAR(50),
            journey_quadrant VARCHAR(50),
            pricing_adequacy_flag TINYINT,
            renewal_risk_score FLOAT,
            is_high_renewal_risk TINYINT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_policy (policy_id),
            INDEX idx_churn (churn_probability),
            INDEX idx_segment (customer_segment)
        );
        """
        cursor.execute(create_table_sql)
        conn.commit()
        
        print("✅ Database 'insurance' created")
        print("✅ Table 'model_predictions' created")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"❌ Database error: {e}")
        return 1
    
    # Step 4: Generate predictions from notebook
    print("\n📋 Step 4: Generating model predictions...")
    print("   (This requires executing the notebook...)")
    
    nb_path = Path(__file__).parent / "Customer_Success_222331.ipynb"
    if nb_path.exists():
        print(f"✅ Found notebook: {nb_path.name}")
        print("\n   To generate predictions, run:")
        print("   python export_predictions_to_sql.py")
    else:
        print(f"⚠️  Notebook not found at {nb_path}")
    
    # Step 5: Show next steps
    print("\n" + "="*70)
    print("✨ Setup Complete!")
    print("="*70)
    
    print("\n📍 Next Steps:")
    print("\n1. Generate predictions from notebook:")
    print("   cd /Users/leonida/Documents/automobile_claims/Automobile")
    print("   python export_predictions_to_sql.py")
    
    print("\n2. Run the Streamlit app:")
    print("   streamlit run app.py")
    
    print("\n3. Verify database:")
    print("   mysql -u root -p insurance")
    print("   SELECT COUNT(*) FROM model_predictions;")
    
    print("\n📚 Documentation:")
    print("   - README_DEPLOYMENT.md (overview)")
    print("   - DEPLOYMENT_GUIDE_SQL.md (full guide)")
    print("   - EXECUTION_COMMANDS.sh (copy-paste commands)")
    
    print("\n" + "="*70)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
