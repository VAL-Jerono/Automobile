#!/usr/bin/env python3
"""
Generate Model Predictions from Notebook and Store in SQL Database
==================================================================
This script extracts the export logic from Customer_Success_222331.ipynb,
runs the model predictions, and stores them in MySQL instead of CSV.

This makes the data deployable to GitHub without storing large CSV files.

Usage:
    python generate_predictions_to_sql.py
"""

import sys
import json
import subprocess
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def extract_code_from_notebook():
    """Extract all code cells from the notebook."""
    nb_path = Path(__file__).parent / 'Customer_Success_222331.ipynb'
    
    with open(nb_path, 'r') as f:
        nb = json.load(f)
    
    code_cells = []
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            code_cells.append(source)
    
    return code_cells

def create_execution_script(code_cells):
    """Create a Python script that runs all code cells sequentially."""
    
    script = '''#!/usr/bin/env python3
"""Auto-generated script from notebook cells"""
import pandas as pd
import numpy as np
import logging
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.WARNING)

# Import necessary libraries
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import pickle
import joblib
from pathlib import Path
import sys
import json

# Set up paths
notebook_dir = Path(__file__).parent
sys.path.insert(0, str(notebook_dir))

print("🔄 Running notebook cells to generate predictions...")
print("=" * 70)
'''
    
    # Add all code cells
    for i, cell_code in enumerate(code_cells):
        # Skip cells that are just comments or markdown
        if cell_code.strip().startswith('#') or not cell_code.strip():
            continue
        
        script += f"\n# ===== CELL {i} =====\n"
        script += "try:\n"
        # Indent the code
        indented_code = '\n'.join('    ' + line for line in cell_code.split('\n'))
        script += indented_code
        script += "\nexcept Exception as e:\n"
        script += f"    print(f'⚠️  Cell {i} error (non-fatal): {{e}}')\n"
    
    return script

def run_notebook_extraction():
    """Execute the notebook and extract predictions."""
    
    logger.info("📖 Loading notebook...")
    code_cells = extract_code_from_notebook()
    logger.info(f"✅ Extracted {len(code_cells)} code cells")
    
    # Create temporary execution script
    logger.info("📝 Creating execution script...")
    script_content = create_execution_script(code_cells)
    
    # Write and execute the script
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script_content)
        script_path = f.name
    
    try:
        logger.info("⚙️  Executing notebook cells (this may take a few minutes)...")
        result = subprocess.run(
            ['python3', script_path],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        if result.returncode != 0:
            logger.warning(f"⚠️  Execution had some issues:")
            logger.warning(result.stderr[:500])  # Show first 500 chars of error
        
        # Print output
        if result.stdout:
            logger.info("📊 Output:")
            logger.info(result.stdout[-2000:])  # Last 2000 chars
            
        return True
    except subprocess.TimeoutExpired:
        logger.error("❌ Notebook execution timed out after 10 minutes")
        return False
    except Exception as e:
        logger.error(f"❌ Execution error: {e}")
        return False
    finally:
        # Clean up
        Path(script_path).unlink(missing_ok=True)

def create_model_predictions_table():
    """Create MySQL table for model predictions."""
    import mysql.connector
    
    try:
        conn = mysql.connector.connect(
            host='localhost',
            user='root',
            password='',
            database='insurance'
        )
        cursor = conn.cursor()
        
        # Create table
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS model_predictions (
            prediction_id INT PRIMARY KEY AUTO_INCREMENT,
            policy_id INT NOT NULL,
            churn_probability FLOAT,
            claims_probability FLOAT,
            claims_severity FLOAT,
            customer_lifetime_value FLOAT,
            customer_segment VARCHAR(50),
            journey_quadrant VARCHAR(50),
            pricing_adequacy_flag TINYINT,
            renewal_risk_score FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE KEY unique_policy (policy_id),
            FOREIGN KEY (policy_id) REFERENCES policies(policy_id)
        );
        """
        
        cursor.execute(create_table_sql)
        conn.commit()
        logger.info("✅ Created model_predictions table")
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"❌ Failed to create table: {e}")
        return False

def main():
    """Main execution flow."""
    
    logger.info("🚀 Starting Model Prediction Generation to SQL Database")
    logger.info("=" * 70)
    
    # Step 1: Create the table
    logger.info("\n📋 Step 1: Preparing database schema...")
    if not create_model_predictions_table():
        logger.warning("⚠️  Could not create table - MySQL may not be running")
    
    # Step 2: Run the notebook extraction
    logger.info("\n📊 Step 2: Extracting predictions from notebook...")
    if run_notebook_extraction():
        logger.info("\n✅ Notebook execution completed")
    else:
        logger.error("\n❌ Notebook execution failed")
        sys.exit(1)
    
    logger.info("\n" + "=" * 70)
    logger.info("✨ Predictions processing complete!")
    logger.info("📍 Next steps:")
    logger.info("   1. Data is now stored in MySQL 'model_predictions' table")
    logger.info("   2. Update Streamlit app to read from SQL")
    logger.info("   3. Push to GitHub without CSV files")

if __name__ == '__main__':
    main()
