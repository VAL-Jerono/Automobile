#!/usr/bin/env python3
"""
Export Predictions from Notebook to SQL Database
================================================
Extracts the RAG export cell from Customer_Success_222331.ipynb,
executes it in a controlled environment, and stores results in MySQL.

This is the deployment-ready approach that doesn't require CSV files.

Usage:
    python export_predictions_to_sql.py
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path
import logging
import os

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def extract_notebook_cells():
    """Extract all code cells from notebook."""
    # Look for notebook in Automobile folder
    nb_path = Path(__file__).parent.parent.parent / 'Customer_Success_222331.ipynb'
    
    logger.info(f"📖 Reading notebook: {nb_path}")
    
    with open(nb_path, 'r') as f:
        nb = json.load(f)
    
    code_cells = []
    export_cell_idx = None
    
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            code_cells.append(source)
            
            if 'EXPORTING MODEL PREDICTIONS' in source:
                export_cell_idx = len(code_cells) - 1
                logger.info(f"✅ Found export cell at index {export_cell_idx}")
    
    return code_cells, export_cell_idx

def create_execution_script(code_cells, export_cell_idx):
    """Create a Python script that runs cells up to and including export."""
    
    # Build the script
    script_lines = [
        "#!/usr/bin/env python3",
        '"""Auto-generated from notebook cells"""',
        "import sys",
        "import os",
        "from pathlib import Path",
        "import warnings",
        "warnings.filterwarnings('ignore')",
        "",
        "# Setup paths",
        "notebook_root = Path(__file__).parent.parent.parent",
        "sys.path.insert(0, str(notebook_root))",
        "",
        "# Imports",
        "import pandas as pd",
        "import numpy as np",
        "import logging",
        "logging.basicConfig(level=logging.WARNING)",
        "",
        "print('🔄 Executing notebook cells...')",
        "print('=' * 70)",
    ]
    
    # Add cells up to and including export
    successful_cells = 0
    for i in range(export_cell_idx + 1):
        cell_code = code_cells[i]
        
        # Skip empty or comment-only cells
        if not cell_code.strip() or cell_code.strip().startswith('#'):
            continue
        
        script_lines.append("")
        script_lines.append("# " + "=" * 64)
        script_lines.append(f"# CELL {i+1}")
        script_lines.append("# " + "=" * 64)
        script_lines.append("try:")
        
        # Indent the cell code
        for line in cell_code.split('\n'):
            script_lines.append(f"    {line}")
        
        script_lines.append(f"    print(f'✅ Cell {i+1} executed')")
        script_lines.append("except Exception as e:")
        script_lines.append(f"    print(f'⚠️  Cell {i+1} error: {{type(e).__name__}}: {{e}}')")
        script_lines.append(f"    if {i} >= {export_cell_idx}:")
        script_lines.append(f"        raise")
        successful_cells += 1
    
    # Add final export to SQL
    script_lines.extend([
        "",
        "# " + "=" * 64,
        "# EXPORT TO SQL DATABASE",
        "# " + "=" * 64,
        "print('\\n📊 Exporting predictions to SQL database...')",
        "try:",
        "    from utils.sql_predictions_manager import SQLModelPredictionsManager",
        "    ",
        "    manager = SQLModelPredictionsManager()",
        "    if not manager.connect():",
        "        print('⚠️  Could not connect to MySQL - saving to CSV instead')",
        "        os.makedirs('model_outputs', exist_ok=True)",
        "        if 'rag_export' in locals():",
        "            rag_export.to_csv('model_outputs/rag_model_predictions.csv', index=False)",
        "            print('✅ Saved to model_outputs/rag_model_predictions.csv')",
        "    else:",
        "        manager.create_predictions_table()",
        "        if 'rag_export' in locals():",
        "            manager.insert_predictions(rag_export)",
        "            summary = manager.get_prediction_summary()",
        "            print(f'✅ SQL Database Summary:')",
        "            print(f'   Total predictions: {summary.get(\"total_predictions\", 0):,}')",
        "            print(f'   Portfolio value: €{summary.get(\"total_portfolio_value\", 0):,.0f}')",
        "            print(f'   High risk: {summary.get(\"high_risk_count\", 0):,}')",
        "        manager.disconnect()",
        "except ImportError as e:",
        "    print(f'⚠️  Import error: {e} - saving to CSV')",
        "    os.makedirs('model_outputs', exist_ok=True)",
        "    if 'rag_export' in locals():",
        "        rag_export.to_csv('model_outputs/rag_model_predictions.csv', index=False)",
        "except Exception as e:",
        "    print(f'⚠️  Export error: {e}')",
        "    import traceback",
        "    traceback.print_exc()",
        "",
        "print('✨ Prediction processing complete!')",
    ])
    
    return '\n'.join(script_lines)

def run_extraction():
    """Execute the extraction script."""
    
    logger.info("📋 Step 1: Extracting notebook cells...")
    code_cells, export_cell_idx = extract_notebook_cells()
    
    if export_cell_idx is None:
        logger.error("❌ Export cell not found in notebook")
        return False
    
    logger.info(f"✅ Found {len(code_cells)} total code cells")
    logger.info(f"✅ Export cell at index {export_cell_idx}")
    
    logger.info("\n📝 Step 2: Creating execution script...")
    script_content = create_execution_script(code_cells, export_cell_idx)
    
    # Write to temporary file
    script_path = Path(__file__).parent / '_temp_extract.py'
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    logger.info(f"✅ Script created: {script_path}")
    
    logger.info("\n⚙️  Step 3: Executing script (this may take 5-10 minutes)...")
    logger.info("=" * 70)
    
    try:
        # Run the script and stream output
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=Path(__file__).parent,
            capture_output=False,  # Show output in real-time
            text=True,
            timeout=900  # 15 minute timeout
        )
        
        logger.info("=" * 70)
        
        if result.returncode == 0:
            logger.info("\n✅ Extraction completed successfully!")
            return True
        else:
            logger.error(f"\n❌ Script failed with exit code {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("\n❌ Execution timed out after 15 minutes")
        return False
    except Exception as e:
        logger.error(f"\n❌ Execution error: {e}")
        return False
    finally:
        # Clean up
        script_path.unlink(missing_ok=True)

def main():
    """Main entry point."""
    
    logger.info("=" * 70)
    logger.info("🚀 PREDICTION EXPORT PIPELINE (Notebook → SQL)")
    logger.info("=" * 70)
    
    success = run_extraction()
    
    if success:
        logger.info("\n" + "=" * 70)
        logger.info("✨ SUCCESS!")
        logger.info("=" * 70)
        logger.info("\n📍 Next steps:")
        logger.info("   1. Check MySQL for data: SELECT COUNT(*) FROM model_predictions;")
        logger.info("   2. Run the Streamlit app: streamlit run app.py")
        logger.info("   3. Commit changes to GitHub (without CSV files)")
        return 0
    else:
        logger.error("\n" + "=" * 70)
        logger.error("❌ FAILED")
        logger.error("=" * 70)
        logger.error("\n🔧 Troubleshooting:")
        logger.error("   1. Ensure MySQL is running")
        logger.error("   2. Check notebook has no errors")
        logger.error("   3. Review output above for specific cell errors")
        return 1

if __name__ == '__main__':
    sys.exit(main())
