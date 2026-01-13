#!/usr/bin/env python3
"""
Export Real Predictions to Database
====================================
Loads the actual prediction data from the notebook and exports to MySQL database.

This script:
1. Reads the Customer_Success_222331.ipynb notebook
2. Extracts the rag_export DataFrame with all predictions
3. Inserts into the model_predictions table
4. Shows summary statistics

Usage:
    python export_real_predictions.py
"""

import json
import pandas as pd
import sys
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def extract_predictions_from_notebook():
    """Extract predictions from notebook by running all cells."""
    nb_path = Path(__file__).parent / 'Customer_Success_222331.ipynb'
    
    logger.info(f"📖 Reading notebook: {nb_path}")
    
    with open(nb_path, 'r') as f:
        nb = json.load(f)
    
    logger.info("🔄 Executing notebook cells to generate predictions...")
    
    # Create execution environment
    import warnings
    warnings.filterwarnings('ignore')
    
    exec_globals = {
        'warnings': warnings,
        '__name__': '__main__',
        '__file__': str(nb_path.parent / 'notebook.py')
    }
    
    try:
        # Execute all code cells in order
        cell_count = 0
        for cell in nb['cells']:
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                
                # Skip empty cells
                if not source.strip():
                    continue
                
                try:
                    exec(source, exec_globals)
                    cell_count += 1
                except Exception as e:
                    # Some cells might fail, but we continue
                    logger.warning(f"Cell execution warning: {type(e).__name__}")
                    continue
        
        logger.info(f"✅ Executed {cell_count} notebook cells")
        
        if 'rag_export' in exec_globals:
            rag_export = exec_globals['rag_export']
            logger.info(f"✅ Generated rag_export with {len(rag_export):,} rows")
            return rag_export
        else:
            logger.error("rag_export not found in execution globals")
            return None
    
    except Exception as e:
        logger.error(f"Error executing notebook: {e}")
        import traceback
        traceback.print_exc()
        return None


def export_to_database(df: pd.DataFrame) -> bool:
    """Export predictions to database."""
    if df is None or df.empty:
        logger.error("No data to export")
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
            return False
        
        # Insert predictions
        rows_inserted = manager.insert_predictions(df)
        
        if rows_inserted > 0:
            logger.info(f"✅ Inserted {rows_inserted:,} predictions")
            
            # Get summary
            summary = manager.get_prediction_summary()
            logger.info("📊 Database Summary:")
            logger.info(f"   Total predictions: {summary.get('total_predictions', 0):,}")
            logger.info(f"   Portfolio value: €{summary.get('total_portfolio_value', 0):,.0f}")
            logger.info(f"   High risk: {summary.get('high_risk_count', 0):,}")
            logger.info(f"   Avg churn: {summary.get('avg_churn_probability', 0):.1%}")
            
            manager.disconnect()
            return True
        else:
            logger.error("No rows inserted")
            manager.disconnect()
            return False
    
    except Exception as e:
        logger.error(f"Database export error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main execution."""
    logger.info("=" * 70)
    logger.info("EXPORT REAL PREDICTIONS TO DATABASE")
    logger.info("=" * 70)
    
    # Extract from notebook
    rag_export = extract_predictions_from_notebook()
    
    if rag_export is None:
        logger.error("❌ Failed to extract predictions")
        return 1
    
    # Show preview
    logger.info(f"\n📊 Preview of predictions:")
    logger.info(f"   Columns: {', '.join(rag_export.columns.tolist())}")
    logger.info(f"   Rows: {len(rag_export):,}")
    logger.info(f"   Memory: {rag_export.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    
    # Export to database
    if export_to_database(rag_export):
        logger.info("\n✨ Export complete!")
        return 0
    else:
        logger.error("\n❌ Export failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
