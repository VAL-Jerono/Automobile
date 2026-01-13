#!/usr/bin/env python3
"""
Direct Notebook Execution to Generate SQL Predictions
=====================================================
Runs the Customer_Success_222331.ipynb notebook using papermill
and stores the model predictions in MySQL.
"""

import subprocess
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def run_with_papermill():
    """Run notebook with papermill."""
    nb_path = Path(__file__).parent / 'Customer_Success_222331.ipynb'
    output_nb = Path(__file__).parent / 'Customer_Success_222331_executed.ipynb'
    
    logger.info("📖 Executing notebook with papermill...")
    logger.info(f"   Source: {nb_path}")
    logger.info(f"   Output: {output_nb}")
    
    try:
        cmd = [
            sys.executable, '-m', 'papermill',
            str(nb_path),
            str(output_nb),
            '--kernel', 'python3',
            '--timeout', '600'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("✅ Notebook executed successfully")
            return True
        else:
            logger.error("❌ Notebook execution failed")
            logger.error(result.stderr[:1000])
            return False
            
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return False

def run_with_jupyter():
    """Run notebook with jupyter nbconvert."""
    nb_path = Path(__file__).parent / 'Customer_Success_222331.ipynb'
    
    logger.info("📖 Executing notebook with jupyter...")
    
    try:
        cmd = [
            'jupyter', 'nbconvert',
            '--to', 'notebook',
            '--execute',
            '--ExecutePreprocessor.timeout=600',
            str(nb_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("✅ Notebook executed successfully")
            return True
        else:
            logger.error("❌ Notebook execution failed")
            logger.error(result.stderr[:1000] if result.stderr else result.stdout[:1000])
            return False
            
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return False

def main():
    logger.info("=" * 70)
    logger.info("🚀 Notebook Execution Pipeline")
    logger.info("=" * 70)
    
    # Try papermill first
    logger.info("\n🔄 Attempting with papermill...")
    if run_with_papermill():
        logger.info("✅ Success with papermill!")
        return 0
    
    # Fallback to jupyter
    logger.info("\n🔄 Attempting with jupyter...")
    if run_with_jupyter():
        logger.info("✅ Success with jupyter!")
        return 0
    
    logger.error("\n❌ Both methods failed")
    return 1

if __name__ == '__main__':
    sys.exit(main())
