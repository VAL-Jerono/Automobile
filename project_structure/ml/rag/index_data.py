"""
Script to index insurance data for RAG.
Loads data from CSV and indexes it using RAGEngine.
"""

import pandas as pd
import logging
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from ml.rag.retrieval import RAGEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting RAG indexing...")
    
    # Initialize RAG Engine
    rag = RAGEngine()
    
    if not rag.client:
        logger.error("RAG engine could not be initialized (missing dependencies?)")
        return

    # Load Data
    # Prefer enriched data if available
    csv_path = 'Enriched_Motor_Data.csv'
    if not os.path.exists(csv_path):
        csv_path = 'Motor_vehicle_insurance_data.csv'
        
    if not os.path.exists(csv_path):
        # Try finding it in data/raw
        if os.path.exists('data/raw/Motor_vehicle_insurance_data.csv'):
            csv_path = 'data/raw/Motor_vehicle_insurance_data.csv'
        elif os.path.exists('../../Motor_vehicle_insurance_data.csv'):
             csv_path = '../../Motor_vehicle_insurance_data.csv'
        elif os.path.exists('../../Enriched_Motor_Data.csv'):
             csv_path = '../../Enriched_Motor_Data.csv'
        
    if not os.path.exists(csv_path):
        logger.error(f"Could not find dataset at {csv_path}")
        return
        
    logger.info(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path, sep=';', low_memory=False)
    
    # Rename columns to match RAGEngine expectation if needed
    # RAGEngine expects: policy_id, customer_age, type_fuel, vehicle_age, premium, n_claims_history, lapse
    # CSV has: ID, Date_birth, Type_fuel, Year_matriculation, Premium, N_claims_history, Lapse
    
    # Simple preprocessing for demo
    df['policy_id'] = df['ID']
    df['premium'] = df['Premium'].astype(str).str.replace(',', '.').astype(float)
    df['n_claims_history'] = df['N_claims_history']
    df['lapse'] = df['Lapse']
    df['type_fuel'] = df['Type_fuel']
    
    # Deduplicate by ID (keep first)
    df.drop_duplicates(subset=['ID'], inplace=True)
    
    # Calculate ages roughly
    df['customer_age'] = 2025 - pd.to_datetime(df['Date_birth'], dayfirst=True).dt.year
    df['vehicle_age'] = 2025 - df['Year_matriculation']
    
    # Index a sample to save time for demo
    sample_size = 1000
    logger.info(f"Indexing sample of {sample_size} policies...")
    rag.index_policies(df.head(sample_size))
    
    rag.persist()
    logger.info("Indexing complete!")

if __name__ == "__main__":
    main()
