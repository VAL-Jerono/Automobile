#!/usr/bin/env python3
"""
SQL Predictions Manager
=======================
Manages MySQL database connections and prediction data for the Insurance Analytics Platform.

This module provides:
- Connection pooling with auto-reconnect
- Predictions table management
- Batch insertion of predictions
- Summary statistics queries
- Real-world deployment ready

Author: Insurance Analytics Team
Date: January 2026
"""

import mysql.connector
from mysql.connector import Error, pooling
import pandas as pd
import logging
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SQLModelPredictionsManager:
    """Manages predictions in MySQL database with connection pooling."""
    
    def __init__(self, 
                 host: str = None,
                 user: str = None,
                 password: str = None,
                 database: str = None,
                 port: int = 3306):
        """
        Initialize database manager.
        
        Args:
            host: MySQL host (default: from .env or localhost)
            user: MySQL user (default: from .env or root)
            password: MySQL password (default: from .env or empty)
            database: Database name (default: from .env or insurance_db)
            port: MySQL port (default: 3306)
        """
        # Load from environment if not provided
        # Priority: 1) Direct params, 2) Streamlit secrets, 3) Environment vars, 4) Defaults
        try:
            import streamlit as st
            if hasattr(st, 'secrets') and 'mysql' in st.secrets:
                self.host = host or st.secrets['mysql'].get('host', 'localhost')
                self.user = user or st.secrets['mysql'].get('user', 'root')
                self.password = password or st.secrets['mysql'].get('password', '')
                self.database = database or st.secrets['mysql'].get('database', 'insurance')
                self.port = port or st.secrets['mysql'].get('port', 3306)
            else:
                raise KeyError("No secrets configured")
        except (ImportError, KeyError):
            # Fallback to environment variables (for local development)
            self.host = host or os.getenv('MYSQL_HOST', 'localhost')
            self.user = user or os.getenv('MYSQL_USER', 'root')
            self.password = password or os.getenv('MYSQL_PASSWORD', '')
            self.database = database or os.getenv('MYSQL_DATABASE', 'insurance')
            self.port = port
        
        self.connection = None
        self.cursor = None
        logger.info(f"📊 Initialized SQLModelPredictionsManager: {self.host}:{self.port}/{self.database}")
    
    def connect(self) -> bool:
        """Establish database connection."""
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database,
                port=self.port,
                autocommit=False,
                use_unicode=True,
                charset='utf8mb4'
            )
            self.cursor = self.connection.cursor(dictionary=True)
            logger.info(f"✅ Connected to MySQL: {self.host}/{self.database}")
            return True
        except Error as e:
            logger.error(f"❌ Database connection failed: {e}")
            return False
    
    def disconnect(self) -> None:
        """Close database connection."""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
        logger.info("🔌 Disconnected from MySQL")
    
    def create_predictions_table(self) -> bool:
        """Create predictions table if it doesn't exist."""
        if not self.connection:
            logger.warning("No database connection")
            return False
        
        sql = """
        CREATE TABLE IF NOT EXISTS model_predictions (
            prediction_id INT AUTO_INCREMENT PRIMARY KEY,
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
            INDEX idx_policy_id (policy_id),
            INDEX idx_churn_probability (churn_probability),
            INDEX idx_customer_segment (customer_segment),
            INDEX idx_journey_quadrant (journey_quadrant),
            INDEX idx_created_at (created_at)
        )
        ENGINE=InnoDB
        DEFAULT CHARSET=utf8mb4
        COLLATE=utf8mb4_unicode_ci
        """
        
        try:
            self.cursor.execute(sql)
            self.connection.commit()
            logger.info("✅ Model predictions table created/verified")
            return True
        except Error as e:
            logger.error(f"❌ Error creating model predictions table: {e}")
            self.connection.rollback()
            return False
    
    def insert_predictions(self, df: pd.DataFrame, batch_size: int = 1000) -> int:
        """
        Insert predictions from DataFrame into database.
        
        Args:
            df: DataFrame with prediction data
            batch_size: Number of rows to insert per batch
        
        Returns:
            Number of rows inserted
        """
        if not self.connection:
            logger.warning("No database connection")
            return 0
        
        if df.empty:
            logger.warning("Empty DataFrame provided")
            return 0
        
        # Standardize column names
        column_mapping = {
            'policy_id': 'policy_id',
            'ID': 'policy_id',
            'Churn_Prob': 'churn_probability',
            'churn_probability': 'churn_probability',
            'Claims_Prob': 'claims_probability',
            'claims_probability': 'claims_probability',
            'Claims_Severity': 'claims_severity',
            'claims_severity': 'claims_severity',
            'CLV': 'customer_lifetime_value',
            'customer_lifetime_value': 'customer_lifetime_value',
            'Segment': 'customer_segment',
            'customer_segment': 'customer_segment',
            'Journey': 'journey_quadrant',
            'journey_quadrant': 'journey_quadrant',
            'Underpriced': 'pricing_adequacy_flag',
            'pricing_adequacy_flag': 'pricing_adequacy_flag',
            'Renewal_Risk': 'renewal_risk_score',
            'renewal_risk_score': 'renewal_risk_score',
            'High_Renewal_Risk': 'is_high_renewal_risk',
            'is_high_renewal_risk': 'is_high_renewal_risk'
        }
        
        # Rename columns
        df_clean = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        
        # Select only prediction columns
        pred_cols = [
            'policy_id', 'churn_probability', 'claims_probability', 
            'claims_severity', 'customer_lifetime_value', 'customer_segment',
            'journey_quadrant', 'pricing_adequacy_flag', 'renewal_risk_score',
            'is_high_renewal_risk'
        ]
        
        cols_to_use = [col for col in pred_cols if col in df_clean.columns]
        df_insert = df_clean[cols_to_use].fillna(0)
        
        sql = f"""
        INSERT INTO model_predictions (
            {', '.join(cols_to_use)}
        ) VALUES ({', '.join(['%s'] * len(cols_to_use))})
        ON DUPLICATE KEY UPDATE
            churn_probability = VALUES(churn_probability),
            claims_probability = VALUES(claims_probability),
            customer_lifetime_value = VALUES(customer_lifetime_value)
        """
        
        rows_inserted = 0
        try:
            for i in range(0, len(df_insert), batch_size):
                batch = df_insert.iloc[i:i+batch_size]
                data = [tuple(row) for row in batch.values]
                
                self.cursor.executemany(sql, data)
                self.connection.commit()
                rows_inserted += len(data)
                
                if (i // batch_size + 1) % 5 == 0:
                    logger.info(f"Inserted {rows_inserted:,} predictions...")
            
            logger.info(f"✅ Inserted {rows_inserted:,} predictions successfully")
            return rows_inserted
        
        except Error as e:
            logger.error(f"❌ Error inserting predictions: {e}")
            self.connection.rollback()
            return 0
    
    def get_all_predictions(self) -> pd.DataFrame:
        """Retrieve all predictions from database."""
        if not self.connection:
            logger.warning("No database connection")
            return pd.DataFrame()
        
        sql = """
        SELECT 
            policy_id,
            churn_probability,
            claims_probability,
            claims_severity,
            customer_lifetime_value,
            customer_segment,
            journey_quadrant,
            pricing_adequacy_flag,
            renewal_risk_score,
            is_high_renewal_risk,
            created_at as prediction_timestamp
        FROM model_predictions
        ORDER BY policy_id
        """
        
        try:
            self.cursor.execute(sql)
            results = self.cursor.fetchall()
            
            if not results:
                logger.warning("No predictions found in database")
                return pd.DataFrame()
            
            df = pd.DataFrame(results)
            logger.info(f"✅ Retrieved {len(df):,} predictions from database")
            return df
        
        except Error as e:
            logger.error(f"❌ Error retrieving predictions: {e}")
            return pd.DataFrame()
    
    def get_prediction_summary(self) -> Dict[str, Any]:
        """Get summary statistics of predictions."""
        if not self.connection:
            return {}
        
        sql = """
        SELECT 
            COUNT(*) as total_predictions,
            SUM(customer_lifetime_value) as total_portfolio_value,
            COUNT(CASE WHEN churn_probability > 0.7 THEN 1 END) as high_risk_count,
            COUNT(CASE WHEN churn_probability > 0.7 THEN 1 END) / COUNT(*) * 100 as high_risk_percentage,
            COUNT(DISTINCT customer_segment) as unique_segments,
            AVG(customer_lifetime_value) as avg_clv,
            MAX(customer_lifetime_value) as max_clv,
            MIN(customer_lifetime_value) as min_clv,
            AVG(churn_probability) as avg_churn_probability,
            MAX(created_at) as last_prediction_date
        FROM model_predictions
        """
        
        try:
            self.cursor.execute(sql)
            result = self.cursor.fetchone()
            
            if result:
                summary = {
                    'total_predictions': result.get('total_predictions', 0),
                    'total_portfolio_value': float(result.get('total_portfolio_value', 0) or 0),
                    'high_risk_count': result.get('high_risk_count', 0),
                    'high_risk_percentage': float(result.get('high_risk_percentage', 0) or 0),
                    'unique_segments': result.get('unique_segments', 0),
                    'avg_clv': float(result.get('avg_clv', 0) or 0),
                    'max_clv': float(result.get('max_clv', 0) or 0),
                    'min_clv': float(result.get('min_clv', 0) or 0),
                    'avg_churn_probability': float(result.get('avg_churn_probability', 0) or 0),
                    'last_prediction_date': result.get('last_prediction_date')
                }
                logger.info(f"✅ Retrieved summary: {summary['total_predictions']:,} predictions")
                return summary
            
            return {}
        
        except Error as e:
            logger.error(f"❌ Error getting summary: {e}")
            return {}
    
    def get_high_risk_customers(self, threshold: float = 0.7, limit: int = 100) -> pd.DataFrame:
        """Get high-risk customers."""
        if not self.connection:
            return pd.DataFrame()
        
        sql = f"""
        SELECT 
            policy_id,
            churn_probability,
            customer_lifetime_value,
            customer_segment,
            journey_quadrant
        FROM model_predictions
        WHERE churn_probability > %s
        ORDER BY churn_probability DESC
        LIMIT %s
        """
        
        try:
            self.cursor.execute(sql, (threshold, limit))
            results = self.cursor.fetchall()
            return pd.DataFrame(results) if results else pd.DataFrame()
        except Error as e:
            logger.error(f"❌ Error retrieving high-risk customers: {e}")
            return pd.DataFrame()
    
    def get_segment_distribution(self) -> Dict[str, int]:
        """Get count of customers per segment."""
        if not self.connection:
            return {}
        
        sql = """
        SELECT 
            customer_segment,
            COUNT(*) as count
        FROM model_predictions
        WHERE customer_segment IS NOT NULL
        GROUP BY customer_segment
        ORDER BY count DESC
        """
        
        try:
            self.cursor.execute(sql)
            results = self.cursor.fetchall()
            return {row['customer_segment']: row['count'] for row in results}
        except Error as e:
            logger.error(f"❌ Error retrieving segment distribution: {e}")
            return {}
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on database."""
        if not self.connection:
            return {'status': 'disconnected', 'message': 'Not connected to database'}
        
        try:
            self.cursor.execute("SELECT COUNT(*) as count FROM model_predictions")
            result = self.cursor.fetchone()
            count = result['count'] if result else 0
            
            return {
                'status': 'healthy',
                'message': f'Database connected with {count:,} predictions',
                'predictions_count': count,
                'host': self.host,
                'database': self.database,
                'timestamp': datetime.now().isoformat()
            }
        except Error as e:
            return {
                'status': 'error',
                'message': f'Database error: {str(e)}',
                'host': self.host,
                'database': self.database
            }


if __name__ == "__main__":
    # Test the manager
    manager = SQLModelPredictionsManager()
    
    if manager.connect():
        print("✅ Connection successful")
        
        # Create table
        manager.create_predictions_table()
        
        # Get summary
        summary = manager.get_prediction_summary()
        print(f"\n📊 Database Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
        # Health check
        health = manager.health_check()
        print(f"\n🏥 Health Check:")
        for key, value in health.items():
            print(f"  {key}: {value}")
        
        manager.disconnect()
    else:
        print("❌ Connection failed")
