"""
Database connection and query utilities for MySQL.
Provides data access for the Agent Portal.
"""

import mysql.connector
from mysql.connector import Error
import pandas as pd
import logging
from typing import Optional, Dict, List, Any
from contextlib import contextmanager
import os

logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    'host': os.getenv('MYSQL_HOST', 'localhost'),
    'port': int(os.getenv('MYSQL_PORT', '3306')),
    'user': os.getenv('MYSQL_USER', 'root'),
    'password': os.getenv('MYSQL_PASSWORD', ''),
    'database': os.getenv('MYSQL_DATABASE', 'insurance')
}

@contextmanager
def get_db_connection():
    """Context manager for database connections."""
    connection = None
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        yield connection
    except Error as e:
        logger.error(f"Database connection error: {e}")
        raise
    finally:
        if connection and connection.is_connected():
            connection.close()

def execute_query(query: str, params: Optional[tuple] = None) -> pd.DataFrame:
    """
    Execute a SELECT query and return results as a DataFrame.
    """
    try:
        with get_db_connection() as conn:
            df = pd.read_sql(query, conn, params=params)
            return df
    except Exception as e:
        logger.error(f"Query execution failed: {e}")
        raise

def execute_update(query: str, params: Optional[tuple] = None) -> int:
    """
    Execute an INSERT/UPDATE/DELETE query and return affected rows.
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            affected_rows = cursor.rowcount
            cursor.close()
            return affected_rows
    except Exception as e:
        logger.error(f"Update execution failed: {e}")
        raise

# ==================== AGENT PORTAL QUERIES ====================

def get_portfolio_summary() -> Dict[str, Any]:
    """
    Get high-level portfolio statistics for the Agent Dashboard.
    Returns: Total policies, customers, premiums, lapse rate, etc.
    """
    queries = {
        'total_policies': "SELECT COUNT(*) as count FROM policies",
        'active_policies': "SELECT COUNT(*) as count FROM policies WHERE lapse = 0",
        'total_customers': "SELECT COUNT(*) as count FROM customers",
        'total_premium': "SELECT SUM(premium) as total FROM policies WHERE lapse = 0",
        'lapse_count': "SELECT COUNT(*) as count FROM policies WHERE lapse = 1",
        'avg_premium': "SELECT AVG(premium) as avg FROM policies WHERE lapse = 0"
    }
    
    summary = {}
    for key, query in queries.items():
        result = execute_query(query)
        summary[key] = result.iloc[0, 0] if not result.empty else 0
    
    # Calculate lapse rate
    if summary['total_policies'] > 0:
        summary['lapse_rate'] = (summary['lapse_count'] / summary['total_policies']) * 100
    else:
        summary['lapse_rate'] = 0
    
    return summary

def get_at_risk_customers(limit: int = 50) -> List[Dict[str, Any]]:
    """
    Get customers with high lapse risk (based on historical patterns).
    For now, we identify at-risk based on:
    - High claims history
    - High premium
    - Long tenure without renewal
    """
    query = """
    SELECT 
        p.policy_id,
        c.customer_id,
        c.date_birth,
        v.year_matriculation,
        v.make,
        v.model,
        p.premium,
        p.n_claims_history,
        p.date_next_renewal,
        p.seniority,
        DATEDIFF(p.date_next_renewal, CURDATE()) as days_to_renewal
    FROM policies p
    JOIN customers c ON p.customer_id = c.customer_id
    JOIN vehicles v ON p.vehicle_id = v.vehicle_id
    WHERE p.lapse = 0
        AND (
            p.n_claims_history > 1 
            OR p.premium > (SELECT AVG(premium) * 1.5 FROM policies)
            OR DATEDIFF(p.date_next_renewal, CURDATE()) BETWEEN 0 AND 30
        )
    ORDER BY 
        p.n_claims_history DESC,
        days_to_renewal ASC
    LIMIT %s
    """
    
    df = execute_query(query, (limit,))
    return df.to_dict('records') if not df.empty else []

def get_premium_distribution() -> List[Dict[str, Any]]:
    """
    Get premium distribution by ranges for visualization.
    """
    query = """
    SELECT 
        CASE 
            WHEN premium < 200 THEN 'Under $200'
            WHEN premium BETWEEN 200 AND 500 THEN '$200-$500'
            WHEN premium BETWEEN 500 AND 1000 THEN '$500-$1000'
            WHEN premium BETWEEN 1000 AND 2000 THEN '$1000-$2000'
            ELSE 'Over $2000'
        END as premium_range,
        COUNT(*) as count
    FROM policies
    WHERE lapse = 0
    GROUP BY premium_range
    ORDER BY MIN(premium)
    """
    
    df = execute_query(query)
    return df.to_dict('records') if not df.empty else []

def get_policy_trends_by_month() -> List[Dict[str, Any]]:
    """
    Get new policy trends over time.
    """
    query = """
    SELECT 
        DATE_FORMAT(date_start_contract, '%Y-%m') as month,
        COUNT(*) as new_policies,
        SUM(premium) as total_premium
    FROM policies
    WHERE date_start_contract >= DATE_SUB(CURDATE(), INTERVAL 12 MONTH)
    GROUP BY month
    ORDER BY month DESC
    LIMIT 12
    """
    
    df = execute_query(query)
    return df.to_dict('records') if not df.empty else []

def get_vehicle_distribution() -> List[Dict[str, Any]]:
    """
    Get distribution of vehicles by make/type.
    """
    query = """
    SELECT 
        v.make,
        COUNT(*) as count
    FROM vehicles v
    JOIN policies p ON v.vehicle_id = p.vehicle_id
    WHERE p.lapse = 0
    GROUP BY v.make
    ORDER BY count DESC
    LIMIT 10
    """
    
    df = execute_query(query)
    return df.to_dict('records') if not df.empty else []

def search_policy(policy_id: Optional[int] = None, customer_id: Optional[int] = None) -> Dict[str, Any]:
    """
    Search for a specific policy by ID or customer ID.
    """
    if policy_id:
        query = """
        SELECT 
            p.*,
            c.date_birth,
            c.date_driving_licence,
            v.make,
            v.model,
            v.year_matriculation,
            v.power,
            v.value_vehicle
        FROM policies p
        JOIN customers c ON p.customer_id = c.customer_id
        JOIN vehicles v ON p.vehicle_id = v.vehicle_id
        WHERE p.policy_id = %s
        """
        df = execute_query(query, (policy_id,))
    elif customer_id:
        query = """
        SELECT 
            p.*,
            c.date_birth,
            v.make,
            v.model
        FROM policies p
        JOIN customers c ON p.customer_id = c.customer_id
        JOIN vehicles v ON p.vehicle_id = v.vehicle_id
        WHERE p.customer_id = %s
        ORDER BY p.date_start_contract DESC
        """
        df = execute_query(query, (customer_id,))
    else:
        return {}
    
    return df.to_dict('records')[0] if not df.empty else {}

def get_renewals_due(days: int = 30) -> List[Dict[str, Any]]:
    """
    Get policies due for renewal in the next N days.
    """
    query = """
    SELECT 
        p.policy_id,
        p.customer_id,
        c.date_birth,
        v.make,
        v.model,
        p.premium,
        p.date_next_renewal,
        DATEDIFF(p.date_next_renewal, CURDATE()) as days_remaining
    FROM policies p
    JOIN customers c ON p.customer_id = c.customer_id
    JOIN vehicles v ON p.vehicle_id = v.vehicle_id
    WHERE p.lapse = 0
        AND DATEDIFF(p.date_next_renewal, CURDATE()) BETWEEN 0 AND %s
    ORDER BY days_remaining ASC
    """
    
    df = execute_query(query, (days,))
    return df.to_dict('records') if not df.empty else []
