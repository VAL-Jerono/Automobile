"""
Simplified FastAPI server for Karbima Agency.
Standalone server without complex module dependencies.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import mysql.connector
from mysql.connector import Error
import pandas as pd
import logging
from contextlib import contextmanager
import os

# Import ML predictions module
try:
    from ml_predictions import predict_churn_probability, predict_claims_probability, calculate_customer_segments, get_model_info
    ML_AVAILABLE = True
except Exception as e:
    logging.warning(f"ML predictions not available: {e}")
    ML_AVAILABLE = False

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Karbima Agency API",
    description="Agent-focused insurance platform",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': '',
    'database': 'insurance'
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
    """Execute a SELECT query and return results as a DataFrame."""
    try:
        with get_db_connection() as conn:
            df = pd.read_sql(query, conn, params=params)
            return df
    except Exception as e:
        logger.error(f"Query execution failed: {e}")
        raise

# ==================== ENDPOINTS ====================

@app.get("/")
async def root():
    """Root endpoint - API status."""
    return {
        "status": "operational",
        "service": "Karbima Agency API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.get("/api/v1/agent/stats/kpis")
async def get_kpis():
    """Get key performance indicators for the agent."""
    try:
        # Total policies
        total_q = "SELECT COUNT(*) as count FROM policies"
        total_df = execute_query(total_q)
        total_policies = int(total_df.iloc[0]['count'])
        
        # Active policies
        active_q = "SELECT COUNT(*) as count FROM policies WHERE lapse = 0"
        active_df = execute_query(active_q)
        active_policies = int(active_df.iloc[0]['count'])
        
        # Total customers
        customers_q = "SELECT COUNT(*) as count FROM customers"
        customers_df = execute_query(customers_q)
        total_customers = int(customers_df.iloc[0]['count'])
        
        # Total premium
        premium_q = "SELECT SUM(premium) as total FROM policies WHERE lapse = 0"
        premium_df = execute_query(premium_q)
        total_premium = float(premium_df.iloc[0]['total']) if premium_df.iloc[0]['total'] else 0
        
        # Average premium
        avg_q = "SELECT AVG(premium) as avg FROM policies WHERE lapse = 0"
        avg_df = execute_query(avg_q)
        avg_premium = float(avg_df.iloc[0]['avg']) if avg_df.iloc[0]['avg'] else 0
        
        # Lapse count
        lapse_q = "SELECT COUNT(*) as count FROM policies WHERE lapse = 1"
        lapse_df = execute_query(lapse_q)
        lapse_count = int(lapse_df.iloc[0]['count'])
        
        # Calculate retention rate
        retention_rate = ((active_policies / total_policies) * 100) if total_policies > 0 else 0
        
        return {
            "active_policies": active_policies,
            "total_premium_value": round(total_premium, 2),
            "avg_policy_value": round(avg_premium, 2),
            "retention_rate": round(retention_rate, 2),
            "renewals_due_30_days": 15,  # Placeholder
            "high_risk_customers": 50,  # Placeholder
            "total_customers": total_customers
        }
    except Exception as e:
        logger.error(f"Failed to get KPIs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/agent/at-risk")
async def get_at_risk_customers(limit: int = 50):
    """Get at-risk customers."""
    try:
        query = """
        SELECT 
            p.policy_id,
            p.customer_id,
            p.premium,
            p.n_claims_history,
            p.date_next_renewal,
            DATEDIFF(p.date_next_renewal, CURDATE()) as days_to_renewal
        FROM policies p
        WHERE p.lapse = 0
            AND (
                p.n_claims_history > 1 
                OR p.premium > (SELECT AVG(premium) * 1.5 FROM policies)
                OR DATEDIFF(p.date_next_renewal, CURDATE()) BETWEEN 0 AND 30
            )
        ORDER BY p.n_claims_history DESC
        LIMIT %s
        """
        
        df = execute_query(query, (limit,))
        customers = df.to_dict('records')
        
        # Calculate risk scores
        for customer in customers:
            risk_score = 0
            risk_score += customer.get('n_claims_history', 0) * 20
            days = customer.get('days_to_renewal')
            if days and days <= 7:
                risk_score += 30
            elif days and days <= 30:
                risk_score += 15
            if customer.get('premium', 0) > 500:
                risk_score += 10
            customer['risk_score'] = min(100, risk_score)
        
        customers_sorted = sorted(customers, key=lambda x: x.get('risk_score', 0), reverse=True)
        
        return {
            "total": len(customers_sorted),
            "customers": customers_sorted
        }
    except Exception as e:
        logger.error(f"Failed to get at-risk customers: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/agent/analytics/premium-distribution")
async def get_premium_distribution():
    """Get premium distribution."""
    try:
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
        distribution = df.to_dict('records')
        
        return {"distribution": distribution}
    except Exception as e:
        logger.error(f"Failed to get premium distribution: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/agent/analytics/policy-trends")
async def get_policy_trends():
    """Get policy trends over last 12 months."""
    try:
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
        trends = df.to_dict('records')
        
        return {"trends": trends}
    except Exception as e:
        logger.error(f"Failed to get policy trends: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== ML ENDPOINTS ====================

@app.get("/api/v1/agent/ml/info")
async def get_ml_info():
    """Get information about loaded ML models."""
    if not ML_AVAILABLE:
        return {
            "ml_enabled": False,
            "message": "ML models not loaded"
        }
    
    try:
        info = get_model_info()
        info['ml_enabled'] = True
        return info
    except Exception as e:
        logger.error(f"Failed to get ML info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/agent/ml/churn-predictions")
async def get_churn_predictions(limit: int = 100):
    """Get ML-powered churn predictions for active policies."""
    if not ML_AVAILABLE:
        raise HTTPException(status_code=503, detail="ML models not available")
    
    try:
        # Fetch customer data with all required features
        query = """
        SELECT 
            p.policy_id,
            p.customer_id,
            p.premium,
            p.n_claims_history,
            p.r_claims_history,
            p.seniority,
            p.policies_in_force,
            p.max_policies,
            p.max_products,
            p.payment,
            p.distribution_channel,
            p.area,
            p.type_risk,
            p.second_driver,
            TIMESTAMPDIFF(YEAR, c.date_birth, CURDATE()) as age,
            TIMESTAMPDIFF(YEAR, c.date_driving_licence, CURDATE()) as driving_experience,
            YEAR(CURDATE()) - v.year_matriculation as vehicle_age,
            v.power,
            v.cylinder_capacity,
            v.value_vehicle,
            v.n_doors,
            v.length,
            v.weight,
            DATEDIFF(p.date_next_renewal, CURDATE()) as days_to_renewal
        FROM policies p
        JOIN customers c ON p.customer_id = c.customer_id
        JOIN vehicles v ON p.vehicle_id = v.vehicle_id
        WHERE p.lapse = 0
        ORDER BY p.premium DESC
        LIMIT %s
        """
        
        df = execute_query(query, (limit,))
        
        if df.empty:
            return {"predictions": [], "total": 0}
        
        # Get ML predictions
        predictions = predict_churn_probability(df)
        
        # Sort by risk-weighted value (churn_prob * premium)
        predictions_sorted = sorted(
            predictions, 
            key=lambda x: x.get('risk_weighted_value', 0), 
            reverse=True
        )
        
        return {
            "predictions": predictions_sorted,
            "total": len(predictions_sorted),
            "model_version": predictions_sorted[0].get('model_version') if predictions_sorted else "unknown"
        }
    except Exception as e:
        logger.error(f"Failed to get churn predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/agent/ml/customer-segments")
async def get_customer_segments():
    """Get customer segmentation based on churn and premium."""
    if not ML_AVAILABLE:
        raise HTTPException(status_code=503, detail="ML models not available")
    
    try:
        # Get churn predictions for all active customers
        query = """
        SELECT 
            p.policy_id,
            p.customer_id,
            p.premium,
            p.n_claims_history,
            TIMESTAMPDIFF(YEAR, c.date_birth, CURDATE()) as age,
            TIMESTAMPDIFF(YEAR, c.date_driving_licence, CURDATE()) as driving_experience,
            YEAR(CURDATE()) - v.year_matriculation as vehicle_age,
            v.power,
            v.cylinder_capacity,
            v.value_vehicle,
            p.seniority
        FROM policies p
        JOIN customers c ON p.customer_id = c.customer_id
        JOIN vehicles v ON p.vehicle_id = v.vehicle_id
        WHERE p.lapse = 0
        LIMIT 1000
        """
        
        df = execute_query(query)
        
        if df.empty:
            return {"segments": {}, "total": 0}
        
        # Get churn predictions
        predictions = predict_churn_probability(df)
        
        # Extract churn probs and premiums
        churn_probs = [p['churn_probability'] for p in predictions]
        premiums = [p.get('premium', 0) for p in predictions]
        
        # Calculate segments
        segments = calculate_customer_segments(churn_probs, premiums)
        
        # Count by segment
        segment_counts = {}
        segment_details = {}
        
        for i, seg in enumerate(segments):
            if seg not in segment_counts:
                segment_counts[seg] = 0
                segment_details[seg] = {
                    'count': 0,
                    'total_premium': 0,
                    'avg_churn_prob': 0,
                    'customers': []
                }
            
            segment_counts[seg] += 1
            segment_details[seg]['count'] += 1
            segment_details[seg]['total_premium'] += premiums[i]
            segment_details[seg]['avg_churn_prob'] += churn_probs[i]
            
            # Add top 5 customers per segment
            if len(segment_details[seg]['customers']) < 5:
                segment_details[seg]['customers'].append({
                    'customer_id': predictions[i]['customer_id'],
                    'premium': premiums[i],
                    'churn_prob': churn_probs[i]
                })
        
        # Calculate averages
        for seg in segment_details:
            if segment_details[seg]['count'] > 0:
                segment_details[seg]['avg_churn_prob'] = round(
                    segment_details[seg]['avg_churn_prob'] / segment_details[seg]['count'], 
                    4
                )
        
        return {
            "segments": segment_details,
            "total_customers": len(predictions)
        }
    except Exception as e:
        logger.error(f"Failed to get customer segments: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/agent/ml/claims-forecast")
async def get_claims_forecast(limit: int = 100):
    """Get claims probability forecast."""
    if not ML_AVAILABLE:
        raise HTTPException(status_code=503, detail="ML models not available")
    
    try:
        # Fetch customer data
        query = """
        SELECT 
            p.policy_id,
            p.customer_id,
            p.premium,
            p.n_claims_history,
            TIMESTAMPDIFF(YEAR, c.date_birth, CURDATE()) as age,
            YEAR(CURDATE()) - v.year_matriculation as vehicle_age,
            v.value_vehicle
        FROM policies p
        JOIN customers c ON p.customer_id = c.customer_id
        JOIN vehicles v ON p.vehicle_id = v.vehicle_id
        WHERE p.lapse = 0
        ORDER BY p.n_claims_history DESC
        LIMIT %s
        """
        
        df = execute_query(query, (limit,))
        
        if df.empty:
            return {"forecast": [], "total_expected_claims": 0}
        
        # Get claims predictions
        claims_forecast = predict_claims_probability(df)
        
        # Calculate total expected claims cost
        total_expected = sum(c['expected_claim_cost'] for c in claims_forecast)
        
        return {
            "forecast": claims_forecast,
            "total_expected_claims_cost": round(total_expected, 2),
            "total_policies_analyzed": len(claims_forecast)
        }
    except Exception as e:
        logger.error(f"Failed to get claims forecast: {e}")
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
