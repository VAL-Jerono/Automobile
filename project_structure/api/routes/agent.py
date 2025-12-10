"""
Agent Portal API endpoints.
Provides data-rich insights for insurance agents.
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
from api.database import (
    get_portfolio_summary,
    get_at_risk_customers,
    get_premium_distribution,
    get_policy_trends_by_month,
    get_vehicle_distribution,
    search_policy,
    get_renewals_due
)

logger = logging.getLogger(__name__)
router = APIRouter()

# ==================== RESPONSE MODELS ====================

class PortfolioSummary(BaseModel):
    total_policies: int
    active_policies: int
    total_customers: int
    total_premium: float
    avg_premium: float
    lapse_count: int
    lapse_rate: float

class AtRiskCustomer(BaseModel):
    policy_id: int
    customer_id: int
    days_to_renewal: Optional[int]
    premium: float
    n_claims_history: int
    risk_score: Optional[float] = None

# ==================== ENDPOINTS ====================

@router.get("/portfolio", response_model=PortfolioSummary)
async def get_agent_portfolio():
    """
    Get comprehensive portfolio summary for the agent dashboard.
    Shows key metrics: total policies, premiums, lapse rate, etc.
    """
    try:
        summary = get_portfolio_summary()
        return PortfolioSummary(**summary)
    except Exception as e:
        logger.error(f"Failed to get portfolio summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/at-risk")
async def get_at_risk_list(limit: int = Query(50, ge=1, le=200)):
    """
    Get list of customers at risk of lapsing.
    Prioritized by claims history and renewal proximity.
    """
    try:
        customers = get_at_risk_customers(limit)
        
        # Add simple risk score calculation
        for customer in customers:
            risk_score = 0
            # Higher claims = higher risk
            risk_score += customer.get('n_claims_history', 0) * 20
            # Renewal coming up soon = higher risk
            days_to_renewal = customer.get('days_to_renewal', 999)
            if days_to_renewal <= 7:
                risk_score += 30
            elif days_to_renewal <= 30:
                risk_score += 15
            # High premium = moderate risk
            if customer.get('premium', 0) > 500:
                risk_score += 10
            
            customer['risk_score'] = min(100, risk_score)
        
        # Sort by risk score
        customers_sorted = sorted(customers, key=lambda x: x.get('risk_score', 0), reverse=True)
        
        return {
            "total": len(customers_sorted),
            "customers": customers_sorted
        }
    except Exception as e:
        logger.error(f"Failed to get at-risk customers: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analytics/premium-distribution")
async def get_premium_analytics():
    """
    Get premium distribution for visualization.
    """
    try:
        distribution = get_premium_distribution()
        return {
            "distribution": distribution
        }
    except Exception as e:
        logger.error(f"Failed to get premium distribution: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analytics/policy-trends")
async def get_policy_trends():
    """
    Get new policy trends over the last 12 months.
    """
    try:
        trends = get_policy_trends_by_month()
        return {
            "trends": trends
        }
    except Exception as e:
        logger.error(f"Failed to get policy trends: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analytics/vehicle-distribution")
async def get_vehicle_analytics():
    """
    Get vehicle distribution by make.
    """
    try:
        distribution = get_vehicle_distribution()
        return {
            "distribution": distribution
        }
    except Exception as e:
        logger.error(f"Failed to get vehicle distribution: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/policy/{policy_id}")
async def get_policy_details(policy_id: int):
    """
    Get detailed information about a specific policy.
    """
    try:
        policy = search_policy(policy_id=policy_id)
        if not policy:
            raise HTTPException(status_code=404, detail="Policy not found")
        return policy
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get policy details: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/customer/{customer_id}/policies")
async def get_customer_policies(customer_id: int):
    """
    Get all policies for a specific customer.
    """
    try:
        policies = search_policy(customer_id=customer_id)
        if not policies:
            raise HTTPException(status_code=404, detail="Customer not found")
        return {
            "customer_id": customer_id,
            "policies": policies if isinstance(policies, list) else [policies]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get customer policies: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/renewals")
async def get_upcoming_renewals(days: int = Query(30, ge=1, le=90)):
    """
    Get policies due for renewal in the next N days.
    """
    try:
        renewals = get_renewals_due(days)
        return {
            "total": len(renewals),
            "renewals": renewals,
            "days_filter": days
        }
    except Exception as e:
        logger.error(f"Failed to get renewals: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats/kpis")
async def get_key_performance_indicators():
    """
    Get key performance indicators for the agent.
    """
    try:
        summary = get_portfolio_summary()
        renewals = get_renewals_due(30)
        at_risk = get_at_risk_customers(50)
        
        return {
            "active_policies": summary.get('active_policies', 0),
            "total_premium_value": round(summary.get('total_premium', 0), 2),
            "avg_policy_value": round(summary.get('avg_premium', 0), 2),
            "retention_rate": round(100 - summary.get('lapse_rate', 0), 2),
            "renewals_due_30_days": len(renewals),
            "high_risk_customers": len(at_risk),
            "total_customers": summary.get('total_customers', 0)
        }
    except Exception as e:
        logger.error(f"Failed to get KPIs: {e}")
        raise HTTPException(status_code=500, detail=str(e))
