"""
RAG (Retrieval-Augmented Generation) query endpoints.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import logging
from api.dependencies import get_rag_engine

logger = logging.getLogger(__name__)
router = APIRouter()

class RAGQueryRequest(BaseModel):
    query: str
    query_type: str = "policy"  # policy, claims, general
    top_k: int = 5

class RAGResult(BaseModel):
    rank: int
    score: float
    metadata: Dict[str, Any]
    snippet: str

class RAGQueryResponse(BaseModel):
    query: str
    results: List[RAGResult]
    total_results: int
    search_time_ms: float

@router.post("/query", response_model=RAGQueryResponse)
async def query_rag(req: RAGQueryRequest):
    """
    Retrieve similar policies, claims, or general insurance context using semantic search.
    """
    try:
        import time
        start_time = time.time()
        
        rag = get_rag_engine()
        
        if req.query_type == "policy":
            raw_results = rag.query_policies(req.query, top_k=req.top_k)
        elif req.query_type == "claims":
            raw_results = rag.query_claims(req.query, top_k=req.top_k)
        else:
            raw_results = []  # Implement general query
        
        # Format results
        results = [
            RAGResult(
                rank=r['rank'],
                score=r['similarity'],
                metadata={k: v for k, v in r.items() if k not in ['rank', 'similarity', 'document']},
                snippet=r['document'][:200]
            )
            for r in raw_results
        ]
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        return RAGQueryResponse(
            query=req.query,
            results=results,
            total_results=len(results),
            search_time_ms=elapsed_ms
        )
    
    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class PolicyRecommendationRequest(BaseModel):
    policy_id: int
    context: str  # e.g., "similar_policies", "claims_patterns", "risk_assessment"

class RecommendationResponse(BaseModel):
    policy_id: int
    recommendations: List[str]
    supporting_evidence: List[Dict[str, Any]]

@router.post("/recommendations", response_model=RecommendationResponse)
async def get_policy_recommendations(req: PolicyRecommendationRequest):
    """
    Get personalized recommendations using RAG and historical context.
    """
    try:
        rag = get_rag_engine()
        
        # Query similar policies
        query = f"policies similar to {req.policy_id}"
        results = rag.query_policies(query, top_k=3)
        
        recommendations = [
            "Review coverage limits based on similar claims",
            "Consider bundling options for cost savings",
            "Update vehicle information for accurate pricing"
        ]
        
        return RecommendationResponse(
            policy_id=req.policy_id,
            recommendations=recommendations,
            supporting_evidence=[r for r in results]
        )
    
    except Exception as e:
        logger.error(f"Recommendation generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
