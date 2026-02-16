"""
Health check routes
"""

from fastapi import APIRouter

router = APIRouter()

@router.get("/ping")
async def ping():
    """Simple ping endpoint"""
    return {"status": "pong"}
