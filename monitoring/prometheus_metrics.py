"""
Prometheus metrics for API monitoring
"""

from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi import Request
import time
from functools import wraps

# Define metrics
api_requests_total = Counter(
    'api_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status']
)

api_request_duration_seconds = Histogram(
    'api_request_duration_seconds',
    'API request duration in seconds',
    ['method', 'endpoint']
)

model_prediction_time_seconds = Histogram(
    'model_prediction_time_seconds',
    'Model prediction time in seconds',
    ['model_name']
)

model_predictions_total = Counter(
    'model_predictions_total',
    'Total model predictions',
    ['model_name', 'risk_category']
)

active_requests = Gauge(
    'active_requests',
    'Number of active requests'
)

churn_predictions_high_risk_count = Gauge(
    'churn_predictions_high_risk_count',
    'Number of high risk churn predictions in last batch'
)

# Middleware for request tracking
async def metrics_middleware(request: Request, call_next):
    """Middleware to track API metrics"""
    active_requests.inc()
    start_time = time.time()
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    
    # Record metrics
    api_requests_total.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    api_request_duration_seconds.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(duration)
    
    active_requests.dec()
    
    return response

def track_prediction(model_name: str):
    """Decorator to track model predictions"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            result = await func(*args, **kwargs)
            duration = time.time() - start_time
            
            # Record prediction time
            model_prediction_time_seconds.labels(
                model_name=model_name
            ).observe(duration)
            
            # Record prediction count
            risk_category = getattr(result, 'risk_category', 'UNKNOWN')
            model_predictions_total.labels(
                model_name=model_name,
                risk_category=risk_category
            ).inc()
            
            return result
        return wrapper
    return decorator
