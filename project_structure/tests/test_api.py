"""
Unit tests for API endpoints.
Run with: pytest tests/test_api.py
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

class TestHealthCheck:
    def test_health_check(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_root_endpoint(self):
        response = client.get("/")
        assert response.status_code == 200
        assert "Insurance Risk Platform API" in response.json()["service"]

class TestPredictionEndpoints:
    def test_predict_lapse_valid(self):
        payload = {
            "policy_id": 123,
            "age": 45,
            "vehicle_age": 3,
            "premium": 250.0,
            "claims_history": 1,
            "second_driver": 0,
            "type_fuel": "P"
        }
        response = client.post("/api/v1/predict/lapse", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "lapse_probability" in data
        assert 0 <= data["lapse_probability"] <= 1
        assert data["lapse_risk"] in ["Low", "Medium", "High"]

    def test_predict_lapse_invalid_age(self):
        payload = {
            "policy_id": 123,
            "age": 150,  # Invalid
            "vehicle_age": 3,
            "premium": 250.0
        }
        response = client.post("/api/v1/predict/lapse", json=payload)
        assert response.status_code == 422  # Validation error

    def test_predict_risk_score(self):
        payload = {
            "policy_id": 456,
            "age": 35,
            "vehicle_age": 5,
            "premium": 300.0,
            "claims_history": 2
        }
        response = client.post("/api/v1/predict/risk_score", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "risk_score" in data
        assert 0 <= data["risk_score"] <= 100

class TestExplanationEndpoints:
    def test_explain_prediction(self):
        payload = {
            "prediction_id": "pred_123",
            "model_type": "ensemble",
            "include_llm_narrative": False
        }
        response = client.post("/api/v1/explain/prediction", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "top_features" in data
        assert len(data["top_features"]) > 0

class TestRAGEndpoints:
    def test_rag_query_policies(self):
        payload = {
            "query": "high-premium vehicles",
            "query_type": "policy",
            "top_k": 5
        }
        response = client.post("/api/v1/rag/query", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "search_time_ms" in data

class TestModelManagement:
    def test_get_model_info(self):
        response = client.get("/api/v1/models/info")
        assert response.status_code == 200
        models = response.json()
        assert len(models) > 0
        assert "name" in models[0]
        assert "accuracy" in models[0]

    def test_drift_check(self):
        response = client.get("/api/v1/models/drift_check")
        assert response.status_code == 200
        data = response.json()
        assert "drift_detected" in data
        assert "drift_score" in data
