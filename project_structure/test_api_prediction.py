#!/usr/bin/env python3
"""
Test API prediction endpoint
"""
import requests
import json

# Test data matching the PolicyData schema
test_policy = {
    "policy_id": 12345,
    "age": 35,
    "vehicle_age": 5,
    "premium": 300.50,
    "claims_history": 1,
    "second_driver": 0,
    "type_fuel": "P"
}

# Test health endpoint first
print("Testing health endpoint...")
health_response = requests.get("http://localhost:8001/health")
print(f"Health status: {health_response.status_code}")
print(f"Response: {health_response.json()}\n")

# Test prediction endpoint
print("Testing prediction endpoint...")
prediction_url = "http://localhost:8001/api/v1/predict/lapse"
headers = {"Content-Type": "application/json"}

try:
    response = requests.post(prediction_url, json=test_policy, headers=headers)
    print(f"Status code: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print("\n✓ Prediction successful!")
        print(json.dumps(result, indent=2))
    else:
        print(f"✗ Error: {response.status_code}")
        print(response.text)
except Exception as e:
    print(f"✗ Request failed: {e}")
