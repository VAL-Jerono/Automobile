#!/usr/bin/env python3
"""
Test LLM integration - Customer & Admin workflows
"""

import sys
import requests
import json

API_BASE = "http://localhost:8001/api/v1"

print("🧪 Testing LLM Integration\n" + "="*60)

# Test 1: Customer Vehicle Check
print("\n1️⃣  Testing Customer Vehicle Check...")
print("   Simulating: Customer checking 2019 Toyota Camry")

vehicle_data = {
    "make": "Toyota",
    "model": "Camry",
    "year": 2019,
    "fuel_type": "P",
    "power": 200,
    "usage": "personal",
    "customer_age": 35
}

try:
    response = requests.post(
        f"{API_BASE}/llm/check-vehicle",
        json=vehicle_data,
        timeout=60
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"   ✅ Status: {result['status']}")
        print(f"   🚗 Vehicle: {result['vehicle']}")
        print(f"   📝 Assessment: {result['assessment'][:150]}...")
        print(f"   ✓ Can proceed: {result['can_proceed_to_quote']}")
    else:
        print(f"   ❌ Error: {response.status_code}")
        
except Exception as e:
    print(f"   ❌ Failed: {e}")

# Test 2: Underwriter Query
print("\n2️⃣  Testing Underwriter AI Assistant...")
print("   Query: What factors increase lapse risk?")

query_data = {
    "query": "What are the top 3 factors that increase insurance policy lapse risk?",
    "context": None
}

try:
    response = requests.post(
        f"{API_BASE}/llm/underwriter-assist",
        json=query_data,
        timeout=60
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"   ✅ Status: {result['status']}")
        print(f"   💬 Query: {result['query']}")
        print(f"   🤖 Answer: {result['answer'][:200]}...")
    else:
        print(f"   ❌ Error: {response.status_code}")
        
except Exception as e:
    print(f"   ❌ Failed: {e}")

# Test 3: Risk Assessment
print("\n3️⃣  Testing Risk Assessment...")
print("   Assessing: High-power BMW X5")

risk_data = {
    "policy_data": {
        "make_model": "2020 BMW X5",
        "power": "335 HP",
        "fuel_type": "Diesel",
        "owner_age": 28,
        "claims_last_year": 2
    }
}

try:
    response = requests.post(
        f"{API_BASE}/llm/assess-risk",
        json=risk_data,
        timeout=60
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"   ✅ Status: {result['status']}")
        print(f"   ⚠️  Risk Level: {result['risk_level'].upper()}")
        print(f"   📋 Assessment: {result['assessment'][:200]}...")
    else:
        print(f"   ❌ Error: {response.status_code}")
        
except Exception as e:
    print(f"   ❌ Failed: {e}")

print("\n" + "="*60)
print("✨ LLM Integration Test Complete!")
print("\nNext steps:")
print("  • Open http://localhost:3000 (Customer Portal)")
print("  • Go to 'Get Quote' section")
print("  • Test 'Check My Vehicle with AI' button")
print("  • Open http://localhost:3000/admin.html (Admin Dashboard)")
print("  • Go to 'AI Assistant' section")
print("  • Test underwriter queries")
print("="*60)
