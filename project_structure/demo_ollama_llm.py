#!/usr/bin/env python3
"""
Quick demo of Ollama LLM working with insurance data.
"""

import sys
sys.path.insert(0, '/Users/leonida/Documents/automobile_claims/project_structure')

from ml.models.llm_fine_tune import OllamaFineTuner
import pandas as pd

print("🤖 AutoGuard Insurance - Ollama LLM Demo\n" + "=" * 60)

# Initialize LLM
llm = OllamaFineTuner(base_model='phi3:mini')

# Test 1: Warm up model with simple query
print("\n1️⃣  Warming up model...")
response = llm.generate_text("Say hello", max_tokens=10)
print(f"   ✅ Model loaded: {response[:50]}...")

# Test 2: Insurance explanation
print("\n2️⃣  Generating insurance lapse explanation...")
explanation = llm.generate_text(
    "Explain in one sentence what causes insurance policy lapse.", 
    max_tokens=50
)
print(f"   📝 {explanation}")

# Test 3: Real policy analysis
print("\n3️⃣  Analyzing real policies from database...")
df = pd.read_csv('Motor vehicle insurance data.csv', sep=';', nrows=3)

for idx, row in df.iterrows():
    lapsed = "LAPSED" if row.get('Lapse', 0) == 1 else "ACTIVE"
    premium = row.get('Premium', 0)
    claims = row.get('N_claims_history', 0)
    
    print(f"\n   Policy #{idx + 1}: ${premium:.2f} premium, {claims} claims, {lapsed}")
    
    prompt = f"""This policy has ${premium:.2f} premium and {claims} prior claims. 
    It is currently {lapsed}. In one sentence, explain the key risk factor:"""
    
    analysis = llm.generate_text(prompt, max_tokens=50)
    print(f"   🤖 {analysis}")

# Test 4: Policy recommendation
print("\n4️⃣  Generating policy recommendation...")
customer = {
    "age": 42,
    "vehicle": "2019 Honda Accord",
    "claims": 0,
    "experience": "15 years"
}

recommendation = llm.generate_policy_recommendation(customer)
print(f"   💼 Recommendation:\n   {recommendation[:200]}...")

# Test 5: Risk assessment
print("\n5️⃣  Assessing vehicle risk...")
vehicle = {
    "make_model": "2021 Tesla Model 3",
    "power": "283 HP",
    "age": "4 years",
    "value": "$45,000",
    "fuel_type": "Electric"
}

assessment = llm.generate_risk_assessment(vehicle)
print(f"   ⚠️  Risk Assessment:\n   {assessment[:200]}...")

print("\n" + "=" * 60)
print("✨ Ollama LLM is fully operational!")
print("=" * 60)
print("\nNext steps:")
print("  • Integrate with FastAPI endpoints")
print("  • Add to admin dashboard")
print("  • Setup RAG with vector database")
print("  • Fine-tune on 191K policies")
