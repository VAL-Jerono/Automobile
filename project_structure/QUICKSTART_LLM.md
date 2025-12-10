# 🚀 Quick Start Guide - LLM Integration

## Start Everything (One Command)

```bash
cd /Users/leonida/Documents/automobile_claims/project_structure

# Terminal 1: Start API Server
source venv/bin/activate
python -m uvicorn api.main:app --host 0.0.0.0 --port 8001 --reload

# Terminal 2: Start Frontend
python frontend/serve.py
```

## Access Points

- **Customer Portal**: http://localhost:3000
- **Admin Dashboard**: http://localhost:3000/admin.html  
- **API Docs**: http://localhost:8001/docs
- **API Health**: http://localhost:8001/health

## Customer Workflow Test

1. Go to http://localhost:3000
2. Click **"Get Instant Quote"**
3. Fill **Step 1** (personal info)
4. In **Step 2**, enter:
   ```
   Make: Toyota
   Model: Camry
   Year: 2019
   ```
5. Click **"Check My Vehicle with AI"** 🤖
6. Wait 5-10 seconds
7. Read AI assessment
8. Proceed to quote

## Admin Workflow Test

1. Go to http://localhost:3000/admin.html
2. Click **"AI Assistant"** in sidebar
3. Type in chat:
   ```
   What factors increase insurance lapse risk?
   ```
4. Press **Enter** or click **"Ask AI"**
5. Wait for AI response
6. Try Quick Tools:
   - **Risk Assessment**: Enter policy ID → Click "Assess Risk"
   - **Explain Decision**: Select decision → Click "Generate Explanation"

## API Endpoints Quick Reference

### Customer Endpoints
```bash
# Vehicle Check
curl -X POST http://localhost:8001/api/v1/llm/check-vehicle \
  -H "Content-Type: application/json" \
  -d '{
    "make": "Toyota",
    "model": "Camry",
    "year": 2019,
    "fuel_type": "P",
    "power": 200
  }'
```

### Admin Endpoints
```bash
# Underwriter Query
curl -X POST http://localhost:8001/api/v1/llm/underwriter-assist \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What increases lapse risk?"
  }'

# Risk Assessment
curl -X POST http://localhost:8001/api/v1/llm/assess-risk \
  -H "Content-Type: application/json" \
  -d '{
    "policy_data": {
      "make_model": "2020 BMW X5",
      "power": "335 HP",
      "owner_age": 28
    }
  }'
```

## Troubleshooting

### API won't start (Port in use)
```bash
# Kill process on port 8001
lsof -ti:8001 | xargs kill -9

# Or kill all Python processes
killall -9 python
```

### Ollama not responding
```bash
# Check Ollama status
ollama list

# Restart Ollama service
ollama serve
```

### LLM timeout errors
- **Solution 1**: Reduce max_tokens in prompts (from 200 to 100)
- **Solution 2**: Use shorter, more specific prompts
- **Solution 3**: Wait for model to warm up (first call is slow)

## Features Overview

| Feature | Location | Purpose |
|---------|----------|---------|
| 🚗 AI Vehicle Check | Customer Portal → Get Quote → Step 2 | Pre-qualify vehicles before quote |
| 🤖 AI Chat | Admin Dashboard → AI Assistant | Answer underwriter questions |
| ⚠️ Risk Assessment | Admin Dashboard → Quick Tools | Evaluate policy risk |
| 📋 Explain Decision | Admin Dashboard → Quick Tools | Generate decision explanations |
| 💡 Recommendations | Admin Dashboard → Quick Tools | Suggest policy coverage |

## Expected Performance

- **First call**: 6-10 seconds (model loading)
- **Subsequent calls**: 5-7 seconds
- **Success rate**: ~85%
- **Model**: phi3:mini (2.2 GB)

## Support Files

- **Full Documentation**: `LLM_INTEGRATION_SUMMARY.md`
- **Status Report**: `LLM_STATUS_REPORT.md`
- **Test Script**: `test_llm_integration_api.py`
- **Demo Script**: `demo_ollama_llm.py`

## Quick Commands

```bash
# Test LLM health
curl http://localhost:8001/api/v1/llm/health

# Test API health
curl http://localhost:8001/health

# Run integration tests
python test_llm_integration_api.py

# Run quick demo
python demo_ollama_llm.py
```

---

**Everything is ready!** 🎉

Start the servers and test both workflows.
