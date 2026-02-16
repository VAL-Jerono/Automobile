# 🚀 QUICK START GUIDE - Layers 3 & 5 Activation

## Current Status ✅

- **Layer 1 (Data)**: MySQL database with predictions ✅ OPERATIONAL
- **Layer 2 (ML Models)**: Trained models saved as CSV ✅ OPERATIONAL
- **Layer 3 (API)**: FastAPI backend ✅ **READY TO ACTIVATE**
- **Layer 4 (Frontend)**: Streamlit dashboard ✅ OPERATIONAL
- **Layer 5 (MLOps)**: Prometheus + Grafana + MLflow ✅ **READY TO ACTIVATE**

---

## 🎯 Quick Activation (3 Options)

### Option 1: Full Docker Stack (Recommended)
```bash
cd /Users/leonida/Documents/automobile_claims/Automobile
docker-compose up -d
```

**Services Started:**
- ✅ MySQL (port 3306)
- ✅ FastAPI (port 8001)
- ✅ Streamlit (port 8501)
- ✅ MLflow (port 5000)
- ✅ Prometheus (port 9090)
- ✅ Grafana (port 3000)

**Access:**
- API Docs: http://localhost:8001/docs
- Dashboard: http://localhost:8501
- MLflow: http://localhost:5000
- Grafana: http://localhost:3000 (admin/admin)

---

### Option 2: Automated Script
```bash
cd /Users/leonida/Documents/automobile_claims/Automobile
./start_services.sh
```

Follow prompts to choose:
1. Docker Compose (full stack)
2. Local services (API + Streamlit only)

---

### Option 3: Manual Activation

#### Terminal 1 - FastAPI (Layer 3)
```bash
cd /Users/leonida/Documents/automobile_claims/Automobile
source venv/bin/activate  # Or create: python3 -m venv venv
pip install fastapi uvicorn[standard] prometheus-client mlflow

# Start API
uvicorn api.main:app --host 0.0.0.0 --port 8001 --reload
```

#### Terminal 2 - Streamlit (Layer 4)
```bash
cd /Users/leonida/Documents/automobile_claims/Automobile
source venv/bin/activate
streamlit run app.py
```

#### Terminal 3 - MLflow (Layer 5)
```bash
cd /Users/leonida/Documents/automobile_claims/Automobile
source venv/bin/activate
mlflow server --backend-store-uri sqlite:///mlflow.db \
              --default-artifact-root ./mlflow_artifacts \
              --host 0.0.0.0 --port 5000
```

#### Terminal 4 - Prometheus (Layer 5)
```bash
# Install Prometheus first: brew install prometheus
cd /Users/leonida/Documents/automobile_claims/Automobile
prometheus --config.file=monitoring/prometheus.yml
```

---

## 🧪 Testing the API

### Health Check
```bash
curl http://localhost:8001/health
```

### Churn Prediction
```bash
curl -X POST "http://localhost:8001/api/v1/predict/churn" \
  -H "Content-Type: application/json" \
  -d '{
    "policy_id": 123456,
    "age": 45,
    "tenure": 2.5,
    "premium": 350.0,
    "vehicle_age": 3,
    "claims_history": 1,
    "channel": "broker"
  }'
```

### Claims Frequency Prediction
```bash
curl -X POST "http://localhost:8001/api/v1/predict/claims-frequency" \
  -H "Content-Type: application/json" \
  -d '{
    "policy_id": 123456,
    "age": 45,
    "vehicle_age": 3,
    "vehicle_type": "Passenger",
    "area": "Urban",
    "power_hp": 120
  }'
```

### Customer Lifetime Value
```bash
curl -X POST "http://localhost:8001/api/v1/predict/clv" \
  -H "Content-Type: application/json" \
  -d '{
    "policy_id": 123456,
    "age": 45,
    "tenure": 2.5,
    "premium": 350.0,
    "claims_history": 1,
    "channel": "agent"
  }'
```

---

## 📊 Accessing Services

| Service | URL | Credentials |
|---------|-----|-------------|
| **API Docs** | http://localhost:8001/docs | None |
| **API Health** | http://localhost:8001/health | None |
| **Streamlit** | http://localhost:8501 | None |
| **MLflow** | http://localhost:5000 | None |
| **Prometheus** | http://localhost:9090 | None |
| **Grafana** | http://localhost:3000 | admin/admin |

---

## 📁 Project Structure (Updated)

```
Automobile/
├── api/                          # ✅ Layer 3 - FastAPI Backend
│   ├── __init__.py
│   ├── main.py                   # Main FastAPI app
│   ├── models.py                 # Model manager
│   ├── schemas.py                # Pydantic schemas
│   └── routes/
│       ├── predictions.py        # Prediction endpoints
│       └── health.py             # Health checks
│
├── monitoring/                   # ✅ Layer 5 - MLOps
│   ├── prometheus_metrics.py    # Metrics collection
│   ├── prometheus.yml            # Prometheus config
│   ├── alerts.yml                # Alert rules
│   └── mlflow_tracking.py        # MLflow integration
│
├── docker-compose.yml            # Multi-service orchestration
├── Dockerfile.api                # API container
├── Dockerfile.streamlit          # Frontend container
├── start_services.sh             # Automated deployment
├── .env                          # Environment variables
│
├── app.py                        # Layer 4 - Streamlit
├── *_model.csv                   # Layer 2 - ML Models (4 files)
├── scripts/                      # Layer 1 - Data scripts
└── README_UNIFIED.md             # This documentation
```

---

## 🔧 Environment Variables (.env)

```bash
# Database
DB_HOST=localhost
DB_PORT=3306
DB_NAME=insurance
DB_USER=insurance_app
DB_PASSWORD=insurance_pass_2024

# API
API_HOST=0.0.0.0
API_PORT=8001

# MLflow
MLFLOW_TRACKING_URI=file:./mlruns
MLFLOW_EXPERIMENT_NAME=insurance_analytics

# Monitoring
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
GRAFANA_PASSWORD=admin
```

---

## 🛠️ Troubleshooting

### Port Already in Use
```bash
# Find process using port
lsof -i :8001  # or :8501, :5000, etc.

# Kill process
kill -9 <PID>
```

### Models Not Loading
```bash
# Verify model files
ls -lh *.csv

# Expected files:
# - churn_model.csv
# - claims_frequency_model.csv
# - claims_severity_model.csv
# - clv_model.csv
```

### Docker Issues
```bash
# Reset Docker containers
docker-compose down -v
docker-compose up -d --build

# View logs
docker-compose logs -f
```

### MySQL Connection Failed
```bash
# Check MySQL status
mysql.server status  # macOS
sudo systemctl status mysql  # Linux

# Start MySQL
mysql.server start  # macOS
```

---

## 📈 Monitoring Metrics

### Prometheus Metrics Available
- `api_requests_total` - Total API requests
- `api_request_duration_seconds` - Request latency
- `model_prediction_time_seconds` - Model inference time
- `model_predictions_total` - Total predictions by model
- `churn_predictions_high_risk_count` - High-risk customers

### Grafana Dashboards
1. **Model Performance**: ROC-AUC, prediction counts
2. **API Health**: Latency, error rates, throughput
3. **Business Metrics**: High-risk customers, segment distribution
4. **System Health**: CPU, memory, disk usage

---

## ✅ Verification Checklist

After deployment, verify:

- [ ] API health check responds: `curl http://localhost:8001/health`
- [ ] API docs accessible: http://localhost:8001/docs
- [ ] Streamlit loads: http://localhost:8501
- [ ] Churn prediction works (see test command above)
- [ ] MLflow UI loads: http://localhost:5000
- [ ] Prometheus targets up: http://localhost:9090/targets
- [ ] Grafana dashboards load: http://localhost:3000

---

## 🚀 Production Deployment

For production deployment to cloud (AWS/GCP/Azure):

```bash
# Build and push Docker images
docker build -t your-registry/insurance-api:latest -f Dockerfile.api .
docker build -t your-registry/insurance-streamlit:latest -f Dockerfile.streamlit .

docker push your-registry/insurance-api:latest
docker push your-registry/insurance-streamlit:latest

# Deploy to Kubernetes, ECS, or Cloud Run
kubectl apply -f k8s/deployment.yaml  # If using K8s
```

See [README_UNIFIED.md](README_UNIFIED.md) for detailed production deployment guide.

---

## 📞 Support

**Issues?** Check logs:
- API: `logs/api.log` or `docker-compose logs api`
- Streamlit: `logs/streamlit.log` or `docker-compose logs streamlit`
- MLflow: `logs/mlflow.log` or `docker-compose logs mlflow`

**Questions?** valerie.jerono@strathmore.edu

---

**Version**: 1.0.0  
**Last Updated**: February 16, 2026  
**Status**: ✅ **LAYERS 3 & 5 READY FOR ACTIVATION**
