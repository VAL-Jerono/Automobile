# Intelligent Insurance Risk Platform

A production-grade ML/AI system for motor vehicle insurance risk assessment, claim prediction, and policy recommendations using ensemble ML models, fine-tuned LLMs, and RAG intelligence.

## 🏗️ Architecture

The platform consists of four integrated layers:

### 1. **Data Layer**
- **MySQL (XAMPP)**: Normalized relational schema for policies, claims, and customer data
- **ETL Pipeline (Airflow)**: Orchestrated data ingestion, validation, and feature engineering
- **Feature Store**: Real-time feature serving and historical feature tracking

### 2. **ML Layer**
- **Ensemble Models**: XGBoost + LightGBM + Neural Networks for lapse and claims prediction
- **Fine-tuned LLM**: LoRA/QLoRA-tuned Ollama (Llama2) for domain-specific text generation
- **RAG System**: Vector embeddings + ChromaDB for policy/claims context retrieval

### 3. **API Layer**
- **FastAPI**: REST endpoints for predictions, explanations, RAG queries, and model management
- **Docker**: Multi-service containerization for reproducible deployments

### 4. **Monitoring**
- **MLflow**: Experiment tracking, model registry, and versioning
- **Prometheus + Grafana**: Real-time metrics, alerts, and dashboards
- **Drift Detection**: Automated data/model drift monitoring with retraining triggers

## 📦 Installation

### Prerequisites
- Python 3.10+
- Docker & Docker Compose
- XAMPP (MySQL) or standalone MySQL 8.0+
- Ollama (for local LLM inference)

### Setup

1. **Clone and install dependencies:**
   ```bash
   cd /Users/leonida/Documents/automobile_claims/project_structure
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your local settings
   ```

3. **Initialize database:**
   ```bash
   python data/scripts/init_db.py
   python data/scripts/load_raw_data.py
   ```

4. **Start services (Docker):**
   ```bash
   docker-compose up -d
   ```

5. **Train baseline models:**
   ```bash
   python ml/train_pipeline.py
   ```

6. **Launch API:**
   ```bash
   python api/main.py
   ```

## 🚀 Quick Start

### API Usage

```bash
# Predict lapse probability
curl -X POST http://localhost:8000/api/v1/predict/lapse \
  -H "Content-Type: application/json" \
  -d '{
    "policy_id": 123,
    "age": 45,
    "vehicle_age": 3,
    "premium": 250.0,
    "claims_history": 1
  }'

# Get RAG-based policy recommendations
curl -X POST http://localhost:8000/api/v1/rag/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What policy features similar to policy 123?"
  }'

# Generate explanation with fine-tuned LLM
curl -X POST http://localhost:8000/api/v1/explain/prediction \
  -H "Content-Type: application/json" \
  -d '{
    "prediction_id": "pred_456",
    "model_type": "ensemble"
  }'
```

## 📊 Project Structure

```
project_structure/
├── data/
│   ├── raw/                      # Original CSV files
│   ├── processed/                # ETL output
│   ├── scripts/
│   │   ├── init_db.py           # Database setup
│   │   ├── load_raw_data.py     # CSV → MySQL
│   │   └── airflow_dags/        # Airflow DAGs
│   └── schemas/                  # DB schemas
├── ml/
│   ├── models/
│   │   ├── ensemble.py          # XGBoost + LightGBM + NN pipeline
│   │   ├── neural_net.py        # TensorFlow model
│   │   └── llm_fine_tune.py     # LoRA/QLoRA training
│   ├── rag/
│   │   ├── embeddings.py        # Sentence-transformer pipeline
│   │   ├── vector_store.py      # ChromaDB integration
│   │   └── retrieval.py         # RAG query engine
│   ├── preprocessing.py          # Feature engineering
│   ├── train_pipeline.py         # End-to-end training
│   ├── evaluate.py              # Metrics & evaluation
│   └── drift_detection.py       # Data/model drift monitoring
├── api/
│   ├── main.py                  # FastAPI app
│   ├── routes/
│   │   ├── predictions.py       # Prediction endpoints
│   │   ├── explanations.py      # LLM explanation endpoints
│   │   ├── rag.py              # RAG query endpoints
│   │   └── model_mgmt.py       # Model management
│   ├── schemas.py               # Pydantic models
│   ├── dependencies.py          # Dependency injection
│   └── utils.py                 # Helper functions
├── monitoring/
│   ├── mlflow_tracking.py       # MLflow integration
│   ├── prometheus_metrics.py    # Prometheus setup
│   ├── grafana_dashboards/      # Dashboard definitions (JSON)
│   └── drift_alerts.py          # Alert rules
├── docker/
│   ├── Dockerfile.api           # API container
│   ├── Dockerfile.ollama        # Ollama container
│   ├── Dockerfile.mlflow        # MLflow container
│   └── docker-compose.yml       # Service orchestration
├── .github/
│   └── workflows/
│       ├── test.yml             # Unit & integration tests
│       ├── model_validation.yml # Model performance checks
│       └── deploy.yml           # Docker build & push
├── docs/
│   ├── ARCHITECTURE.md          # Detailed architecture
│   ├── API.md                   # API documentation
│   ├── DATA_PIPELINE.md         # ETL & feature engineering
│   └── DEPLOYMENT.md            # Production deployment guide
├── config.yaml                  # Main configuration
├── .env.example                 # Environment template
└── requirements.txt             # Python dependencies
```

## 🔧 Key Components

### Ensemble ML Pipeline
```python
# ml/models/ensemble.py
from ml.models.ensemble import InsuranceEnsembleModel

model = InsuranceEnsembleModel(config='config.yaml')
model.train(X_train, y_train)
predictions = model.predict(X_test)
explanations = model.explain(X_test)  # SHAP values
```

### RAG Query Engine
```python
# ml/rag/retrieval.py
from ml.rag.retrieval import RAGEngine

rag = RAGEngine(embedding_model='all-MiniLM-L6-v2', vector_db='chromadb')
results = rag.query("policies similar to policy #123")
```

### Fine-tuned LLM Integration
```python
# ml/models/llm_fine_tune.py
from ml.models.llm_fine_tune import OllamaFineTuner

tuner = OllamaFineTuner(base_model='llama2', method='LoRA')
tuner.train(insurance_texts, lora_rank=8)
explanation = tuner.generate("Explain claim denial for policy #456")
```

### FastAPI Endpoints
```python
# api/main.py
from fastapi import FastAPI
from api.routes import predictions, explanations, rag

app = FastAPI(title="Insurance Risk API", version="1.0.0")
app.include_router(predictions.router, prefix="/api/v1/predict")
app.include_router(explanations.router, prefix="/api/v1/explain")
app.include_router(rag.router, prefix="/api/v1/rag")
```

## 📈 Monitoring & Observability

### MLflow Experiment Tracking
```bash
# View experiments
mlflow ui --host 0.0.0.0 --port 5000
```

### Prometheus Metrics
```bash
# Scrape metrics at http://localhost:9090
# Predefined alerts: model drift, API latency, training failure
```

### Grafana Dashboards
- **Model Performance**: AUC, precision, recall over time
- **API Health**: Request latency, error rates, throughput
- **Data Drift**: Feature distribution changes, drift scores
- **LLM Quality**: Generation metrics, fine-tuning loss

## 🧪 Testing

```bash
# Unit tests
pytest tests/unit -v --cov=ml --cov=api

# Integration tests
pytest tests/integration -v

# Model validation
python tests/validate_models.py
```

## 🚢 Deployment

### Docker Compose (Local Development)
```bash
docker-compose -f docker/docker-compose.yml up -d
```

### GitHub Actions (CI/CD)
- **On PR**: Run tests, validate models, check coverage
- **On Merge**: Build Docker images, push to registry, deploy to staging
- **On Release**: Deploy to production with canary strategy

## 📚 Documentation

- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - Detailed system design and data flows
- **[API.md](docs/API.md)** - Full API reference with examples
- **[DATA_PIPELINE.md](docs/DATA_PIPELINE.md)** - Feature engineering & Airflow DAGs
- **[DEPLOYMENT.md](docs/DEPLOYMENT.md)** - Production setup and scaling

## 🔐 Security

- **API Authentication**: JWT tokens for endpoints (optional)
- **Database**: Encrypted credentials, network isolation
- **Model Storage**: Model registry with versioning and access control
- **Data Privacy**: PII detection and anonymization in logs

## 🤝 Contributing

1. Create a feature branch: `git checkout -b feature/my-feature`
2. Make changes and test: `pytest tests/`
3. Format code: `black . && isort .`
4. Commit and push: `git commit -m "Add feature" && git push origin feature/my-feature`
5. Open a pull request

## 📝 License

Proprietary - Insurance Risk Platform

## 📧 Support

For issues or questions, open a GitHub issue or contact the development team.

---

**Version**: 1.0.0  
**Last Updated**: December 2024  
**Status**: Production Ready
