"""
PROJECT STRUCTURE & FILE INVENTORY
Insurance Risk Platform - Production ML/AI System
"""

# COMPLETE PROJECT TREE

automobile_claims/project_structure/
│
├── 📄 ROOT CONFIGURATION FILES
│   ├── .env.example              [Environment variables template - 15 vars]
│   ├── config.yaml               [Centralized configuration - 60+ settings]
│   ├── requirements.txt           [Python dependencies - 60+ packages]
│   ├── setup.sh                  [Linux/macOS auto-setup script]
│   └── setup.bat                 [Windows auto-setup script]
│
├── 📚 DOCUMENTATION
│   ├── README.md                 [800+ line architecture & setup guide]
│   ├── IMPLEMENTATION_SUMMARY.md [Project status & progress tracking]
│   └── DEVELOPER_REFERENCE.md    [Quick commands & troubleshooting]
│
├── 📁 DATA LAYER (MySQL + ETL)
│   └── data/
│       ├── scripts/
│       │   ├── init_db.py        [Database schema initialization]
│       │   └── load_raw_data.py  [CSV→MySQL ETL pipeline (105K rows)]
│       └── schemas/
│           └── mysql_schema.py   [8-table normalized schema definition]
│
├── 🤖 ML LAYER (Models + RAG + LLM)
│   └── ml/
│       ├── models/
│       │   ├── ensemble.py       [XGBoost+LightGBM+NN with SHAP]
│       │   └── llm_fine_tune.py  [Ollama fine-tuning & generation]
│       ├── rag/
│       │   └── retrieval.py      [ChromaDB semantic search]
│       └── train_pipeline.py     [End-to-end training with MLflow]
│
├── 🌐 API LAYER (FastAPI)
│   └── api/
│       ├── main.py               [FastAPI app + lifecycle + middleware]
│       ├── dependencies.py        [Model/RAG/LLM dependency injection]
│       └── routes/
│           ├── predictions.py     [Lapse, claims, risk endpoints]
│           ├── explanations.py    [SHAP + LLM narrative explanations]
│           ├── rag.py             [Semantic search queries]
│           └── model_mgmt.py      [Model registry & drift checking]
│
├── 🐳 CONTAINERIZATION
│   └── docker/
│       ├── Dockerfile.api        [FastAPI service image]
│       ├── Dockerfile.ollama     [Ollama LLM service image]
│       └── docker-compose.yml    [6-service orchestration]
│
├── 📊 MONITORING
│   └── monitoring/
│       └── prometheus.yml        [Metrics scrape configuration]
│
├── ✅ TESTING
│   └── tests/
│       ├── test_api.py           [11 API endpoint tests]
│       └── test_models.py        [ML model validation tests]
│
├── 🔄 CI/CD WORKFLOWS
│   └── .github/workflows/
│       ├── test.yml              [Unit tests + coverage + linting]
│       ├── model_validation.yml  [Train + validate model AUC]
│       └── deploy.yml            [Docker build/push + deployment]
│
└── 📋 PROJECT ARTIFACTS
    └── docs/                     [Reserved for additional docs]


# FILE COUNT SUMMARY

✅ COMPLETE PROJECT: 27 Files Total

Configuration Files:        5  (.env, config, requirements, setup scripts)
Documentation:              3  (README, SUMMARY, REFERENCE)
Data Layer:                 3  (schema, init, ETL loader)
ML Layer:                   4  (ensemble, LLM, RAG, training pipeline)
API Layer:                  6  (main, dependencies, 4 route modules)
Containerization:           3  (3 Docker files)
Monitoring:                 1  (Prometheus config)
Testing:                    2  (API tests, model tests)
CI/CD:                      3  (GitHub Actions workflows)
────────────────────────────
Total:                     30+ files
Total Size:                ~50 KB of production Python code


# LAYER-BY-LAYER BREAKDOWN

## LAYER 1: DATA (MySQL + ETL)
├─ Database: 8-table normalized schema
│  ├─ customers (customer_id, demographics, computed age)
│  ├─ vehicles (vehicle_id, specs, computed vehicle_age)
│  ├─ policies (policy_id, dates, target: Lapse)
│  ├─ claims (claim_id, amounts, status)
│  ├─ predictions (model outputs + confidence)
│  ├─ feature_store (historical snapshots)
│  ├─ audit_log (ETL tracking)
│  └─ indexes for query optimization
│
├─ ETL Pipeline: CSV→MySQL
│  ├─ Read Motor_vehicle_insurance_data.csv (105,555 rows, 30 columns)
│  ├─ Parse dates (DD/MM/YYYY format handling)
│  ├─ Deduplicate customers & vehicles
│  ├─ Map relationships (policy→customer, policy→vehicle)
│  └─ Batch load with progress tracking (tqdm)
│
└─ Feature Engineering: Derived columns
   ├─ contract_days (date_start - date_lapse)
   ├─ age_at_start (age when policy started)
   ├─ licence_years (driving experience)
   └─ vehicle_age (matriculation_year to now)

## LAYER 2: ML (Models + Interpretability)
├─ Ensemble Classifier (Lapse/Risk Prediction)
│  ├─ XGBoost component (n_estimators=100, depth=6)
│  ├─ LightGBM component (n_estimators=100, leaves=31)
│  ├─ Neural Network (3-layer with dropout: [128→64→32])
│  ├─ Ensemble strategy: equal weighting (1/3 each)
│  ├─ Output: Probability [0.0-1.0]
│  └─ Interpretability: SHAP TreeExplainer on XGBoost
│
├─ RAG System (Semantic Search)
│  ├─ Embedding Model: all-MiniLM-L6-v2 (384-dim vectors)
│  ├─ Vector DB: ChromaDB with cosine similarity
│  ├─ Indexed Datasets: Policies + Claims history
│  ├─ Query: Returns top-K similar documents
│  └─ Use Case: Find similar policies for recommendations
│
├─ Fine-tuned LLM (Ollama + LoRA)
│  ├─ Base Model: llama2 (quantized Q4_K_M)
│  ├─ Fine-tuning: LoRA (rank=8, alpha=16)
│  ├─ Tasks:
│  │  ├─ generate_claim_explanation()
│  │  ├─ generate_policy_recommendation()
│  │  ├─ generate_risk_assessment()
│  │  └─ batch_generate_explanations()
│  └─ Inference: HTTP API to Ollama service
│
└─ Training Pipeline (MLflow Integration)
   ├─ Data loading (from CSV with 80/10/10 split)
   ├─ Preprocessing (encoding, scaling, imputation)
   ├─ Model training (with validation monitoring)
   ├─ Cross-validation (5-fold stratified)
   ├─ Metrics logging (accuracy, precision, recall, F1, AUC)
   └─ Model persistence (joblib + HDF5)

## LAYER 3: API (FastAPI REST)
├─ Core Application (main.py)
│  ├─ FastAPI initialization with lifespan management
│  ├─ CORS middleware for cross-origin requests
│  ├─ Health check endpoints
│  ├─ Auto-generated documentation (Swagger UI + ReDoc)
│  └─ Error handling & logging
│
├─ Route: Predictions (/api/v1/predict/*)
│  ├─ POST /lapse
│  │  ├─ Input: PolicyData (age, vehicle_age, premium, claims, fuel_type)
│  │  └─ Output: Probability, risk_level (Low/Medium/High), action
│  ├─ POST /risk_score
│  │  ├─ Input: PolicyData
│  │  └─ Output: 0-100 score, category, factors, mitigation strategies
│  └─ POST /claims_amount
│     ├─ Input: ClaimsData
│     └─ Output: Expected amount, frequency, severity
│
├─ Route: Explanations (/api/v1/explain/*)
│  ├─ POST /prediction
│  │  ├─ Input: ExplanationRequest (prediction_id, include_llm)
│  │  └─ Output: Top features (SHAP), narrative
│  └─ POST /narrative
│     ├─ Input: PredictionNarrativeRequest
│     └─ Output: LLM-generated explanation, insights, next steps
│
├─ Route: RAG (/api/v1/rag/*)
│  ├─ POST /query
│  │  ├─ Input: Query text, type (policy|claims), top_k
│  │  └─ Output: Ranked results with similarity scores
│  └─ POST /recommendations
│     ├─ Input: PolicyId, context
│     └─ Output: Personalized recommendations + evidence
│
├─ Route: Model Management (/api/v1/models/*)
│  ├─ GET /info
│  │  └─ Output: List[ModelInfo] (name, version, accuracy, AUC)
│  ├─ POST /retrain
│  │  ├─ Input: ModelName, date_range
│  │  └─ Output: Task ID, status, ETA
│  └─ GET /drift_check
│     └─ Output: Drift detected, score, affected features
│
└─ Dependencies (dependency_injection.py)
   ├─ Lazy loading of models
   ├─ Model instance caching
   ├─ Error handling for missing services

## LAYER 4: MONITORING & OBSERVABILITY
├─ MLflow (Experiment Tracking)
│  ├─ Runs: Parameters, metrics, artifacts
│  ├─ Model Registry: Versioning, staging, production
│  ├─ Backend: MySQL (persistent)
│  └─ UI: http://localhost:5000
│
├─ Prometheus (Metrics Collection)
│  ├─ Scrape Targets: API, Ollama, MySQL, MLflow
│  ├─ Metrics:
│  │  ├─ api_requests_total (count, latency)
│  │  ├─ model_predictions_counter
│  │  ├─ data_drift_score
│  │  └─ custom business metrics
│  └─ Retention: 15 days default
│
├─ Grafana (Dashboards)
│  ├─ Model Performance: AUC, accuracy, precision, recall
│  ├─ API Health: Latency, error rate, throughput
│  ├─ Data Drift: Feature distributions vs baseline
│  ├─ Database: Query performance, connection pool
│  └─ UI: http://localhost:3000 (admin/admin)
│
└─ Drift Detection (Evidently)
   ├─ Feature Distribution: KS-test (p-value threshold 0.05)
   ├─ Target Distribution: Chi-square for categoricals
   ├─ Triggers: Retraining if drift detected
   └─ Alerting: Slack/Email notifications


# CONFIGURATION HIERARCHY

config.yaml (centralized settings)
  ├─ project metadata (name, version, description)
  ├─ data paths & parameters (CSV, target, split ratios)
  ├─ database settings (MySQL connection params)
  ├─ ml.ensemble (XGBoost, LightGBM, NN hyperparams)
  ├─ ml.rag (embedding model, similarity threshold)
  ├─ ml.llm (Ollama config, LoRA fine-tuning params)
  ├─ api (host, port, debug mode, workers)
  └─ monitoring (MLflow, Prometheus, drift detection thresholds)

.env (runtime secrets - NOT in git)
  ├─ MySQL credentials
  ├─ Ollama host
  ├─ MLflow tracking URI
  ├─ API port & debug mode
  └─ Directory paths

requirements.txt (Python dependencies - 60+ packages)
  ├─ Core ML: pandas, numpy, scikit-learn
  ├─ Models: xgboost, lightgbm, tensorflow, transformers
  ├─ Vector DB: chromadb, sentence-transformers
  ├─ API: fastapi, uvicorn, pydantic
  ├─ Database: mysql-connector-python, sqlalchemy
  ├─ Orchestration: apache-airflow
  ├─ Monitoring: mlflow, prometheus-client
  └─ Dev: pytest, black, flake8, mypy


# DEPLOYMENT STACK

Services in docker-compose.yml:
1. mysql:8.0               (Database - port 3306)
2. insurance-api           (FastAPI - port 8000)
3. ollama:latest           (LLM - port 11434)
4. mlflow:latest           (Experiment Tracking - port 5000)
5. prometheus:latest       (Metrics - port 9090)
6. grafana:latest          (Dashboards - port 3000)

Volumes:
- mysql_data               (MySQL persistence)
- ollama_data              (Model cache)
- mlflow_artifacts         (Experiment artifacts)
- prometheus_data          (Metrics storage)
- grafana_data             (Dashboard configs)

Networks:
- insurance_network        (Inter-service communication)


# CONTINUOUS INTEGRATION & DEPLOYMENT

GitHub Actions Workflows:
1. test.yml
   ├─ Trigger: Push/PR to main/develop
   ├─ Matrix: Python 3.9, 3.10, 3.11
   ├─ Steps: Lint (flake8), format (black), type check (mypy)
   ├─ Tests: pytest with coverage reporting
   └─ Gate: Coverage ≥80%

2. model_validation.yml
   ├─ Trigger: Push to main (ML files changed)
   ├─ MySQL: Service container for testing
   ├─ Steps: Init DB → Load data → Train → Validate AUC
   └─ Gate: AUC ≥0.75 (blocks merge if failed)

3. deploy.yml
   ├─ Trigger: Version tags (v*.*)
   ├─ Steps: Docker build → push to registry → deploy staging
   ├─ Health checks: Wait for API readiness
   ├─ Smoke tests: Sample predictions + explanations
   └─ Notifications: Success/failure alerts


# TESTING COVERAGE

Unit Tests (pytest):
1. test_api.py (11 tests)
   ├─ Health check endpoint
   ├─ Prediction endpoints (valid/invalid inputs)
   ├─ Explanation generation
   ├─ RAG queries
   ├─ Model management endpoints
   └─ Error handling & validation

2. test_models.py (5+ tests)
   ├─ Model initialization
   ├─ Train/predict workflow
   ├─ Explanation generation
   ├─ Feature engineering
   └─ Preprocessing pipeline

Coverage Target: ≥80% of ml/ and api/ modules


# QUICK REFERENCE

## Commands
setup.sh / setup.bat         Auto-setup with venv + deps
python data/scripts/init_db.py     Initialize MySQL schema
python data/scripts/load_raw_data.py   Load 105K rows from CSV
python ml/train_pipeline.py     Train ensemble with MLflow logging
uvicorn api.main:app --reload   Start API (dev mode)
pytest tests/ -v                Run all tests
docker-compose up -d            Start full stack

## Endpoints (when API running)
http://localhost:8000           API root
http://localhost:8000/docs      Swagger UI (interactive testing)
http://localhost:5000           MLflow tracking server
http://localhost:9090           Prometheus metrics
http://localhost:3000           Grafana dashboards

## Performance Targets
- API latency: <500ms for predictions
- Model AUC: ≥0.75
- Code coverage: ≥80%
- Test execution: <60 seconds
- Data drift threshold: KS-test p-value > 0.05


# STATUS INDICATORS

✅ Complete & Ready:
  - Project structure & configuration
  - Data layer (MySQL schema + ETL)
  - ML models (ensemble, RAG, LLM interface)
  - API layer (6 route modules)
  - Docker containerization
  - Testing suite
  - CI/CD workflows
  - Documentation

🔄 In Progress:
  - Monitoring integration (Prometheus → API metrics)
  - MLflow integration in training pipeline
  - Grafana dashboard creation

⏳ Planned:
  - Airflow DAG for data orchestration
  - Feature store pipeline
  - Drift detection rules & alerting
  - Production Kubernetes deployment

---

Generated: 2024-12-15
Project Status: MVP Ready (foundation complete, integration pending)
