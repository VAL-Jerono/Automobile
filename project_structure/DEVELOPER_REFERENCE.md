"""
DEVELOPER QUICK REFERENCE
Common commands, code snippets, and troubleshooting
"""

## COMMON COMMANDS

### Setup & Installation

```bash
# Clone repository
git clone <repo> && cd automobile_claims/project_structure

# Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install development tools
pip install pytest black flake8 mypy jupyter

# Copy environment template
cp .env.example .env
# Edit .env with your database credentials
```

### Database Management

```bash
# Initialize database (creates schema)
python data/scripts/init_db.py

# Load data from CSV
python data/scripts/load_raw_data.py

# Access MySQL directly
mysql -h localhost -u insurance_user -p insurance_db
  # Password: from .env MYSQL_PASSWORD
  
  # Common MySQL queries
  SELECT COUNT(*) FROM policies;
  SELECT COUNT(*) FROM claims;
  SELECT lapse, COUNT(*) FROM policies GROUP BY lapse;
```

### Model Training

```bash
# Train ensemble model with MLflow logging
python ml/train_pipeline.py

# View MLflow experiments
mlflow ui  # http://localhost:5000

# Load trained model in Python
from ml.models.ensemble import InsuranceEnsembleModel
model = InsuranceEnsembleModel()
model.load('models/ensemble_model_20241215_143020')
predictions = model.predict(X_test)
```

### API Development

```bash
# Start API locally (with auto-reload)
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# API docs
# Swagger UI: http://localhost:8000/docs
# ReDoc:      http://localhost:8000/redoc

# Test endpoint
curl -X POST http://localhost:8000/api/v1/predict/lapse \
  -H "Content-Type: application/json" \
  -d '{
    "policy_id": 123,
    "age": 45,
    "vehicle_age": 3,
    "premium": 250.0,
    "claims_history": 1,
    "second_driver": 0,
    "type_fuel": "P"
  }'
```

### Docker & Containers

```bash
# Build Docker image
docker build -f docker/Dockerfile.api -t insurance-api:latest .

# Run API container
docker run -p 8000:8000 \
  -e MYSQL_HOST=host.docker.internal \
  -e OLLAMA_HOST=http://host.docker.internal:11434 \
  insurance-api:latest

# Use docker-compose (all services)
docker-compose -f docker/docker-compose.yml up -d
docker-compose -f docker/docker-compose.yml logs -f api
docker-compose -f docker/docker-compose.yml down

# Access services in compose
# MySQL: localhost:3306 (from host) or mysql:3306 (from containers)
# API: localhost:8000
# MLflow: localhost:5000
# Prometheus: localhost:9090
# Grafana: localhost:3000
```

### Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_api.py::TestPredictionEndpoints::test_predict_lapse_valid -v

# Run with coverage
pytest tests/ --cov=ml --cov=api --cov-report=html
# View coverage report: open htmlcov/index.html

# Run in watch mode (requires pytest-watch)
ptw tests/

# Test API directly
pytest tests/test_api.py -v -s  # -s shows print statements
```

### Code Quality

```bash
# Format code with black
black . --line-length=100

# Check formatting without changing
black . --check --line-length=100

# Lint with flake8
flake8 ml/ api/ data/ --max-line-length=100

# Type checking with mypy
mypy ml/ api/ --ignore-missing-imports

# Sort imports with isort
isort ml/ api/ data/
```

### Git Workflow

```bash
# Create feature branch
git checkout -b feature/new-endpoint

# Commit changes
git add .
git commit -m "Add new prediction endpoint"

# Push to remote
git push origin feature/new-endpoint

# Create pull request (on GitHub)
# → Triggers test.yml workflow
# → Must pass tests to merge

# Merge after PR approval
git checkout main
git pull
git merge feature/new-endpoint
git push

# Tag for release/deployment
git tag -a v1.0.0 -m "Release version 1.0.0"
git push origin v1.0.0
# → Triggers deploy.yml workflow
```

---

## CODE SNIPPETS

### Loading & Training a Model

```python
import pandas as pd
from ml.models.ensemble import InsuranceEnsembleModel
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('Motor vehicle insurance data.csv', sep=';', encoding='utf-8')
X = df.drop(['ID', 'Lapse'], axis=1)
y = df['Lapse'].astype(int)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train
model = InsuranceEnsembleModel()
model.train(X_train, y_train)

# Predict
probabilities = model.predict(X_test)
explanations = model.explain(X_test, instance_idx=0)

# Save
model.save('models/my_model')

# Load
model.load('models/my_model')
```

### Using RAG System

```python
from ml.rag.retrieval import RAGEngine
import pandas as pd

# Initialize RAG
rag = RAGEngine(
    embedding_model='sentence-transformers/all-MiniLM-L6-v2',
    vector_db_path='./vector_db'
)

# Index policies from dataframe
df_policies = pd.read_csv('policies.csv')
rag.index_policies(df_policies)

# Query
results = rag.query_policies(
    "high-premium vehicle policies with claims history",
    top_k=5
)

for result in results:
    print(f"Rank: {result['rank']}")
    print(f"Similarity: {result['similarity']:.3f}")
    print(f"Policy ID: {result.get('policy_id')}")
    print(f"Document: {result['document'][:100]}...")
```

### Generating LLM Explanations

```python
from ml.models.llm_fine_tune import OllamaFineTuner

# Initialize
llm = OllamaFineTuner(
    base_model='llama2',
    ollama_host='http://localhost:11434'
)

# Check if model is available
if llm.check_model_availability():
    # Generate explanation
    explanation = llm.generate_claim_explanation(
        policy_id=123,
        claim_reason='approved'
    )
    print(explanation)
    
    # Generate recommendation
    recommendation = llm.generate_policy_recommendation(
        customer_profile={'age': 45, 'vehicle_age': 3, 'claims': 1}
    )
    print(recommendation)
else:
    print("Ollama service not available")
```

### Creating an API Endpoint

```python
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

class RequestSchema(BaseModel):
    param1: str
    param2: float

class ResponseSchema(BaseModel):
    result: str
    confidence: float

@router.post("/my_endpoint", response_model=ResponseSchema)
async def my_endpoint(req: RequestSchema):
    """
    Description of what this endpoint does.
    """
    try:
        # Your logic here
        result = f"Processed {req.param1}"
        confidence = 0.95
        
        return ResponseSchema(
            result=result,
            confidence=confidence
        )
    
    except Exception as e:
        logger.error(f"Error in my_endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# In api/main.py, include the router:
# app.include_router(router, prefix="/api/v1/custom", tags=["custom"])
```

### MLflow Experiment Tracking

```python
import mlflow
from ml.models.ensemble import InsuranceEnsembleModel

# Setup MLflow
mlflow.set_tracking_uri('http://localhost:5000')
mlflow.set_experiment('insurance_models')

# Start a run
with mlflow.start_run():
    # Log parameters
    mlflow.log_params({
        'model_type': 'ensemble',
        'n_estimators': 100,
        'learning_rate': 0.1
    })
    
    # Log metrics
    mlflow.log_metrics({
        'accuracy': 0.92,
        'precision': 0.88,
        'recall': 0.90,
        'f1_score': 0.89,
        'roc_auc': 0.95
    })
    
    # Log model
    model = InsuranceEnsembleModel()
    model.train(X_train, y_train)
    mlflow.sklearn.log_model(model.estimators_['xgb'], 'xgb_model')
    
    # Log artifacts
    mlflow.log_artifact('models/model.pkl')
    
# View at http://localhost:5000
```

---

## TROUBLESHOOTING

### MySQL Connection Issues

```
Error: Access denied for user 'insurance_user'@'localhost'
→ Check .env file MYSQL_PASSWORD is correct
→ Verify MySQL is running: mysql -u root -p

Error: Can't connect to MySQL server on 'localhost'
→ Start MySQL: 
  # macOS: brew services start mysql
  # Docker: docker run -d -p 3306:3306 -e MYSQL_ROOT_PASSWORD=root mysql:8.0
```

### Model Training Hangs

```
Problem: train_pipeline.py freezes during cross-validation
→ Set CV folds lower in config.yaml (cross_validation_folds: 3)
→ Use smaller dataset for testing
→ Check memory: top / Activity Monitor
→ Increase timeout in subprocess calls
```

### API Won't Start

```
Error: "Address already in use" on port 8000
→ Kill process: lsof -ti:8000 | xargs kill -9
→ Or use different port: uvicorn api.main:app --port 8001

Error: ModuleNotFoundError: No module named 'api'
→ Ensure you're in project root: /path/to/automobile_claims/project_structure
→ Check PYTHONPATH: export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Docker Compose Issues

```
Error: Port 3306 already in use
→ Stop MySQL: brew services stop mysql
→ Or use different port in docker-compose.yml

Error: Service 'mysql' failed to start
→ Check logs: docker-compose logs mysql
→ Ensure MySQL image is pulled: docker pull mysql:8.0

Error: Permission denied /var/lib/mysql
→ Fix permissions: sudo chown 999:999 /path/to/mysql_data
→ Or use named volumes (already in docker-compose.yml)
```

### RAG Retrieval Slow

```
Problem: query_policies() takes >5 seconds
→ Check ChromaDB is in-memory or persistent?
→ Reduce top_k parameter
→ Pre-index policies before deployment
→ Use vector DB with GPU support (if available)

Problem: No results returned
→ Lower similarity_threshold in config.yaml (default 0.7)
→ Verify policies were indexed: rag.index_policies(df)
→ Check query text matches indexed documents
```

### Tests Failing

```
Error: pytest: command not found
→ Install: pip install pytest pytest-cov

Error: ModuleNotFoundError in test files
→ Install test dependencies: pip install -e .
→ Or run from project root: pytest tests/

FAILED tests/test_api.py - AssertionError: assert response.status_code == 200
→ Check TestClient is using correct app
→ Verify dependencies (models, RAG) are initialized
→ Check request payload matches schema
```

---

## PERFORMANCE OPTIMIZATION TIPS

### Model Training Speed

```python
# Use subset for testing
df_sample = df.sample(n=10000, random_state=42)  # Instead of all 105K

# Reduce cross-validation folds
config['ml']['validation']['cross_validation_folds'] = 3  # Instead of 5

# Disable SHAP explanations during training
model.train(X, y, compute_shap=False)

# Use parallel processing
import joblib
with joblib.parallel_backend('threading', n_jobs=-1):
    model.train(X, y)
```

### API Response Time

```python
# Cache model predictions for repeated queries
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_prediction(policy_id: int):
    return model.predict([policy_id])

# Use connection pooling for database
from mysql.connector import pooling
pool = pooling.MySQLConnectionPool(pool_name='pool', pool_size=5)

# Batch predictions instead of single
predictions = model.predict(X_batch)  # Not X_single in loop
```

### Memory Usage

```python
# Stream large CSV instead of loading all
for chunk in pd.read_csv('large_file.csv', chunksize=10000):
    process(chunk)

# Delete unused dataframes
del df_temp
import gc
gc.collect()

# Profile memory usage
from memory_profiler import profile

@profile
def expensive_function():
    ...
```

---

## USEFUL LINKS & RESOURCES

- **FastAPI Docs:** https://fastapi.tiangolo.com
- **MLflow Docs:** https://mlflow.org/docs
- **ChromaDB:** https://docs.trychroma.com
- **Ollama:** https://ollama.ai
- **Prometheus:** https://prometheus.io/docs
- **Grafana:** https://grafana.com/docs
- **Docker Compose:** https://docs.docker.com/compose/
- **GitHub Actions:** https://docs.github.com/en/actions

---

## CONTACT & SUPPORT

**Questions?**
- Check README.md for architecture overview
- Review IMPLEMENTATION_SUMMARY.md for feature status
- Look at code docstrings for detailed documentation
- Check GitHub Issues for known problems

**Contributing:**
- Fork repository
- Create feature branch
- Ensure tests pass (pytest tests/ -v)
- Code quality checks (black, flake8, mypy)
- Submit PR with description

---

Last Updated: 2024-12-15
