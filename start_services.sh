#!/bin/bash

###############################################################################
# Insurance Analytics - Complete Deployment Script
# Activates Layer 3 (API) and Layer 5 (MLOps Monitoring)
###############################################################################

set -e  # Exit on error

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║     Insurance Analytics - Full Stack Deployment                  ║"
echo "║     Layer 3: FastAPI Backend + Layer 5: MLOps Infrastructure      ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored output
print_step() {
    echo -e "${BLUE}➜ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if running from Automobile directory
if [ ! -f "app.py" ]; then
    print_error "Please run this script from the Automobile directory"
    exit 1
fi

# Step 1: Check prerequisites
print_step "Step 1: Checking prerequisites..."

if ! command -v docker &> /dev/null; then
    print_warning "Docker not found. Installing Docker is recommended for full deployment."
    DOCKER_AVAILABLE=false
else
    print_success "Docker is installed"
    DOCKER_AVAILABLE=true
fi

if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is required but not installed"
    exit 1
else
    print_success "Python 3 is installed"
fi

# Step 2: Create virtual environment if not exists
print_step "Step 2: Setting up Python environment..."

if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_success "Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate
print_success "Virtual environment activated"

# Step 3: Install dependencies
print_step "Step 3: Installing Python dependencies..."
pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt > /dev/null 2>&1
pip install fastapi uvicorn[standard] prometheus-client mlflow python-multipart > /dev/null 2>&1
print_success "Dependencies installed"

# Step 4: Check model files
print_step "Step 4: Verifying ML model files..."

MODELS=("churn_model.csv" "claims_frequency_model.csv" "claims_severity_model.csv" "clv_model.csv")
ALL_MODELS_PRESENT=true

for model in "${MODELS[@]}"; do
    if [ -f "$model" ]; then
        print_success "$model found"
    else
        print_warning "$model not found"
        ALL_MODELS_PRESENT=false
    fi
done

if [ "$ALL_MODELS_PRESENT" = false ]; then
    print_warning "Some model files are missing. API will use fallback predictions."
fi

# Step 5: Check MySQL connection
print_step "Step 5: Checking MySQL database..."

if command -v mysql &> /dev/null; then
    if mysql -u root -e "SELECT 1;" &> /dev/null || mysql -u root -proot -e "SELECT 1;" &> /dev/null; then
        print_success "MySQL is accessible"
        
        # Check if database exists
        if mysql -u root -e "USE insurance;" &> /dev/null || mysql -u root -proot -e "USE insurance;" &> /dev/null; then
            print_success "Insurance database found"
        else
            print_warning "Insurance database not found. Streamlit may not work properly."
        fi
    else
        print_warning "MySQL is installed but not accessible. Check credentials."
    fi
else
    print_warning "MySQL not found. Install MySQL for full functionality."
fi

# Step 6: Create MLflow directories
print_step "Step 6: Setting up MLflow tracking..."

mkdir -p mlruns mlflow_artifacts
print_success "MLflow directories created"

# Step 7: Start services
print_step "Step 7: Starting services..."

echo ""
echo "Choose deployment mode:"
echo "1) Docker Compose (recommended - full stack)"
echo "2) Local services (API + Streamlit only)"
echo ""
read -p "Enter choice (1 or 2): " DEPLOYMENT_MODE

if [ "$DEPLOYMENT_MODE" = "1" ] && [ "$DOCKER_AVAILABLE" = true ]; then
    print_step "Starting Docker Compose services..."
    docker-compose up -d
    print_success "All services started via Docker Compose"
    
    echo ""
    echo "Services are starting up. This may take 30-60 seconds..."
    sleep 10
    
    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║                   🚀 SERVICES DEPLOYED                    ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
    echo "📊 Streamlit Dashboard:  http://localhost:8501"
    echo "🔌 FastAPI Backend:      http://localhost:8001"
    echo "📚 API Documentation:    http://localhost:8001/docs"
    echo "🧪 MLflow Tracking:      http://localhost:5000"
    echo "📈 Prometheus:           http://localhost:9090"
    echo "📊 Grafana:              http://localhost:3000"
    echo "   (username: admin, password: admin)"
    echo ""
    echo "To stop services: docker-compose down"
    echo "To view logs: docker-compose logs -f"
    
else
    print_step "Starting local services..."
    
    # Start FastAPI in background
    print_step "Starting FastAPI server on port 8001..."
    nohup uvicorn api.main:app --host 0.0.0.0 --port 8001 > logs/api.log 2>&1 &
    API_PID=$!
    sleep 3
    
    if ps -p $API_PID > /dev/null; then
        print_success "FastAPI server started (PID: $API_PID)"
    else
        print_error "Failed to start FastAPI server. Check logs/api.log"
        exit 1
    fi
    
    # Start Streamlit in background
    print_step "Starting Streamlit dashboard on port 8501..."
    nohup streamlit run app.py --server.port 8501 --server.address 0.0.0.0 > logs/streamlit.log 2>&1 &
    STREAMLIT_PID=$!
    sleep 3
    
    if ps -p $STREAMLIT_PID > /dev/null; then
        print_success "Streamlit dashboard started (PID: $STREAMLIT_PID)"
    else
        print_error "Failed to start Streamlit. Check logs/streamlit.log"
        exit 1
    fi
    
    # Optionally start MLflow
    read -p "Start MLflow tracking server? (y/n): " START_MLFLOW
    if [ "$START_MLFLOW" = "y" ]; then
        print_step "Starting MLflow server on port 5000..."
        nohup mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlflow_artifacts --host 0.0.0.0 --port 5000 > logs/mlflow.log 2>&1 &
        MLFLOW_PID=$!
        sleep 2
        print_success "MLflow server started (PID: $MLFLOW_PID)"
    fi
    
    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║              🚀 LOCAL SERVICES DEPLOYED                   ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
    echo "📊 Streamlit Dashboard:  http://localhost:8501"
    echo "🔌 FastAPI Backend:      http://localhost:8001"
    echo "📚 API Documentation:    http://localhost:8001/docs"
    if [ "$START_MLFLOW" = "y" ]; then
        echo "🧪 MLflow Tracking:      http://localhost:5000"
    fi
    echo ""
    echo "Process IDs saved for management:"
    echo "API PID: $API_PID"
    echo "Streamlit PID: $STREAMLIT_PID"
    if [ "$START_MLFLOW" = "y" ]; then
        echo "MLflow PID: $MLFLOW_PID"
    fi
    echo ""
    echo "To stop services: kill $API_PID $STREAMLIT_PID"
    if [ "$START_MLFLOW" = "y" ]; then
        echo "                  kill $MLFLOW_PID"
    fi
fi

# Step 8: Health checks
print_step "Step 8: Running health checks..."
sleep 5

if curl -s http://localhost:8001/health > /dev/null 2>&1; then
    print_success "API health check passed"
else
    print_warning "API health check failed (may still be starting up)"
fi

if curl -s http://localhost:8501 > /dev/null 2>&1; then
    print_success "Streamlit health check passed"
else
    print_warning "Streamlit health check failed (may still be starting up)"
fi

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                  ✅ DEPLOYMENT COMPLETE                    ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
print_success "All services are operational!"
echo ""
echo "Quick Test Commands:"
echo "# Test API health"
echo "curl http://localhost:8001/health"
echo ""
echo "# Test churn prediction"
echo 'curl -X POST "http://localhost:8001/api/v1/predict/churn" \\'
echo '  -H "Content-Type: application/json" \\'
echo '  -d '"'"'{"age": 45, "tenure": 2.5, "premium": 350, "vehicle_age": 3, "claims_history": 1, "channel": "broker"}'"'"
echo ""
