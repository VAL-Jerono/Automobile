#!/bin/bash
# Quick Start: Generate Predictions and Deploy
# ==============================================
# This script handles the complete pipeline from notebook to production

set -e  # Exit on error

echo "🚀 Insurance Analytics Deployment Pipeline"
echo "==========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Check prerequisites
echo -e "\n${YELLOW}Step 1: Checking prerequisites...${NC}"

if ! command -v mysql &> /dev/null; then
    echo -e "${RED}❌ MySQL not found. Install with: brew install mysql${NC}"
    exit 1
fi

if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 not found${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Prerequisites met${NC}"

# Step 2: Start MySQL
echo -e "\n${YELLOW}Step 2: Starting MySQL...${NC}"

if mysql.server status > /dev/null 2>&1; then
    echo -e "${GREEN}✅ MySQL is already running${NC}"
else
    echo "Starting MySQL..."
    mysql.server start
    sleep 2
    echo -e "${GREEN}✅ MySQL started${NC}"
fi

# Step 3: Navigate to project directory
cd "$(dirname "$0")"

echo -e "\n${YELLOW}Step 3: Installing Python dependencies...${NC}"
pip install -r requirements.txt > /dev/null 2>&1
pip install -r project_structure/requirements-docker.txt > /dev/null 2>&1
echo -e "${GREEN}✅ Dependencies installed${NC}"

# Step 4: Initialize database
echo -e "\n${YELLOW}Step 4: Initializing database schema...${NC}"

# Check if database already initialized
if mysql -u root insurance -e "SELECT COUNT(*) FROM model_predictions" > /dev/null 2>&1; then
    echo "Database already initialized. Skipping..."
    echo -e "${GREEN}✅ Database schema ready${NC}"
else
    echo "Initializing fresh database..."
    
    # Look for insurance data
    if [ -f "../Motor vehicle insurance data.csv" ]; then
        CSV_PATH="../Motor vehicle insurance data.csv"
    elif [ -f "Motor vehicle insurance data.csv" ]; then
        CSV_PATH="Motor vehicle insurance data.csv"
    elif [ -f "Motor_vehicle_insurance_data.csv" ]; then
        CSV_PATH="Motor_vehicle_insurance_data.csv"
    else
        echo -e "${RED}❌ Could not find insurance data CSV${NC}"
        echo "   Expected: Motor vehicle insurance data.csv or Motor_vehicle_insurance_data.csv"
        exit 1
    fi
    
    echo "Using CSV: $CSV_PATH"
    cd project_structure
    python sql_init.py --csv-path "$CSV_PATH"
    cd ..
    
    echo -e "${GREEN}✅ Database initialized${NC}"
fi

# Step 5: Generate predictions
echo -e "\n${YELLOW}Step 5: Generating model predictions (this takes 5-10 minutes)...${NC}"
echo "This will:"
echo "  • Run 66 notebook cells to train ML models"
echo "  • Generate predictions for 105,555 customers"
echo "  • Store results in MySQL database"
echo ""

python export_predictions_to_sql.py

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Predictions generated successfully${NC}"
else
    echo -e "${RED}❌ Prediction generation failed${NC}"
    exit 1
fi

# Step 6: Verify data
echo -e "\n${YELLOW}Step 6: Verifying data in database...${NC}"

PRED_COUNT=$(mysql -u root insurance -e "SELECT COUNT(*) FROM model_predictions;" | tail -1)
if [ "$PRED_COUNT" -gt "100000" ]; then
    echo -e "${GREEN}✅ Database verified: $PRED_COUNT predictions${NC}"
else
    echo -e "${RED}❌ Prediction count seems low: $PRED_COUNT${NC}"
fi

# Step 7: Test Streamlit app
echo -e "\n${YELLOW}Step 7: Testing Streamlit app...${NC}"

echo "The app will now open in your browser."
echo "Press Ctrl+C to stop the server."
echo ""

streamlit run app.py

echo -e "\n${GREEN}✅ Deployment complete!${NC}"
echo ""
echo "Next steps:"
echo "  1. Push to GitHub: git add .; git commit -m 'Data pipeline complete'; git push"
echo "  2. Deploy: docker-compose -f project_structure/docker-compose.yml up -d"
echo "  3. Monitor: Check logs with 'docker logs <container_id>'"
