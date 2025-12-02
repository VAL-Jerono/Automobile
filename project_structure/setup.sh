#!/bin/bash
# Setup script for insurance platform
# Usage: chmod +x setup.sh && ./setup.sh

set -e  # Exit on error

echo "=================================="
echo "Insurance Risk Platform Setup"
echo "=================================="

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 1. Check Python version
echo -e "\n${BLUE}[1/6] Checking Python version...${NC}"
python --version
if python -c 'import sys; exit(0 if sys.version_info >= (3, 9) else 1)'; then
    echo -e "${GREEN}✓ Python 3.9+ detected${NC}"
else
    echo -e "${YELLOW}⚠ Python 3.9+ required${NC}"
    exit 1
fi

# 2. Create virtual environment
echo -e "\n${BLUE}[2/6] Creating virtual environment...${NC}"
if [ ! -d "venv" ]; then
    python -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment already exists${NC}"
fi

# Activate venv
source venv/bin/activate

# 3. Install dependencies
echo -e "\n${BLUE}[3/6] Installing dependencies...${NC}"
pip install --upgrade pip setuptools wheel
echo "Installing requirements (this may take 3-5 minutes)..."
pip install -r requirements.txt --no-build-isolation
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Dependencies installed${NC}"
else
    echo -e "${YELLOW}⚠ Some dependencies failed - trying with build isolation disabled${NC}"
    pip install --no-build-isolation -q -r requirements.txt
fi

# 4. Create environment file
echo -e "\n${BLUE}[4/6] Setting up environment variables...${NC}"
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo -e "${YELLOW}⚠ Created .env file - please edit with your database credentials${NC}"
else
    echo -e "${GREEN}✓ .env file exists${NC}"
fi

# 5. Create required directories
echo -e "\n${BLUE}[5/6] Creating directories...${NC}"
mkdir -p models logs data vector_db
chmod 755 models logs data vector_db
echo -e "${GREEN}✓ Directories created${NC}"

# 6. Setup complete
echo -e "\n${BLUE}[6/6] Setup complete!${NC}"
echo -e "${GREEN}✓ All systems ready${NC}"

echo ""
echo "=================================="
echo "Next Steps:"
echo "=================================="
echo ""
echo "1. Edit .env file with database credentials:"
echo "   nano .env"
echo ""
echo "2. Activate virtual environment (if not already active):"
echo "   source venv/bin/activate"
echo ""
echo "3. Initialize database:"
echo "   python data/scripts/init_db.py"
echo ""
echo "4. Load data:"
echo "   python data/scripts/load_raw_data.py"
echo ""
echo "5. Train model:"
echo "   python ml/train_pipeline.py"
echo ""
echo "6. Start API (development):"
echo "   uvicorn api.main:app --reload"
echo ""
echo "7. Or use Docker Compose (all services):"
echo "   docker-compose -f docker/docker-compose.yml up -d"
echo ""
echo "API will be available at: http://localhost:8000"
echo "Swagger UI: http://localhost:8000/docs"
echo ""
echo "=================================="
