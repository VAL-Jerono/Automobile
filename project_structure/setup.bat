"""
Windows batch script for setup.
Usage: setup.bat
"""

@echo off
echo ==================================
echo Insurance Risk Platform Setup
echo ==================================

REM Check Python version
echo.
echo [1/6] Checking Python version...
python --version
python -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)"
if errorlevel 1 (
    echo Error: Python 3.9+ required
    exit /b 1
)
echo OK - Python 3.9+ detected

REM Create virtual environment
echo.
echo [2/6] Creating virtual environment...
if not exist "venv" (
    python -m venv venv
    echo OK - Virtual environment created
) else (
    echo OK - Virtual environment already exists
)

REM Activate venv
call venv\Scripts\activate.bat

REM Install dependencies
echo.
echo [3/6] Installing dependencies...
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
echo OK - Dependencies installed

REM Create environment file
echo.
echo [4/6] Setting up environment variables...
if not exist ".env" (
    copy .env.example .env
    echo WARNING - Created .env file - please edit with your database credentials
) else (
    echo OK - .env file exists
)

REM Create required directories
echo.
echo [5/6] Creating directories...
if not exist "models" mkdir models
if not exist "logs" mkdir logs
if not exist "data" mkdir data
if not exist "vector_db" mkdir vector_db
echo OK - Directories created

REM Setup complete
echo.
echo [6/6] Setup complete!
echo OK - All systems ready

echo.
echo ==================================
echo Next Steps:
echo ==================================
echo.
echo 1. Edit .env file with database credentials:
echo    notepad .env
echo.
echo 2. Virtual environment is active. To activate in new terminal:
echo    venv\Scripts\activate.bat
echo.
echo 3. Initialize database:
echo    python data\scripts\init_db.py
echo.
echo 4. Load data:
echo    python data\scripts\load_raw_data.py
echo.
echo 5. Train model:
echo    python ml\train_pipeline.py
echo.
echo 6. Start API (development):
echo    uvicorn api.main:app --reload
echo.
echo 7. Or use Docker Compose (all services):
echo    docker-compose -f docker/docker-compose.yml up -d
echo.
echo API will be available at: http://localhost:8000
echo Swagger UI: http://localhost:8000/docs
echo.
echo ==================================
pause
