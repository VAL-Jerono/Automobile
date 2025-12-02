# Setup Troubleshooting Guide

## Issue 1: Python 3.13 Compatibility Error

**Error Message:**
```
AttributeError: module 'pkgutil' has no attribute 'ImpImporter'
ERROR: Failed to build 'numpy'
```

**Cause:** Python 3.13 removed `pkgutil.ImpImporter`, causing setuptools to fail during numpy build.

**Solution (Recommended):**

### Option A: Use Python 3.9-3.12 (BEST)
```bash
# Check Python version
python --version

# If you have 3.13, use an earlier Python version
# Via Conda:
conda create -n insurance-py311 python=3.11
conda activate insurance-py311

# Or via homebrew:
brew install python@3.11
/usr/local/bin/python3.11 -m venv venv
source venv/bin/activate
```

### Option B: Use Pre-built Wheels (FAST)
```bash
pip install --upgrade pip wheel
pip install -r requirements.txt --only-binary :all: --no-build-isolation
```

### Option C: Docker (EASIEST - No Local Setup)
```bash
# Start all services without local Python
docker-compose -f docker/docker-compose.yml up -d

# API will be available at http://localhost:8000/docs
# (Requires Docker Desktop running)
```

---

## Issue 2: Docker Daemon Not Running

**Error Message:**
```
unable to get image 'mysql:8.0': Cannot connect to the Docker daemon
Is the docker daemon running?
```

**Cause:** Docker Desktop is not running.

**Solution:**

### macOS
```bash
# Start Docker Desktop
open /Applications/Docker.app

# Or start via command line
launchctl start com.docker.vmnetd

# Wait 30 seconds for Docker to fully start
sleep 30

# Verify Docker is running
docker ps
```

### Windows
```cmd
# Start Docker Desktop from Applications
# Or via PowerShell (Admin)
Start-Service Docker

# Wait for Docker to start
sleep 30
docker ps
```

### Linux
```bash
# Start Docker daemon
sudo systemctl start docker
sudo systemctl enable docker

# Add your user to docker group (no sudo needed)
sudo usermod -aG docker $USER
newgrp docker

# Verify
docker ps
```

---

## Issue 3: Port Already in Use

**Error Message:**
```
Error starting userland proxy: listen tcp 0.0.0.0:3306: bind: address already in use
```

**Cause:** MySQL or another service is already running on port 3306.

**Solution:**

### Option A: Kill Existing Service
```bash
# Find process on port 3306
lsof -i :3306

# Kill it
kill -9 <PID>

# Or stop MySQL if running
brew services stop mysql
```

### Option B: Use Different Port
```bash
# Edit docker-compose.yml
sed -i '' 's/"3306:3306"/"3307:3306"/' docker/docker-compose.yml

# Update .env to use new port
echo "MYSQL_PORT=3307" >> .env
```

---

## Issue 4: Virtual Environment Issues

**Error Message:**
```
command not found: venv
ModuleNotFoundError: No module named 'venv'
```

**Cause:** venv module not available (Python may be broken).

**Solution:**

```bash
# Reinstall Python venv
python -m pip install --upgrade pip

# Create virtual environment with conda instead
conda create -n insurance python=3.11
conda activate insurance
pip install -r requirements.txt
```

---

## Issue 5: Missing Dependencies

**Error Message:**
```
ModuleNotFoundError: No module named 'pandas'
```

**Cause:** Dependencies not installed or using wrong Python/venv.

**Solution:**

```bash
# Verify you're in virtual environment
which python  # Should show path with 'venv' in it

# Reinstall dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Or install minimal set first
pip install pandas numpy scikit-learn fastapi uvicorn
```

---

## Issue 6: MySQL Connection Failed

**Error Message:**
```
Access denied for user 'insurance_user'@'localhost'
Can't connect to MySQL server on 'localhost'
```

**Cause:** MySQL not running or wrong credentials.

**Solution:**

```bash
# Check if MySQL is running
ps aux | grep mysql

# Start MySQL
brew services start mysql  # macOS
# or
docker run -d -p 3306:3306 \
  -e MYSQL_ROOT_PASSWORD=root \
  -e MYSQL_DATABASE=insurance_db \
  mysql:8.0

# Check .env file
cat .env | grep MYSQL

# Update .env with correct credentials
nano .env
```

---

## Issue 7: Ollama Not Found

**Error Message:**
```
Connection refused - Ollama service not available
```

**Cause:** Ollama not installed or not running.

**Solution:**

### Option A: Using Docker (Recommended)
```bash
# Docker-compose will handle Ollama
docker-compose -f docker/docker-compose.yml up -d ollama

# Wait for it to start
sleep 10

# Pull model
docker exec insurance_ollama ollama pull llama2
```

### Option B: Local Installation
```bash
# Install Ollama
brew install ollama  # macOS
# or download from https://ollama.ai

# Start Ollama service
ollama serve

# In another terminal, pull model
ollama pull llama2
```

---

## Issue 8: API Won't Start

**Error Message:**
```
Error binding to 0.0.0.0:8000
Address already in use
```

**Cause:** Port 8000 is already in use.

**Solution:**

```bash
# Find process on port 8000
lsof -i :8000

# Kill it
kill -9 <PID>

# Or use different port
uvicorn api.main:app --port 8001
```

---

## Quick Diagnostic Commands

```bash
# Check Python version
python --version

# Check virtual environment
which python

# Check installed packages
pip list | grep -E 'pandas|numpy|fastapi'

# Check Docker status
docker ps
docker-compose -f docker/docker-compose.yml ps

# Check MySQL
mysql -u insurance_user -p insurance_db -e "SELECT 1"

# Check if ports are available
lsof -i :3306  # MySQL
lsof -i :5000  # MLflow
lsof -i :8000  # API
lsof -i :9090  # Prometheus
```

---

## Recommended Setup Path

**For Best Results:**

1. ✅ Use Python 3.11 or 3.12 (NOT 3.13)
2. ✅ Start with Docker if you have it running
3. ✅ Fall back to local setup if Docker unavailable
4. ✅ Use `--no-build-isolation` flag for pip if you get build errors

**Fastest Setup (with Docker):**
```bash
# 1. Start Docker Desktop
open /Applications/Docker.app

# 2. Wait 30 seconds
sleep 30

# 3. Start services
docker-compose -f docker/docker-compose.yml up -d

# 4. Verify
curl http://localhost:8000/docs
```

**Fast Setup (Local - Python 3.11):**
```bash
# 1. Create Python 3.11 environment
conda create -n ins python=3.11
conda activate ins

# 2. Run setup
chmod +x setup.sh
./setup.sh

# 3. Start API
uvicorn api.main:app --reload
```

---

## Getting Help

If you're still stuck:

1. **Check logs:**
   ```bash
   docker-compose -f docker/docker-compose.yml logs -f api
   ```

2. **Check requirements compatibility:**
   ```bash
   pip check
   ```

3. **Try minimal install:**
   ```bash
   pip install pandas numpy scikit-learn fastapi
   ```

4. **Check environment:**
   ```bash
   python -c "import sys; print(sys.version)"
   python -c "import pandas; print(pandas.__version__)"
   ```

---

**Updated:** December 2, 2025
**Status:** All common issues covered

---

## TL;DR - The Quick Fixes

| Issue | Fix |
|-------|-----|
| Python 3.13 error | Use Python 3.11 or 3.12 instead |
| Docker daemon error | Start Docker Desktop / `sudo systemctl start docker` |
| Port already in use | `lsof -i :3306` → `kill -9 <PID>` |
| Missing dependencies | `pip install -r requirements.txt --no-build-isolation` |
| MySQL connection failed | Check `.env`, start MySQL service |
| Ollama not found | Run `docker-compose up -d ollama` or `brew install ollama` |
| API won't start | Kill process: `lsof -i :8000` → `kill -9 <PID>` |
