SQL-BASED DEPLOYMENT GUIDE
==========================

## Overview

This guide explains how to deploy the Insurance Analytics Platform using SQL (MySQL) instead of CSV files. This approach:

✅ **Smaller Repository**: No large CSV or FAISS files  
✅ **Production-Ready**: Real database backend  
✅ **Scalable**: Can handle growing data  
✅ **GitHub-Compatible**: Push only code, not data  

---

## Architecture

### Before (CSV-based - Not GitHub-friendly):
```
app.py → load CSV file → FAISS index → predictions → UI
(Large files: 100MB+ CSV + FAISS)
```

### After (SQL-based - GitHub-friendly):
```
app.py → SQL query → MySQL database → predictions → UI  
(Only code in repo; data in separate MySQL instance)
```

---

## Step 1: Setup MySQL Database

### Option A: Docker (Recommended)

```bash
# Run MySQL in Docker
docker run --name insurance-db \
  -e MYSQL_ROOT_PASSWORD=password123 \
  -e MYSQL_DATABASE=insurance \
  -p 3306:3306 \
  -d mysql:8.0

# Wait for MySQL to start (10-15 seconds)
sleep 15

# Verify connection
mysql -h 127.0.0.1 -u root -ppassword123 -e "SHOW DATABASES;"
```

### Option B: Local Installation

```bash
# macOS (using Homebrew)
brew install mysql
brew services start mysql
mysql_secure_installation

# Linux (Ubuntu/Debian)
sudo apt-get install mysql-server
sudo mysql_secure_installation

# Windows
# Download from https://dev.mysql.com/downloads/mysql/
# Run installer
```

### Option C: Cloud (AWS RDS, Google Cloud SQL, etc.)

Use your cloud provider's managed MySQL service. Update environment variables:

```bash
export MYSQL_HOST=your-rds-endpoint.rds.amazonaws.com
export MYSQL_PORT=3306
export MYSQL_USER=admin
export MYSQL_PASSWORD=your-password
export MYSQL_DATABASE=insurance
```

---

## Step 2: Initialize Database

### 1. Locate Your CSV File

The script expects the original data file. Find it:

```bash
# Usually in the root directory
ls -lh Motor_vehicle_insurance_data.csv

# Or in project_structure
ls -lh project_structure/Motor_vehicle_insurance_data.csv
```

### 2. Run Initialization Script

```bash
# Set environment variables (if needed)
export MYSQL_HOST=localhost
export MYSQL_PORT=3306
export MYSQL_USER=root
export MYSQL_PASSWORD=password123
export MYSQL_DATABASE=insurance

# Run the initialization script (one-time operation)
cd Automobile/project_structure

python sql_init.py --csv-path /path/to/Motor_vehicle_insurance_data.csv

# Expected output:
# ✅ Connected to MySQL at localhost:3306
# ✅ Database 'insurance' ready
# ✅ Created 'customers' table
# ✅ Created 'vehicles' table
# ✅ Created 'policies' table with indexes
# ✅ Created 'claims' table with indexes
# ✅ Created 'model_predictions' table
# ✅ Inserted 53,502 customers
# ✅ Inserted 105,555 vehicles
# ✅ Inserted 105,555 policies
# ✅ Data loaded successfully!
```

This script:
- ✅ Creates the database schema
- ✅ Transforms CSV to normalized tables
- ✅ Creates indexes for performance
- ✅ Takes ~2-5 minutes for 105K policies

**Note**: Run this **only once**. After this, the CSV is in MySQL.

---

## Step 3: Update Streamlit App

The app now uses SQL instead of CSV:

```python
# Old way (❌ NOT used anymore)
df = pd.read_csv('Motor_vehicle_insurance_data.csv')

# New way (✅ SQL-based)
from sql_data_manager import SQLDataManager

db = SQLDataManager()
db.connect()
df = db.load_all_policies()
```

The updated `app.py` uses `sql_data_manager.py` for all data access.

---

## Step 4: Install Dependencies

```bash
pip install -r requirements.txt

# Key packages:
# - streamlit
# - mysql-connector-python  (for SQL)
# - pandas
# - scikit-learn
# - plotly
```

Check `requirements.txt` includes:
```
mysql-connector-python>=8.0.33
```

---

## Step 5: Configure Environment

Create `.env` file in project root:

```bash
# MySQL Configuration
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=password123
MYSQL_DATABASE=insurance

# App Configuration
STREAMLIT_PORT=8501
LOG_LEVEL=INFO
```

Or set as environment variables:

```bash
export MYSQL_HOST=localhost
export MYSQL_USER=root
export MYSQL_PASSWORD=password123
```

---

## Step 6: Run the Application

```bash
# Start Streamlit app
streamlit run app.py

# Expected output:
# ✅ Connected to MySQL at localhost:3306
# ✅ Loaded 105,555 policies from MySQL
# You can now view your app in your browser at:
#   Local URL: http://localhost:8501
```

---

## GitHub Deployment Workflow

### What to Push (✅ Small files):
```
Automobile/
├── app.py                          (3 KB)
├── requirements.txt                (1 KB)
├── .gitignore                      (1 KB)
├── project_structure/
│   ├── sql_init.py                (15 KB)  ← Database init script
│   ├── sql_data_manager.py         (12 KB) ← Data access layer
│   ├── api/
│   │   └── database.py             (8 KB)  ← API database queries
│   ├── config.yaml                 (2 KB)
│   └── README.md                   (5 KB)
└── .env.example                    (1 KB)
```

**Total: ~50 KB** ✅ Easy to push to GitHub

### What NOT to Push (❌ Large files):
```
❌ Motor_vehicle_insurance_data.csv  (50+ MB)
❌ enhanced_faiss_index/             (100+ MB)
❌ vector_db/                        (50+ MB)
❌ models/*.pkl                      (optional, can cache)
❌ mlruns/                           (optional, experiment tracking)
```

These are already in `.gitignore` - good!

---

## Production Deployment

### On Your Server/Cloud Instance

```bash
# 1. Clone repository
git clone https://github.com/VAL-Jerono/Automobile.git
cd Automobile

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set environment variables
export MYSQL_HOST=your-db-host.com
export MYSQL_USER=admin
export MYSQL_PASSWORD=your-secure-password

# 4. Initialize database (one-time)
# Copy Motor_vehicle_insurance_data.csv to server, then:
python project_structure/sql_init.py --csv-path Motor_vehicle_insurance_data.csv

# 5. Start app
streamlit run app.py --server.port 8501
```

### Using Docker

```dockerfile
# Dockerfile
FROM python:3.10

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["streamlit", "run", "app.py", "--server.port", "8501"]
```

```bash
# Build
docker build -t insurance-app .

# Run
docker run \
  -e MYSQL_HOST=host.docker.internal \
  -e MYSQL_USER=root \
  -e MYSQL_PASSWORD=password123 \
  -p 8501:8501 \
  insurance-app
```

---

## Data Refresh Workflow

### When You Have New Data

```bash
# 1. Get new CSV file
cp /path/to/new/Motor_vehicle_insurance_data.csv .

# 2. (Optional) Backup current database
mysqldump -u root -p insurance > insurance_backup_$(date +%Y%m%d).sql

# 3. Re-run initialization (deletes old data, loads new)
python project_structure/sql_init.py --csv-path Motor_vehicle_insurance_data.csv

# 4. Restart app
# (No changes needed - app queries the updated database)
```

---

## Monitoring & Performance

### Check Database Health

```bash
# Connect to MySQL
mysql -u root -p

# In MySQL shell:
USE insurance;

# See table sizes
SELECT 
    table_name,
    ROUND(((data_length + index_length) / 1024 / 1024), 2) AS size_mb
FROM information_schema.TABLES
WHERE table_schema = 'insurance'
ORDER BY size_mb DESC;

# Check policy count
SELECT COUNT(*) FROM policies;

# Check indexes are working
EXPLAIN SELECT * FROM policies WHERE lapse = 1 LIMIT 10;
```

### View Query Performance

```bash
# Enable slow query log
SET GLOBAL slow_query_log = 'ON';
SET GLOBAL long_query_time = 1;  # Log queries > 1 second

# Check slow queries
tail -f /var/log/mysql/slow.log
```

---

## Troubleshooting

### Issue: "MySQL connection error"

```bash
# Check MySQL is running
mysql -u root -p -e "SELECT 1;"

# If not running, start it:
# Docker: docker start insurance-db
# Local: mysql.server start (macOS) or sudo service mysql start (Linux)

# Check credentials
export MYSQL_HOST=localhost
export MYSQL_USER=root
export MYSQL_PASSWORD=your_password
```

### Issue: "Database 'insurance' doesn't exist"

```bash
# Run initialization script
python project_structure/sql_init.py --csv-path Motor_vehicle_insurance_data.csv
```

### Issue: "CSV file not found"

```bash
# Locate CSV file
find / -name "Motor_vehicle_insurance_data.csv" 2>/dev/null

# Copy to project directory
cp /path/to/Motor_vehicle_insurance_data.csv .

# Run init script with correct path
python project_structure/sql_init.py --csv-path ./Motor_vehicle_insurance_data.csv
```

### Issue: "Slow queries"

```bash
# Add missing indexes
mysql -u root -p insurance < project_structure/add_indexes.sql

# Or rebuild in MySQL shell:
ALTER TABLE policies ADD INDEX idx_lapse (lapse);
ALTER TABLE policies ADD INDEX idx_renewal (date_next_renewal);
ALTER TABLE policies ADD INDEX idx_premium (premium);
```

---

## API Integration

The FastAPI backend also uses SQL:

```python
# project_structure/api/main.py uses database.py
# which queries MySQL directly

# Example:
GET /api/v1/portfolio/summary
→ database.get_portfolio_summary()
→ SQL query → JSON response

GET /api/v1/customer/{id}/profile  
→ database.search_policy(id)
→ SQL query + predictions → JSON response
```

---

## Benefits Summary

| Aspect | CSV-Based | SQL-Based |
|--------|-----------|-----------|
| **Repository Size** | 100+ MB | < 1 MB |
| **GitHub Push** | ❌ Slow, fails | ✅ Fast |
| **Data Updates** | Replace file | SQL UPDATE |
| **Scalability** | ~100K records max | Millions possible |
| **Performance** | Load entire file | Query only needed data |
| **Production** | Not suitable | Enterprise-ready |
| **Multi-user** | File locks | Concurrent access |
| **Backup/Recovery** | Manual | Native DB tools |

---

## Next Steps

1. ✅ Setup MySQL (Docker or local)
2. ✅ Run `sql_init.py` to load data
3. ✅ Update `app.py` to use SQL (already done)
4. ✅ Test locally with `streamlit run app.py`
5. ✅ Push to GitHub (only code, not CSV!)
6. ✅ Deploy to production server/cloud
7. ✅ Monitor database performance

---

## Support

For issues or questions:
1. Check "Troubleshooting" section above
2. Review logs: `streamlit run app.py 2>&1 | tail -100`
3. Test MySQL: `mysql -u root -p insurance -e "SELECT COUNT(*) FROM policies;"`
4. Check environment variables: `env | grep MYSQL`

---

**Last Updated**: January 12, 2026  
**Version**: 2.0 (SQL-Based)  
**Author**: Insurance Analytics Team
