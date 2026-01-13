# GITHUB-READY DEPLOYMENT
Quick Start for Pushing to GitHub

## Problem Solved ✅

**Before**: Repository was 100+ MB (CSV + FAISS)  
- ❌ Too large for GitHub
- ❌ Can't push without LFS
- ❌ Not production-ready

**After**: Repository is now < 1 MB (only code)  
- ✅ Git-friendly
- ✅ Easy to push/pull
- ✅ Production-ready with SQL
- ✅ Scalable architecture

---

## What Changed

### Files You CAN Push to GitHub (Code)
```
✅ app.py
✅ requirements.txt  
✅ project_structure/sql_init.py (NEW)
✅ project_structure/sql_data_manager.py (NEW)
✅ project_structure/api/database.py
✅ project_structure/config.yaml
✅ SQL_DEPLOYMENT_GUIDE.md (NEW)
✅ .env.example
✅ .gitignore (UPDATED)
```

### Files You CANNOT Push (Too Large)
```
❌ Motor_vehicle_insurance_data.csv (50+ MB)
❌ enhanced_faiss_index/ (100+ MB)
❌ vector_db/ (50+ MB)
```

These are already in .gitignore ✓

---

## Quick Setup (5 Minutes)

### 1. Install MySQL (One-time)
```bash
# Option A: Docker (easiest)
docker run --name insurance-db \
  -e MYSQL_ROOT_PASSWORD=password123 \
  -e MYSQL_DATABASE=insurance \
  -p 3306:3306 \
  -d mysql:8.0

# Option B: Homebrew (macOS)
brew install mysql && brew services start mysql

# Option C: APT (Linux)
sudo apt install mysql-server
```

### 2. Initialize Database (One-time)
```bash
# Ensure you have the CSV file somewhere
# Then run:
python project_structure/sql_init.py \
  --csv-path /path/to/Motor_vehicle_insurance_data.csv

# This creates the database and loads data (~5 min)
```

### 3. Configure Environment
```bash
# Copy example to actual .env
cp .env.example .env

# Edit .env with your database credentials
nano .env

# OR set environment variables
export MYSQL_HOST=localhost
export MYSQL_USER=root
export MYSQL_PASSWORD=password123
```

### 4. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 5. Run the App
```bash
streamlit run app.py

# Open browser to http://localhost:8501
```

**Done!** ✅

---

## Push to GitHub

```bash
# 1. Initialize repo (if new)
git init
git add .
git commit -m "Initial commit: Insurance analytics with SQL backend"

# 2. Add remote
git remote add origin https://github.com/VAL-Jerono/Automobile.git

# 3. Push to GitHub
git push -u origin main

# Verify: Check GitHub - repo should be ~500 KB (not 100+ MB!)
```

---

## Deploy to Production Server

```bash
# 1. On your server, clone the repo
git clone https://github.com/VAL-Jerono/Automobile.git
cd Automobile

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Copy your CSV file to server
# (Via SCP, FTP, or download from your source)
scp Motor_vehicle_insurance_data.csv user@server:/tmp/

# 4. Initialize database (one-time)
export MYSQL_HOST=your-db-server.com
export MYSQL_USER=admin
export MYSQL_PASSWORD=your-password

python project_structure/sql_init.py \
  --csv-path /tmp/Motor_vehicle_insurance_data.csv

# 5. Start app
nohup streamlit run app.py --server.port 8501 &

# Or use Docker:
docker build -t insurance-app .
docker run -p 8501:8501 \
  -e MYSQL_HOST=your-db.com \
  insurance-app
```

---

## Key Points

### ✅ What Works Now
- Small GitHub repo (push/pull in seconds)
- Real SQL database backend
- Scalable to millions of policies
- Production-ready
- Proper credential management (.env)

### ⚠️ Important Notes
1. **Never commit .env** - Credentials stay secure
2. **CSV file is never stored in git** - Data stays separate
3. **Database init is one-time** - After that, just query
4. **No file size limits** - SQL handles large datasets

### 📋 Deployment Checklist
- [ ] MySQL running (local or cloud)
- [ ] Database initialized with SQL script
- [ ] .env file created (with real credentials)
- [ ] requirements.txt installed
- [ ] App runs locally: `streamlit run app.py`
- [ ] GitHub repo created
- [ ] Code pushed to GitHub
- [ ] Production server has MySQL access
- [ ] app.py running on production

---

## Example: Full Push to GitHub

```bash
# Current directory: ~/Automobile

# 1. Check git status
git status
# Should show only Python files, NOT CSV files

# 2. Make first commit
git add -A
git commit -m "Insurance platform: SQL-based, GitHub-ready

- Replaced CSV with MySQL database
- Added sql_init.py for one-time data load
- Updated requirements with mysql-connector
- Repository now < 1MB (was 100+ MB)"

# 3. Add GitHub remote
git remote add origin https://github.com/YOUR-USERNAME/Automobile.git

# 4. Push
git push -u origin main

# 5. Verify on GitHub (should be fast!)
# Check repo size: GitHub repo settings → About
# Should show "500 KB" or similar, NOT "100 MB"
```

---

## Repository Structure After Push

```
Automobile/
├── .git/                               (Git metadata)
├── .gitignore                          (Excludes CSV, .env)
├── .env.example                        (Template, safe to commit)
├── SQL_DEPLOYMENT_GUIDE.md             (This guide)
├── app.py                              (Streamlit app)
├── requirements.txt                    (Dependencies + mysql-connector)
├── project_structure/
│   ├── sql_init.py                     (Database init script)
│   ├── sql_data_manager.py             (Data access layer)
│   ├── config.yaml
│   └── api/
│       ├── database.py
│       ├── main.py
│       └── ...
├── Customer_Success_222331.ipynb       (Research notebook)
└── README.md

Total Size: ~500 KB ✅
(Compare to before: 100+ MB ❌)
```

---

## FAQ

**Q: What if I lose the CSV file?**  
A: The data is now in MySQL. Keep regular database backups:
```bash
mysqldump -u root -p insurance > backup_$(date +%Y%m%d).sql
```

**Q: Can I update data without re-running sql_init.py?**  
A: Yes, in production you can:
```sql
UPDATE policies SET premium = 100 WHERE policy_id = 123;
```
Or re-run `sql_init.py` to reload from CSV.

**Q: What about the FAISS index?**  
A: Optional for RAG features. Can be pre-computed and cached, not needed in GitHub.

**Q: Will the app work without MySQL?**  
A: No, MySQL is now required. It's the production database.

**Q: Can I use a different database (PostgreSQL)?**  
A: Yes, update `sql_data_manager.py` to use `psycopg2` instead of `mysql-connector-python`.

---

## Support Resources

1. **SQL_DEPLOYMENT_GUIDE.md** - Full deployment documentation
2. **requirements.txt** - All dependencies
3. **project_structure/sql_init.py** - Source code for initialization
4. **project_structure/sql_data_manager.py** - Source code for data access

---

**Status**: ✅ Ready for GitHub  
**Repository Size**: ~500 KB (was 100+ MB)  
**Deployment Time**: 5-10 minutes  
**Production Ready**: Yes ✅

Push to GitHub now! 🚀
