# ☁️ Cloud Deployment Guide

## Current Status
⚠️ **App configured for cloud deployment, but database not yet set up**

The app runs locally with XAMPP but fails on Streamlit Cloud because there's no database connection.

---

## Quick Fix: Deploy to Streamlit Cloud

### Step 1: Set Up Cloud Database

**Option A: PlanetScale (Recommended - Free Tier)**
1. Go to [planetscale.com](https://planetscale.com)
2. Sign up and create a new database: `insurance-analytics`
3. Create a password and save these credentials:
   ```
   Host: xxx.planetscale.service.psdb.cloud
   Username: xxx
   Password: pscale_pw_xxx
   Database: insurance-analytics
   Port: 3306
   ```
4. Connect using their CLI or web interface
5. Import your data:
   ```bash
   # Export from local MySQL
   mysqldump -u root insurance model_predictions > backup.sql
   
   # Import to PlanetScale (use their web console or CLI)
   ```

**Option B: AWS RDS MySQL (Production)**
1. Go to AWS RDS Console
2. Create MySQL 8.0 instance (db.t3.micro for free tier)
3. Set security group to allow connections from Streamlit Cloud IPs
4. Save credentials from AWS console

**Option C: Google Cloud SQL**
1. Go to Google Cloud Console → SQL
2. Create MySQL instance
3. Configure Cloud SQL Auth proxy or allow Streamlit Cloud IPs
4. Save connection details

---

### Step 2: Configure Streamlit Cloud Secrets

1. Push your code to GitHub:
   ```bash
   git add .
   git commit -m "feat: Add cloud deployment support"
   git push origin main
   ```

2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)

3. Deploy your app:
   - Repository: `VAL-Jerono/Automobile`
   - Branch: `main`
   - Main file: `Automobile/app.py`

4. Go to **App Settings → Secrets**

5. Add your database credentials:
   ```toml
   [mysql]
   host = "your-database-host.com"
   user = "your-username"
   password = "your-password"
   database = "insurance"
   port = 3306
   ```

6. Click **Save** and restart the app

---

### Step 3: Load Data to Cloud Database

**If using PlanetScale:**
```bash
# Connect to PlanetScale
pscale connect insurance-analytics main --port 3309

# In another terminal, load data
python scripts/database/export_predictions_to_sql.py
```

**If using AWS RDS or Google Cloud:**
```bash
# Update database connection in export script temporarily
# Or set environment variables:
export MYSQL_HOST="your-cloud-host.com"
export MYSQL_USER="your-username"
export MYSQL_PASSWORD="your-password"
export MYSQL_DATABASE="insurance"

python scripts/database/export_predictions_to_sql.py
```

---

## Architecture

### Local Development (Current)
```
Streamlit App (localhost:8501)
    ↓
MySQL (XAMPP - localhost:3306)
    ↓
insurance.model_predictions (53,502 records)
```

### Cloud Deployment (Target)
```
Streamlit Cloud (*.streamlit.app)
    ↓ (encrypted connection)
PlanetScale/AWS RDS/Google Cloud SQL
    ↓
insurance.model_predictions (53,502 records)
```

---

## Testing Cloud Connection Locally

Before deploying, test the cloud database locally:

1. Create `.streamlit/secrets.toml` with your cloud credentials:
   ```toml
   [mysql]
   host = "your-cloud-host.com"
   user = "your-username"
   password = "your-password"
   database = "insurance"
   port = 3306
   ```

2. Run the app:
   ```bash
   streamlit run app.py
   ```

3. If it works locally with cloud DB, it will work on Streamlit Cloud

---

## Cost Estimate

| Service | Free Tier | Paid |
|---------|-----------|------|
| **Streamlit Cloud** | ✅ 1 app free | $20/month for 3 apps |
| **PlanetScale** | ✅ 5GB free | $29/month for 25GB |
| **AWS RDS (db.t3.micro)** | ✅ 750 hrs/month (1 year) | ~$15/month |
| **Google Cloud SQL (db-f1-micro)** | ❌ No free tier | ~$10/month |

**Recommendation:** Start with PlanetScale (free 5GB) + Streamlit Cloud (free)

---

## Troubleshooting

### "Database connection failed" on Streamlit Cloud
- ✅ Check secrets are configured correctly in Streamlit Cloud settings
- ✅ Verify cloud database is running and accessible
- ✅ Test connection from local machine first
- ✅ Check database host allows connections from Streamlit Cloud IPs

### "Access denied" error
- Check username and password in Streamlit secrets
- Verify database user has correct permissions
- For PlanetScale: ensure password hasn't expired

### "Can't connect to MySQL server"
- Verify host address is correct
- Check port (usually 3306)
- Ensure firewall allows connections
- For AWS RDS: check security group rules

### Slow queries on cloud database
- Add indexes to frequently queried columns
- Use connection pooling (already implemented)
- Consider upgrading database tier

---

## Next Steps

1. **Choose a cloud database provider** (PlanetScale recommended)
2. **Set up the database** (5-10 minutes)
3. **Load your 53,502 predictions** (using export script)
4. **Configure Streamlit Cloud secrets** (2 minutes)
5. **Deploy and test** 🚀

**Need help?** Check the main [README.md](../README.md) for support resources.
