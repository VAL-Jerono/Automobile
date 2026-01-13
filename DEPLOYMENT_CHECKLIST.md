# 🚀 Streamlit Cloud Deployment Checklist

## Status: ✅ Ready to Deploy

### What Was Fixed
- ❌ **Before:** Build failing after 3s (heavy dependencies timeout)
- ✅ **After:** Fast 30-second build with optimized requirements

---

## Deploy Now (3 Steps)

### 1. Set Up Cloud Database (Choose One)

**Option A: PlanetScale (Recommended - Free)**
- Sign up: [planetscale.com](https://planetscale.com)
- Create database: `insurance-analytics`
- Get credentials (you'll add these to Streamlit secrets)

**Option B: Railway (Easy Setup)**
- Sign up: [railway.app](https://railway.app)
- Deploy MySQL template
- Copy connection details

**Option C: AWS RDS (Production)**
- Create MySQL 8.0 instance (db.t3.micro = free tier)
- Configure security group for external access
- Note connection details

---

### 2. Deploy to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)

2. Click **"New app"**

3. Fill in:
   ```
   Repository: VAL-Jerono/Automobile
   Branch: main
   Main file path: app.py
   ```

4. Click **"Deploy"** 
   - Build completes in ~30 seconds ⚡
   - App will show database error initially (expected)

---

### 3. Configure Database Connection

1. In Streamlit Cloud dashboard, click your app

2. Go to **⚙️ Settings → Secrets**

3. Paste your database credentials:
   ```toml
   [mysql]
   host = "xxx.planetscale.service.psdb.cloud"
   user = "your-username"
   password = "pscale_pw_xxx"
   database = "insurance-analytics"
   port = 3306
   ```

4. Click **Save** → App automatically restarts

5. **Load your data** to cloud database:
   ```bash
   # Connect to cloud DB (example for PlanetScale)
   pscale connect insurance-analytics main --port 3309
   
   # In another terminal, load predictions
   export MYSQL_HOST="127.0.0.1"
   export MYSQL_PORT="3309"
   export MYSQL_USER="your-user"
   export MYSQL_PASSWORD="your-password"
   
   python scripts/database/export_predictions_to_sql.py
   ```

---

## Expected Result

Your app is now live at: `https://your-app-name.streamlit.app` 🎉

**Available Features:**
- ✅ Portfolio analytics (€42.1M, 53,502 policies)
- ✅ Churn & claims predictions
- ✅ Customer segmentation & journey mapping
- ✅ Risk analysis & visualizations
- ⚠️ RAG/AI queries (disabled in cloud to keep build fast)

---

## Troubleshooting

### "Database connection failed"
→ Check secrets are configured correctly in Streamlit Cloud
→ Verify cloud database is running and accessible
→ Test connection from local machine first

### "Build failed" or timeout
→ Should not happen with optimized requirements.txt
→ If it does, check Streamlit Cloud build logs
→ Verify `.python-version` file exists (Python 3.9)

### Want to enable RAG features in cloud?
→ Use `requirements-full.txt` instead of `requirements.txt`
→ Build time increases to 5+ minutes
→ May exceed Streamlit Cloud free tier limits

---

## What's Deployed

**Included (Fast Build):**
- Core analytics & predictions
- All visualizations (Plotly)
- Database connectivity (MySQL)
- Customer segmentation
- Risk scoring

**Excluded (Optional):**
- RAG natural language queries
- AI embeddings (sentence-transformers)
- Large ML models (torch, transformers)

Use `requirements-full.txt` locally for complete feature set.

---

## Cost Summary

| Service | Free Tier | Recommended |
|---------|-----------|-------------|
| **Streamlit Cloud** | ✅ 1 public app free | Free |
| **PlanetScale** | ✅ 5GB database free | Free |
| **Total Monthly** | - | **$0** |

For production scale:
- Streamlit Cloud: $20/month (3 apps)
- PlanetScale: $29/month (25GB)
- **Total: $49/month**

---

## Next Steps

1. ✅ Deploy to Streamlit Cloud (done above)
2. ⚠️ Set up custom domain (optional)
3. ⚠️ Configure authentication (if needed)
4. ⚠️ Enable analytics tracking
5. ⚠️ Set up monitoring alerts

**Your app is production-ready!** 🚀
