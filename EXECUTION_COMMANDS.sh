#!/bin/bash
# EXECUTION COMMANDS - Step-by-Step Instructions
# ================================================
# Copy and paste these commands in order

echo "🎯 Insurance Analytics: SQL Deployment Commands"
echo "================================================="
echo ""
echo "Run these commands in order to set up the system"
echo ""

# Step 1
echo "STEP 1: Navigate to project"
echo "---"
echo "cd /Users/leonida/Documents/automobile_claims/Automobile"
echo ""

# Step 2
echo "STEP 2: Initialize database (one-time)"
echo "---"
echo "cd project_structure"
echo "python sql_init.py --csv-path ../../Motor\ vehicle\ insurance\ data.csv"
echo ""
echo "Expected output:"
echo "  ✅ Database 'insurance' ready"
echo "  ✅ Created tables: customers, vehicles, policies, claims"
echo "  ✅ Loaded 105,555 policies from CSV"
echo ""

# Step 3
echo "STEP 3: Generate model predictions (5-10 minutes)"
echo "---"
echo "cd .."
echo "python export_predictions_to_sql.py"
echo ""
echo "Expected output:"
echo "  ✅ Extracted 66 code cells from notebook"
echo "  ✅ Executing notebook cells..."
echo "  ✅ Training 4 ML models"
echo "  ✅ Inserted 105,555 predictions into SQL"
echo "  ✅ Predictions generated successfully"
echo ""

# Step 4
echo "STEP 4: Verify data in database"
echo "---"
echo "mysql -u root -p insurance"
echo ""
echo "Then run in MySQL:"
echo "  SELECT COUNT(*) FROM model_predictions;"
echo ""
echo "Expected: 105555"
echo ""

# Step 5
echo "STEP 5: Run Streamlit app"
echo "---"
echo "streamlit run app.py"
echo ""
echo "Expected:"
echo "  ✅ Loaded predictions from MySQL database (SQL mode)"
echo "  ✅ App opens in browser at http://localhost:8501"
echo "  ✅ See all 4 dashboards and customer insights"
echo ""

# Step 6
echo "STEP 6: Push to GitHub"
echo "---"
echo "git add ."
echo "git commit -m 'feat: SQL-based prediction storage for deployment'"
echo "git push origin main"
echo ""
echo "Expected:"
echo "  ✅ Files pushed to GitHub"
echo "  ✅ GitHub Actions pipeline starts automatically"
echo "  ✅ Check Actions tab for progress"
echo ""

# Step 7
echo "STEP 7: Deploy to production (optional)"
echo "---"
echo "cd project_structure"
echo "docker-compose up -d"
echo ""
echo "Expected:"
echo "  ✅ MySQL container starts (port 3306)"
echo "  ✅ API container starts (port 8000)"
echo "  ✅ Streamlit container starts (port 8501)"
echo "  ✅ Access app at http://localhost:8501"
echo ""

echo "================================================="
echo "✨ Complete! Insurance Analytics is running"
echo "================================================="
