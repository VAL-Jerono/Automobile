# 🏢 Insurance Customer Success Platform

**Streamlit Dashboard for Insurance Agents & Managers**

Built with models from `Auto_Analysis_Notebook.ipynb` | Powered by 8 predictive models + RAG system

---

## ✨ Features

### 📊 Executive Dashboard
- Real-time KPIs (customers, risk, CLV)
- Risk distribution & segment portfolio
- Claims risk analysis
- Action priority matrix

### 👥 Customer Intelligence
- Individual customer profiles
- Risk assessments (churn, claims, renewal)
- Value metrics (CLV, pricing adequacy)
- Recommended actions
- Portfolio comparison

### 🎯 Action Center
- **Critical Risk** - Immediate interventions
- **PROTECT** - High-value retention
- **DEVELOP** - Growth opportunities
- **Claims Risk** - Monitoring list

### 🔍 Smart Search (RAG)
- Natural language customer search
- Semantic similarity matching
- Advanced metadata filters
- Example queries

### 📈 Model Performance
- 8 model summary
- Prediction distributions
- Feature importance
- Data quality metrics

---

## 🚀 Quick Start

### Prerequisites

✅ **Required:** Run export cell in [Auto_Analysis_Notebook.ipynb](Auto_Analysis_Notebook.ipynb)
- Generates `model_outputs/rag_model_predictions.csv` with all 8 model predictions

⚠️ **Optional:** Run [project_structure/rag.ipynb](project_structure/rag.ipynb) Steps 1-6
- Creates `enhanced_faiss_index/` for Smart Search feature

### Launch (3 ways)

**Option 1: Quick Launch Script**
```bash
./launch_app.sh
```

**Option 2: Manual Launch**
```bash
streamlit run app.py
```

**Option 3: Network Access**
```bash
streamlit run app.py --server.address 0.0.0.0
# Access from any device: http://<your-ip>:8501
```

**The app will open automatically at:** `http://localhost:8501`

---

## 📦 What's Included

```
automobile_claims/
├── app.py                          # Main Streamlit application (1,000+ lines)
├── launch_app.sh                   # Quick launch script
├── requirements_streamlit.txt      # Python dependencies
├── DEPLOYMENT_GUIDE.md             # Full deployment documentation
├── README_STREAMLIT.md             # This file
│
├── model_outputs/
│   └── rag_model_predictions.csv  # ✅ REQUIRED: 8 model predictions
│
└── project_structure/
    └── enhanced_faiss_index/      # Optional: For Smart Search
        ├── index.faiss
        └── index.pkl
```

---

## 🎯 Use Cases

### For Insurance Agents
- **Before customer call:** View complete customer profile
- **Retention planning:** See recommended actions based on risk/value
- **Upsell opportunities:** Find DEVELOP segment customers
- **Smart search:** "High value customers in urban areas"

### For Managers
- **Weekly reviews:** Executive dashboard with portfolio health
- **Team assignments:** Export action lists to CSV
- **Performance tracking:** Monitor at-risk CLV and conversion
- **Strategic planning:** Analyze segment distribution

### For Analysts
- **Model monitoring:** Track prediction distributions
- **Data quality:** Check coverage and completeness
- **Feature analysis:** Review top predictors
- **Business intelligence:** CLV by segment, channel ROI

---

## 🎨 Screenshots

### Executive Dashboard
- Portfolio KPIs at a glance
- Risk/segment visualizations
- Action priority matrix

### Customer Intelligence
- Individual customer cards
- Risk gauges and metrics
- Recommended actions
- Benchmark comparisons

### Action Center
- Filterable customer lists
- Export to CSV
- Prioritized by CLV

---

## 🔧 Technical Details

### Models Integrated

| # | Model | Type | Performance | Key Predictor |
|---|-------|------|-------------|---------------|
| 1 | Customer Retention | Classification | AUC: 0.715 | R_Claims_history |
| 2 | Claims Frequency | Classification | AUC: 0.923 | R_Claims_history |
| 3 | Claim Severity | Regression | MAE: €189 | Premium |
| 4 | Customer Lifetime Value | Regression | R²: 0.89 | Premium + Seniority |
| 5 | Renewal Risk | Composite | Composite | Churn + Claims |
| 6 | Pricing Optimization | Business Logic | Business Rules | Expected Cost |
| 7 | Customer Segmentation | Rule-Based | Quadrant | CLV + Risk |
| 8 | Channel Attribution | Attribution | ROI Analysis | Distribution Channel |

### Data Flow

```
Auto_Analysis_Notebook.ipynb
    ↓ (Export predictions)
model_outputs/rag_model_predictions.csv (105,555 customers × 27 columns)
    ↓ (Load in app.py)
Streamlit Dashboard (5 pages)
    ↓ (Optional: RAG search)
enhanced_faiss_index/ (TF-IDF vectors + metadata)
```

### Performance

- **Dataset:** 105,555 customers
- **Load time:** ~2-3 seconds
- **Memory:** ~500MB RAM
- **RAG search:** ~100ms per query

---

## 📚 Documentation

- **[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)** - Complete deployment instructions
- **[Auto_Analysis_Notebook.ipynb](Auto_Analysis_Notebook.ipynb)** - Model training source
- **[project_structure/rag.ipynb](project_structure/rag.ipynb)** - RAG system creation
- **[app.py](app.py)** - Source code with detailed comments

---

## 🐛 Troubleshooting

### "Data file not found"
→ Run export cell in Auto_Analysis_Notebook.ipynb

### "RAG system not available"
→ Optional: Run project_structure/rag.ipynb Steps 1-6
→ Or: Use app without Smart Search (all other features work)

### "Module not found: streamlit"
```bash
pip install -r requirements_streamlit.txt
```

### Slow performance
→ Reduce table display to 25 rows (currently 50)
→ Add pagination for large datasets

---

## 🔐 Security Notes

**For Production Deployment:**

1. Add authentication (Streamlit Cloud, OAuth)
2. Encrypt sensitive data
3. Implement role-based access control
4. Enable audit logging for customer lookups
5. Mask PII in exports

---

## 🚀 Deployment Options

### Local (Development)
```bash
streamlit run app.py
```

### Streamlit Cloud (Free Hosting)
1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Deploy with one click

### Docker (Production)
```bash
docker build -t insurance-platform .
docker run -p 8501:8501 insurance-platform
```

See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for details.

---

## 📈 Future Enhancements

- [ ] Real-time database integration
- [ ] CRM export (Salesforce, HubSpot)
- [ ] Email alerts for critical risks
- [ ] Mobile-responsive design
- [ ] A/B testing for interventions
- [ ] Feedback loop for model retraining

---

## 📞 Support

**Questions?** Check the documentation:
- Technical issues → [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
- Model questions → [Auto_Analysis_Notebook.ipynb](Auto_Analysis_Notebook.ipynb)
- RAG system → [project_structure/rag.ipynb](project_structure/rag.ipynb)

---

## 📝 Version

**v1.0** (December 2025)
- Initial release
- 5 pages, 8 models, RAG integration
- 1,000+ lines of production code

---

## 🎉 Ready to Launch?

**Your app is running at:** http://localhost:8501

**Network URL:** http://10.57.0.65:8501

**External URL:** http://156.0.233.57:8501

---

**Built with ❤️ for Insurance Customer Success Teams**
