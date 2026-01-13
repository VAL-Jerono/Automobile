# Landing Page Component - Concise & Action-Oriented

This is the improved landing page to replace the wordy version. Use this in your Streamlit app.

---

## **OPTION 1: Ultra-Concise (Recommended)**

```python
# Landing Page - The 4 Critical Questions
st.markdown("""
### 🎯 Every Agent Must Answer 4 Questions:

| | Question | Model | Power |
|---|----------|-------|-------|
| 🔴 | Will they leave? | Churn Prediction | 71.5% accuracy |
| 💰 | Will they cost? | Claims Risk | 92.3% accuracy |
| 💎 | What are they worth? | Lifetime Value | €25.8M portfolio |
| 🧭 | Where are they headed? | Journey Segmentation | 4 segments |

**→ Use sidebar to explore portfolio intelligence**
""")

st.divider()
```

---

## **OPTION 2: Expanded with Numbers (More Context)**

```python
# Landing Page - The 4 Critical Questions with Key Insights
st.markdown("""
### 🎯 Every Agent Must Answer 4 Questions:

**🔴 Will this customer leave?**
- Churn Prediction Model (71.5% accuracy)
- Critical insight: Years 1-3 show 26.5% churn vs 16.7% at 10+ years
- Action: Identify at-risk customers, activate retention

**💰 Will this customer cost money?**
- Claims Risk Model (92.3% accuracy)
- Critical insight: Urban vans 26.8% claims vs agricultural 0.1%
- Action: Flag underpriced policies (14% = €500K opportunity)

**💎 What is this customer worth?**
- Lifetime Value Model (€25.8M validated portfolio)
- Critical insight: Top 2.8% of customers = 15.7% of value
- Action: Protect high-value customers, grow agent channel

**🧭 Where is this customer headed?**
- Journey Segmentation (PROTECT / DEVELOP / MANAGE / EXIT)
- Critical insight: €797K migration risk PROTECT→EXIT
- Action: Route to appropriate strategy per segment

---
### 📊 How to Use This Platform:

1. **Portfolio Dashboard** - Portfolio health snapshot
2. **Customer Search** - Individual customer intelligence
3. **Segment Analysis** - Deep dive into any segment
4. **Quick Actions** - Export lists for campaigns
5. **Documentation** - Reference all models & rules

---
""")
```

---

## **OPTION 3: Dashboard Cards (Visual)**

```python
import streamlit as st

# Landing Page with Metric Cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    ### 🔴 Will They Leave?
    **Churn Prediction**
    
    71.5% Accuracy
    
    22.1% Portfolio Rate
    """)
    
with col2:
    st.markdown("""
    ### 💰 Will They Cost?
    **Claims Risk**
    
    92.3% Accuracy
    
    18.6% Portfolio Rate
    """)

with col3:
    st.markdown("""
    ### 💎 What Worth?
    **Lifetime Value**
    
    €25.8M Portfolio
    
    €244 Average CLV
    """)

with col4:
    st.markdown("""
    ### 🧭 Where Headed?
    **Journey Segments**
    
    4 Quadrants
    
    €797K At Risk
    """)

st.divider()
st.markdown("### 📊 Navigate sidebar to explore →")
```

---

## **OPTION 4: Single Statement (Absolute Minimum)**

```python
st.markdown("""
🎯 **Insurance Agent Analytics** | Answer 4 Critical Questions | €25.8M Portfolio

🔴 Will they leave? (Churn: 71.5%) | 💰 Will they cost? (Claims: 92.3%) | 💎 What worth? (CLV: €25.8M) | 🧭 Where headed? (4 Segments)

→ Use sidebar navigation to explore
""")
```

---

## **Recommended Approach for Your App**

Use **OPTION 2** (Expanded) for the homepage because:
- ✅ Concise but not cryptic
- ✅ Provides context for each question
- ✅ Shows actionable insights
- ✅ Not too wordy
- ✅ Guides users to next steps

Place it **ABOVE** the navigation sidebar, like:

```python
# ============================================================================
# LANDING PAGE
# ============================================================================

st.markdown('<div class="main-header">🎯 Insurance Agent Analytics</div>', unsafe_allow_html=True)

col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("**The 4 Questions That Drive Portfolio Success**")
with col2:
    st.metric("Portfolio", "€25.8M")

# LANDING PAGE CONTENT (use Option 2 above)
st.markdown("""
### 🎯 Every Agent Must Answer 4 Questions:
...etc
""")

st.divider()

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

st.sidebar.markdown("### 🎯 Navigation")
page = st.sidebar.radio(
    "Select View",
    ["📊 Portfolio Dashboard", "👥 Customer Search", "📈 Segment Analysis", "⚡ Quick Actions", "📚 Documentation"]
)
```

---

## **How to Implement**

### **Step 1: Find This In Your app.py:**
```python
# Main header
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown('<div class="main-header">📊 Insurance Analytics Platform</div>', unsafe_allow_html=True)
    st.markdown('**Real-time dashboard powered by SQL database + 6 ML models**')
with col2:
    st.metric("Last Update", datetime.now().strftime("%H:%M:%S"))
```

### **Step 2: Replace With This:**
```python
# Main header
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown('<div class="main-header">🎯 Insurance Agent Analytics</div>', unsafe_allow_html=True)
    st.markdown('**The 4 Questions That Drive Portfolio Success**')
with col2:
    st.metric("Portfolio Value", "€25.8M")

# Landing Page - The 4 Critical Questions
st.markdown("""
### 🎯 Every Agent Must Answer 4 Questions:

**🔴 Will this customer leave?**
- Churn Prediction Model (71.5% accuracy)
- Critical insight: Years 1-3 show 26.5% churn vs 16.7% at 10+ years
- Action: Identify at-risk customers, activate retention

**💰 Will this customer cost money?**
- Claims Risk Model (92.3% accuracy)
- Critical insight: Urban vans 26.8% claims vs agricultural 0.1%
- Action: Flag underpriced policies (14% = €500K opportunity)

**💎 What is this customer worth?**
- Lifetime Value Model (€25.8M validated portfolio)
- Critical insight: Top 2.8% of customers = 15.7% of value
- Action: Protect high-value customers, grow agent channel

**🧭 Where is this customer headed?**
- Journey Segmentation (PROTECT / DEVELOP / MANAGE / EXIT)
- Critical insight: €797K migration risk PROTECT→EXIT
- Action: Route to appropriate strategy per segment

---
### 📊 How to Use This Platform:

1. **Portfolio Dashboard** - Portfolio health snapshot
2. **Customer Search** - Individual customer intelligence
3. **Segment Analysis** - Deep dive into any segment
4. **Quick Actions** - Export lists for campaigns
5. **Documentation** - Reference all models & rules
""")

st.divider()
```

### **Step 3: Keep Everything After This As-Is**

The rest of the app (sidebar navigation, pages, etc.) stays the same.

---

## **Result**

Before:
```
❌ Overly wordy landing page
❌ Confusing "how it works" section
❌ CSV file error message
❌ Low trust in data source
```

After:
```
✅ Concise 4-question framework
✅ Clear actions for each question
✅ Real data from MySQL
✅ Professional, trustworthy
```

---

## **Testing After Update**

```bash
# Restart app with new landing page:
streamlit cache clear
streamlit run app.py

# Should see:
✅ "🎯 Insurance Agent Analytics" header
✅ "The 4 Questions That Drive Portfolio Success"
✅ 4 question sections with actions
✅ "Use sidebar to explore" guidance
✅ NO CSV error messages
```

---

**Replace old landing page with Option 2 above and you're done! 🚀**
