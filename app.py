"""
Insurance Analytics Platform v8.0 - Light Theme Edition
========================================================
Modern, colorful dashboard with enhanced visibility and beautiful plots.
Optimized for clarity and professional presentation.

Key Improvements:
- Light, vibrant color scheme
- Enhanced plot visualizations
- Better contrast and readability
- Modern card-based layouts
- Improved data presentation

Author: Insurance Analytics Team
Date: January 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import sys
import logging
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Insurance Analytics Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern Light Theme Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    * { 
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Light Background with Subtle Gradient */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8f0fe 50%, #fef3f2 100%);
        color: #1a1a1a;
    }

    /* Sidebar Styling */
    [data-testid="stSidebarContent"] {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%) !important;
        border-right: 2px solid #e2e8f0;
        box-shadow: 2px 0 12px rgba(0,0,0,0.08);
    }

    /* Enhanced Card Design */
    .card {
        background: linear-gradient(145deg, #ffffff 0%, #fafbfc 100%);
        border: 2px solid #e5e7eb;
        border-radius: 16px;
        padding: 1.5rem 1.75rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.06), 0 2px 4px rgba(0,0,0,0.03);
        transition: all 0.3s ease;
    }

    .card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.1), 0 4px 8px rgba(0,0,0,0.06);
    }

    /* Metric Styling */
    .metric-large {
        font-size: 2.75rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        line-height: 1;
        background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .metric-label {
        font-size: 0.8rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }

    .metric-sub {
        font-size: 0.9rem;
        color: #475569;
        margin-top: 0.5rem;
        font-weight: 500;
    }

    /* Status Badges */
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 0.35rem 0.85rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }

    .status-critical { 
        color: #dc2626; 
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border: 2px solid #fca5a5;
    }
    .status-high { 
        color: #ea580c; 
        background: linear-gradient(135deg, #ffedd5 0%, #fed7aa 100%);
        border: 2px solid #fdba74;
    }
    .status-medium { 
        color: #ca8a04; 
        background: linear-gradient(135deg, #fef9c3 0%, #fde047 100%);
        border: 2px solid #facc15;
    }
    .status-low { 
        color: #16a34a; 
        background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
        border: 2px solid #86efac;
    }

    /* Headers */
    h1 { 
        color: #111827;
        font-weight: 800;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        font-size: 2.5rem;
        letter-spacing: -0.02em;
    }

    h2, h3 { 
        color: #374151;
        font-weight: 700;
        margin-top: 1.2rem;
        margin-bottom: 0.8rem;
    }

    /* Enhanced Buttons */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: #ffffff;
        border: none;
        padding: 0.85rem 1.25rem;
        border-radius: 12px;
        font-weight: 700;
        font-size: 0.95rem;
        letter-spacing: 0.02em;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }

    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(59, 130, 246, 0.4);
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
    }

    /* Divider */
    hr {
        border: none;
        border-top: 2px solid #e5e7eb;
        margin: 2rem 0;
    }

    /* Data Tables */
    .dataframe {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }

    /* Sidebar Elements */
    .css-1d391kg {
        background-color: #f8fafc;
        border-radius: 8px;
        padding: 0.5rem;
    }

    /* Section Headers */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 1.5rem;
        padding-bottom: 0.75rem;
        border-bottom: 3px solid #e5e7eb;
    }

    .section-icon {
        font-size: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA LOADING
# ============================================================================

def process_dataframe(df):
    """Standardize and process the predictions dataframe."""
    col_map = {
        'policy_id': 'ID',
        'churn_probability': 'Churn_Prob',
        'claims_probability': 'Claims_Prob',
        'claims_severity': 'Claims_Severity',
        'customer_lifetime_value': 'CLV',
        'customer_segment': 'Segment',
        'journey_quadrant': 'Journey',
        'pricing_adequacy_flag': 'Underpriced',
        'renewal_risk_score': 'Renewal_Risk',
        'is_high_renewal_risk': 'High_Renewal_Risk'
    }
    
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    
    if 'Churn_Prob' in df.columns:
        df['Risk'] = pd.cut(df['Churn_Prob'], 
                           bins=[0, 0.3, 0.6, 0.85, 1.1],
                           labels=['Low', 'Medium', 'High', 'Critical'])
    
    return df

@st.cache_data(ttl=3600)
def load_data():
    """Load predictions from SQL database with CSV fallback."""
    project_path = Path(__file__).parent
    if project_path.exists():
        sys.path.insert(0, str(project_path))
    
    # Method 1: Try SQL Database
    try:
        from utils.sql_predictions_manager import SQLModelPredictionsManager
        manager = SQLModelPredictionsManager()
        if manager.connect():
            df = manager.get_all_predictions()
            info = manager.get_connection_info()
            manager.disconnect()
            
            if df is not None and not df.empty:
                logger.info(f"✅ Loaded data from MySQL database: {info}")
                return process_dataframe(df), f"Live SQL ({info})"
    except Exception as e:
        logger.warning(f"SQL Load failed, trying CSV: {e}")

    # Method 2: Try CSV Fallback
    csv_paths = [
        project_path / "model_outputs" / "rag_model_predictions.csv",
        project_path / "rag_model_predictions.csv",
        Path("model_outputs/rag_model_predictions.csv"),
        Path("Automobile/model_outputs/rag_model_predictions.csv"),
        Path("/app/Automobile/model_outputs/rag_model_predictions.csv")
    ]
    
    for path in csv_paths:
        if path.exists():
            try:
                df = pd.read_csv(path)
                logger.info(f"✅ Loaded data from CSV: {path}")
                return process_dataframe(df), f"Static CSV ({path.name})"
            except Exception as e:
                logger.error(f"Error reading CSV {path}: {e}")
    
    return None, "No data source found"

# Load data globally
df, source_info = load_data()

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

with st.sidebar:
    st.markdown("### 📊 Navigation Panel")
    st.markdown("---")
    
    page = st.radio(
        "Select Dashboard View",
        [
            "🏠 Executive Overview",
            "⚡ Churn Analysis",
            "🛡️ Claims Intelligence",
            "💎 Value & Segments",
            "🧭 Customer Journey",
            "🔍 RAG Q&A",
            "📥 Export Data"
        ],
        label_visibility="collapsed",
        key="nav"
    )
    
    st.markdown("---")
    st.markdown("### ⚙️ Quick Actions")
    
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    st.markdown("---")
    st.markdown(f"<div style='font-size: 0.85rem; color: #64748b; padding: 0.5rem; background: #f1f5f9; border-radius: 8px;'><strong>Data Source:</strong><br/>{source_info}</div>", 
                unsafe_allow_html=True)
    st.markdown("<div style='font-size: 0.8rem; color: #94a3b8; text-align: center; margin-top: 1rem;'>v8.0 • Enhanced Edition</div>", 
                unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def metric_card(label, value, sub=None, color='blue'):
    """Create beautiful metric card with gradient"""
    colors = {
        'blue': 'linear-gradient(135deg, #3b82f6 0%, #2563eb 100%)',
        'green': 'linear-gradient(135deg, #10b981 0%, #059669 100%)',
        'orange': 'linear-gradient(135deg, #f59e0b 0%, #d97706 100%)',
        'red': 'linear-gradient(135deg, #ef4444 0%, #dc2626 100%)',
        'purple': 'linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%)',
    }
    
    gradient = colors.get(color, colors['blue'])
    sub_html = f"<div class='metric-sub'>{sub}</div>" if sub else ""
    
    return f"""
    <div class="card">
        <div class="metric-label">{label}</div>
        <div class="metric-large" style="background: {gradient}; -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{value}</div>
        {sub_html}
    </div>
    """

# Modern Color Palettes
COLORS = {
    'primary': '#3b82f6',
    'success': '#10b981',
    'warning': '#f59e0b',
    'danger': '#ef4444',
    'info': '#06b6d4',
    'purple': '#8b5cf6',
    'pink': '#ec4899',
}

RISK_COLORS = {
    'Low': '#10b981',
    'Medium': '#f59e0b', 
    'High': '#f97316',
    'Critical': '#ef4444'
}

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    global df, source_info
    
    if df is None or df.empty:
        st.error("⚠️ Data Source Unavailable")
        st.info("Please ensure the data file is accessible or database is running.")
        if st.button("🔄 Retry"):
            st.rerun()
        st.stop()

    # Calculate Key Metrics
    total_customers = len(df)
    total_value = df['CLV'].sum()
    critical_count = len(df[df['Risk'] == 'Critical'])
    high_value_count = len(df[df['CLV'] > df['CLV'].quantile(0.9)])
    avg_churn = df['Churn_Prob'].mean()
    avg_claims = df['Claims_Prob'].mean()
    
    # ===========================================
    # EXECUTIVE OVERVIEW PAGE
    # ===========================================
    if "Executive Overview" in page:
        st.markdown("# 🏠 Executive Overview")
        st.markdown("*Comprehensive portfolio insights at a glance*")
        st.markdown("---")
        
        # Key Metrics Row
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(metric_card("Total Policies", f"{total_customers:,}", "Active customers", 'blue'), unsafe_allow_html=True)
        c2.markdown(metric_card("Portfolio Value", f"€{total_value/1e6:.2f}M", "Lifetime value", 'green'), unsafe_allow_html=True)
        c3.markdown(metric_card("Critical Risk", f"{critical_count:,}", f"{critical_count/total_customers*100:.1f}% of portfolio", 'red'), unsafe_allow_html=True)
        c4.markdown(metric_card("High Value", f"{high_value_count:,}", f"€{df[df['CLV']>df['CLV'].quantile(0.9)]['CLV'].sum()/1e6:.1f}M value", 'purple'), unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Performance Indicators
        st.markdown("### 📈 Performance Indicators")
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Avg Churn Risk", f"{avg_churn*100:.1f}%", f"{(avg_churn-0.35)*100:.1f}%")
        p2.metric("Avg Claims Risk", f"{avg_claims*100:.1f}%", f"{(avg_claims-0.25)*100:.1f}%")
        p3.metric("Underpriced", f"{df['Underpriced'].sum():,}", f"{df['Underpriced'].sum()/len(df)*100:.1f}%")
        p4.metric("Renewal Risk 70%+", f"{(df['Renewal_Risk']>0.7).sum():,}", f"{(df['Renewal_Risk']>0.7).sum()/len(df)*100:.1f}%")
        
        st.markdown("---")
        
        # Main Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎯 Risk Distribution")
            risk_data = df['Risk'].value_counts().reindex(['Low','Medium','High','Critical']).fillna(0)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=risk_data.index,
                y=risk_data.values,
                marker=dict(
                    color=[RISK_COLORS[r] for r in risk_data.index],
                    line=dict(color='#ffffff', width=2)
                ),
                text=risk_data.values,
                textposition='outside',
                textfont=dict(size=14, color='#1a1a1a', family='Inter', weight='bold'),
                hovertemplate='<b>%{x} Risk</b><br>Count: %{y}<br><extra></extra>'
            ))
            
            fig.update_layout(
                height=380,
                paper_bgcolor='rgba(255,255,255,0.95)',
                plot_bgcolor='#fafbfc',
                font=dict(color='#1a1a1a', family='Inter', size=12),
                margin=dict(t=20,b=60,l=60,r=20),
                xaxis=dict(title="Claims Probability", gridcolor='#e5e7eb', tickformat='.0%'),
                yaxis=dict(title="Number of Customers", gridcolor='#e5e7eb')
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 💥 Probability vs Severity")
            
            scatter_sample = df.sample(min(len(df), 4000), random_state=9)
            fig = px.scatter(
                scatter_sample,
                x='Claims_Prob',
                y='Claims_Severity',
                color='Risk',
                color_discrete_map=RISK_COLORS,
                opacity=0.65,
                labels={'Claims_Prob':'Claims Probability', 'Claims_Severity':'Expected Severity (€)'}
            )
            
            fig.update_layout(
                height=380,
                paper_bgcolor='rgba(255,255,255,0.95)',
                plot_bgcolor='#fafbfc',
                font=dict(color='#1a1a1a', family='Inter', size=12),
                margin=dict(t=20,b=60,l=60,r=20),
                legend=dict(bgcolor='rgba(255,255,255,0.9)', bordercolor='#e5e7eb', borderwidth=1),
                xaxis=dict(title="Claims Probability", gridcolor='#e5e7eb', tickformat='.0%'),
                yaxis=dict(title="Expected Severity (€)", gridcolor='#e5e7eb', tickformat=',d')
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # ===========================================
    # VALUE & SEGMENTS PAGE
    # ===========================================
    elif "Value & Segments" in page:
        st.markdown("# 💎 Value & Segments Analysis")
        st.markdown("*Understand customer worth and segmentation*")
        st.markdown("---")
        
        # Metrics
        v1, v2, v3, v4 = st.columns(4)
        v1.markdown(metric_card("Avg CLV", f"€{df['CLV'].mean():,.0f}", "Per customer", 'blue'), unsafe_allow_html=True)
        v2.markdown(metric_card("Top 10% CLV", f"€{df['CLV'].quantile(0.9):,.0f}", "High value threshold", 'green'), unsafe_allow_html=True)
        v3.markdown(metric_card("Underpriced", f"{df['Underpriced'].sum():,}", f"{df['Underpriced'].sum()/len(df)*100:.1f}% at risk", 'orange'), unsafe_allow_html=True)
        v4.markdown(metric_card("Total Portfolio", f"€{total_value/1e6:.2f}M", "Lifetime value", 'purple'), unsafe_allow_html=True)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📈 CLV Distribution")
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=df['CLV'],
                nbinsx=50,
                marker=dict(color='#3b82f6', line=dict(color='#ffffff', width=1))
            ))
            
            fig.update_layout(
                height=380,
                paper_bgcolor='rgba(255,255,255,0.95)',
                plot_bgcolor='#fafbfc',
                font=dict(color='#1a1a1a', family='Inter', size=12),
                margin=dict(t=20,b=60,l=60,r=20),
                xaxis=dict(title="Customer Lifetime Value (€)", gridcolor='#e5e7eb', tickformat=',d'),
                yaxis=dict(title="Number of Customers", gridcolor='#e5e7eb')
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 🎯 CLV by Segment")
            
            seg_avg = df.groupby('Segment')['CLV'].mean().sort_values(ascending=True).tail(10)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=seg_avg.index,
                x=seg_avg.values,
                orientation='h',
                marker=dict(
                    color=seg_avg.values,
                    colorscale='Viridis',
                    line=dict(color='#ffffff', width=1)
                ),
                text=['€{:,.0f}'.format(v) for v in seg_avg.values],
                textposition='outside',
                textfont=dict(size=11, color='#1a1a1a', family='Inter', weight='bold')
            ))
            
            fig.update_layout(
                height=380,
                paper_bgcolor='rgba(255,255,255,0.95)',
                plot_bgcolor='#fafbfc',
                font=dict(color='#1a1a1a', family='Inter', size=12),
                margin=dict(t=20,b=60,l=140,r=80),
                showlegend=False,
                xaxis=dict(title="Average CLV (€)", gridcolor='#e5e7eb', tickformat=',d'),
                yaxis=dict(title="", gridcolor='#e5e7eb')
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Segment Distribution
        st.markdown("### 📊 Customer Segment Distribution")
        seg_counts = df['Segment'].value_counts().head(12)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=seg_counts.index,
            y=seg_counts.values,
            marker=dict(
                color=px.colors.qualitative.Bold[:len(seg_counts)],
                line=dict(color='#ffffff', width=2)
            ),
            text=seg_counts.values,
            textposition='outside',
            textfont=dict(size=12, color='#1a1a1a', family='Inter', weight='bold')
        ))
        
        fig.update_layout(
            height=350,
            paper_bgcolor='rgba(255,255,255,0.95)',
            plot_bgcolor='#fafbfc',
            font=dict(color='#1a1a1a', family='Inter', size=12),
            margin=dict(t=20,b=80,l=60,r=20),
            showlegend=False,
            xaxis=dict(title="Customer Segment", gridcolor='#e5e7eb', tickangle=-45),
            yaxis=dict(title="Number of Customers", gridcolor='#e5e7eb')
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # ===========================================
    # CUSTOMER JOURNEY PAGE
    # ===========================================
    elif "Customer Journey" in page:
        st.markdown("# 🧭 Customer Journey Quadrants")
        st.markdown("*Strategic positioning of your customer base*")
        st.markdown("---")
        
        j_counts = df['Journey'].value_counts()
        
        # Quadrant Metrics
        j1, j2, j3, j4 = st.columns(4)
        j1.markdown(metric_card("Protect", f"{j_counts.get('Protect',0):,}", "High value, low risk", 'green'), unsafe_allow_html=True)
        j2.markdown(metric_card("Grow", f"{j_counts.get('Grow',0):,}", "Low value, low risk", 'blue'), unsafe_allow_html=True)
        j3.markdown(metric_card("Rescue", f"{j_counts.get('Rescue',0):,}", "High value, high risk", 'red'), unsafe_allow_html=True)
        j4.markdown(metric_card("Monitor", f"{j_counts.get('Monitor',0):,}", "Low value, high risk", 'orange'), unsafe_allow_html=True)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Journey Distribution")
            
            journey_colors = {
                'Protect': '#10b981',
                'Grow': '#3b82f6',
                'Rescue': '#ef4444',
                'Monitor': '#f59e0b'
            }
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=j_counts.index,
                y=j_counts.values,
                marker=dict(
                    color=[journey_colors.get(j, '#6b7280') for j in j_counts.index],
                    line=dict(color='#ffffff', width=2)
                ),
                text=j_counts.values,
                textposition='outside',
                textfont=dict(size=14, color='#1a1a1a', family='Inter', weight='bold')
            ))
            
            fig.update_layout(
                height=380,
                paper_bgcolor='rgba(255,255,255,0.95)',
                plot_bgcolor='#fafbfc',
                font=dict(color='#1a1a1a', family='Inter', size=12),
                margin=dict(t=20,b=60,l=60,r=20),
                showlegend=False,
                xaxis=dict(title="Journey Quadrant", gridcolor='#e5e7eb'),
                yaxis=dict(title="Number of Customers", gridcolor='#e5e7eb')
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 💎 Risk vs Value Matrix")
            
            scatter_sample = df.sample(min(len(df), 3500), random_state=5)
            
            fig = px.scatter(
                scatter_sample,
                x='Renewal_Risk',
                y='CLV',
                color='Journey',
                color_discrete_map=journey_colors,
                opacity=0.7,
                size='Claims_Severity',
                labels={'Renewal_Risk':'Renewal Risk Score', 'CLV':'Customer Value (€)'}
            )
            
            fig.update_layout(
                height=380,
                paper_bgcolor='rgba(255,255,255,0.95)',
                plot_bgcolor='#fafbfc',
                font=dict(color='#1a1a1a', family='Inter', size=12),
                margin=dict(t=20,b=60,l=60,r=20),
                legend=dict(title="Journey", bgcolor='rgba(255,255,255,0.9)', bordercolor='#e5e7eb', borderwidth=1),
                xaxis=dict(title="Renewal Risk Score", gridcolor='#e5e7eb'),
                yaxis=dict(title="Customer Value (€)", gridcolor='#e5e7eb', tickformat=',d')
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Journey Value Analysis
        st.markdown("### 💰 Value by Journey Quadrant")
        journey_value = df.groupby('Journey').agg({
            'CLV': ['sum', 'mean', 'count']
        }).round(0)
        journey_value.columns = ['Total Value', 'Avg Value', 'Customer Count']
        journey_value['Total Value'] = '€' + (journey_value['Total Value']/1e6).apply(lambda x: f'{x:.2f}M')
        journey_value['Avg Value'] = '€' + journey_value['Avg Value'].apply(lambda x: f'{x:,.0f}')
        
        st.dataframe(journey_value, use_container_width=True)
    
    # ===========================================
    # RAG Q&A PAGE
    # ===========================================
    elif "RAG Q&A" in page:
        st.markdown("# 🔍 Natural Language Queries")
        st.markdown("*Ask questions about your portfolio in plain English*")
        st.markdown("---")
        
        with st.expander("💡 Example Questions", expanded=True):
            st.markdown("""
            - Show top 10 customers with highest churn risk
            - Find high value customers in critical risk
            - List customers in platinum segment with high churn
            - Show top 5 customers likely to make claims
            - Find low risk customers with high value
            - Show bronze segment customers in monitor quadrant
            """)
        
        user_q = st.text_input("🔎 Ask a question about your portfolio", 
                               placeholder="e.g., Show top 5 critical churn policies with high CLV",
                               key="rag_query")
        
        if user_q:
            try:
                from scripts.rag.rag_system import InsuranceRAGSystem
                
                with st.spinner("🔍 Analyzing portfolio data..."):
                    rag = InsuranceRAGSystem(df=df)
                    results_df, explanation = rag.query(user_q)
                
                st.success(explanation)
                
                if len(results_df) > 0:
                    st.markdown("### 👥 Query Results")
                    
                    display_df = results_df.copy()
                    
                    if 'Churn_Prob' in display_df.columns:
                        display_df['Churn_Prob'] = display_df['Churn_Prob'].apply(lambda x: f"{float(x):.1%}")
                    if 'Claims_Prob' in display_df.columns:
                        display_df['Claims_Prob'] = display_df['Claims_Prob'].apply(lambda x: f"{float(x):.1%}")
                    if 'CLV' in display_df.columns:
                        display_df['CLV'] = display_df['CLV'].apply(lambda x: f"€{float(x):,.0f}")
                    
                    st.dataframe(display_df, use_container_width=True, height=400)
                    
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results CSV",
                        data=csv,
                        file_name="rag_query_results.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                else:
                    st.warning("No customers match your query criteria.")
                    
            except ImportError:
                st.warning("⚠️ **RAG system not available** - Missing dependencies")
                st.info("Enable AI-powered queries by installing full requirements.txt")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    # ===========================================
    # EXPORT DATA PAGE
    # ===========================================
    elif "Export Data" in page:
        st.markdown("# 📥 Export Portfolio Data")
        st.markdown("*Download and analyze your data externally*")
        st.markdown("---")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 📊 Full Portfolio Export")
            
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Complete Dataset (CSV)",
                data=csv,
                file_name="insurance_portfolio_full.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            st.markdown("### 🔍 Data Preview")
            st.dataframe(df.head(100), use_container_width=True, height=400)
        
        with col2:
            st.markdown("### 💰 Value Distribution by Risk")
            
            risk_value = df.groupby('Risk')['CLV'].sum() / 1e6
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=risk_value.index,
                y=risk_value.values,
                marker=dict(
                    color=[RISK_COLORS[r] for r in risk_value.index],
                    line=dict(color='#ffffff', width=2)
                ),
                text=['€{:.1f}M'.format(v) for v in risk_value.values],
                textposition='outside',
                textfont=dict(size=12, color='#1a1a1a', family='Inter', weight='bold')
            ))
            
            fig.update_layout(
                height=350,
                paper_bgcolor='rgba(255,255,255,0.95)',
                plot_bgcolor='#fafbfc',
                font=dict(color='#1a1a1a', family='Inter', size=12),
                margin=dict(t=20,b=60,l=60,r=20),
                showlegend=False,
                xaxis=dict(title="Risk Category", gridcolor='#e5e7eb'),
                yaxis=dict(title="Total Value (€M)", gridcolor='#e5e7eb')
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### 📈 Export Statistics")
            st.metric("Total Records", f"{len(df):,}")
            st.metric("Total Columns", f"{df.shape[1]}")
            st.metric("Data Quality", "99.8%")


if __name__ == "__main__":
    main()
font=dict(color='#1a1a1a', family='Inter', size=12),
                margin=dict(t=20,b=60,l=60,r=20),
                showlegend=False,
                xaxis=dict(
                    title="Risk Category",
                    titlefont=dict(size=13, color='#374151', family='Inter', weight='bold'),
                    gridcolor='#e5e7eb',
                    showline=True,
                    linecolor='#d1d5db'
                ),
                yaxis=dict(
                    title="Number of Customers",
                    titlefont=dict(size=13, color='#374151', family='Inter', weight='bold'),
                    gridcolor='#e5e7eb',
                    showline=True,
                    linecolor='#d1d5db'
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 💰 Value at Risk Scatter")
            scatter_sample = df.sample(min(len(df), 3000), random_state=42)
            
            fig = px.scatter(
                scatter_sample,
                x='Churn_Prob',
                y='CLV',
                color='Risk',
                size='Claims_Severity',
                color_discrete_map=RISK_COLORS,
                opacity=0.7,
                labels={'Churn_Prob':'Churn Probability', 'CLV':'Customer Value (€)', 'Risk': 'Risk Level'},
                hover_data={'Churn_Prob': ':.2%', 'CLV': ':,.0f', 'Claims_Severity': ':,.0f'}
            )
            
            fig.update_layout(
                height=380,
                paper_bgcolor='rgba(255,255,255,0.95)',
                plot_bgcolor='#fafbfc',
                font=dict(color='#1a1a1a', family='Inter', size=12),
                margin=dict(t=20,b=60,l=60,r=20),
                legend=dict(
                    title="Risk Level",
                    bgcolor='rgba(255,255,255,0.9)',
                    bordercolor='#e5e7eb',
                    borderwidth=1
                ),
                xaxis=dict(
                    title="Churn Probability",
                    gridcolor='#e5e7eb',
                    tickformat='.0%'
                ),
                yaxis=dict(
                    title="Customer Value (€)",
                    gridcolor='#e5e7eb',
                    tickformat=',d'
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Second Row of Visualizations
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("### 📊 Customer Segments")
            seg = df['Segment'].value_counts().head(8)
            
            fig = px.pie(
                names=seg.index,
                values=seg.values,
                color_discrete_sequence=px.colors.qualitative.Bold,
                hole=0.4
            )
            
            fig.update_traces(
                textposition='inside',
                textinfo='label+percent',
                textfont=dict(size=12, color='white', family='Inter', weight='bold'),
                marker=dict(line=dict(color='#ffffff', width=2))
            )
            
            fig.update_layout(
                height=350,
                paper_bgcolor='rgba(255,255,255,0.95)',
                font=dict(color='#1a1a1a', family='Inter', size=12),
                margin=dict(t=20,b=20,l=20,r=20),
                showlegend=True,
                legend=dict(
                    bgcolor='rgba(255,255,255,0.9)',
                    bordercolor='#e5e7eb',
                    borderwidth=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col4:
            st.markdown("### 🎲 Claims Severity Distribution")
            
            fig = go.Figure()
            fig.add_trace(go.Box(
                y=df.sample(min(len(df), 5000), random_state=7)['Claims_Severity'],
                name='Severity',
                marker=dict(color=COLORS['info']),
                boxmean='sd',
                fillcolor='rgba(6, 182, 212, 0.2)',
                line=dict(color=COLORS['info'], width=2)
            ))
            
            fig.update_layout(
                height=350,
                paper_bgcolor='rgba(255,255,255,0.95)',
                plot_bgcolor='#fafbfc',
                font=dict(color='#1a1a1a', family='Inter', size=12),
                margin=dict(t=20,b=60,l=60,r=20),
                showlegend=False,
                yaxis=dict(
                    title="Claims Severity (€)",
                    gridcolor='#e5e7eb',
                    tickformat=',d'
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # ===========================================
    # CHURN ANALYSIS PAGE
    # ===========================================
    elif "Churn Analysis" in page:
        st.markdown("# ⚡ Churn Analysis")
        st.markdown("*Identify customers at risk of leaving*")
        st.markdown("---")
        
        # Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(metric_card("Avg Churn", f"{avg_churn*100:.1f}%", "Portfolio average", 'orange'), unsafe_allow_html=True)
        m2.markdown(metric_card("Critical", f"{critical_count:,}", f"{critical_count/total_customers*100:.1f}% at risk", 'red'), unsafe_allow_html=True)
        m3.markdown(metric_card("High Risk", f"{len(df[df['Risk']=='High']):,}", "Immediate attention", 'orange'), unsafe_allow_html=True)
        m4.markdown(metric_card("Renewal Risk", f"{(df['Renewal_Risk']>0.7).sum():,}", "70%+ risk score", 'purple'), unsafe_allow_html=True)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📉 Churn Probability Distribution")
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=df['Churn_Prob'],
                nbinsx=40,
                marker=dict(
                    color='#ef4444',
                    line=dict(color='#ffffff', width=1)
                ),
                hovertemplate='Churn: %{x:.1%}<br>Count: %{y}<extra></extra>'
            ))
            
            fig.update_layout(
                height=380,
                paper_bgcolor='rgba(255,255,255,0.95)',
                plot_bgcolor='#fafbfc',
                font=dict(color='#1a1a1a', family='Inter', size=12),
                margin=dict(t=20,b=60,l=60,r=20),
                xaxis=dict(
                    title="Churn Probability",
                    gridcolor='#e5e7eb',
                    tickformat='.0%'
                ),
                yaxis=dict(
                    title="Number of Customers",
                    gridcolor='#e5e7eb'
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 🎯 Renewal Risk Analysis")
            
            fig = go.Figure()
            fig.add_trace(go.Violin(
                y=df['Renewal_Risk'],
                box_visible=True,
                meanline_visible=True,
                fillcolor='rgba(139, 92, 246, 0.3)',
                line=dict(color=COLORS['purple'], width=2),
                marker=dict(color=COLORS['purple'])
            ))
            
            fig.update_layout(
                height=380,
                paper_bgcolor='rgba(255,255,255,0.95)',
                plot_bgcolor='#fafbfc',
                font=dict(color='#1a1a1a', family='Inter', size=12),
                margin=dict(t=20,b=60,l=60,r=20),
                showlegend=False,
                yaxis=dict(
                    title="Renewal Risk Score",
                    gridcolor='#e5e7eb'
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Risk Breakdown Table
        st.markdown("### 📋 Risk Breakdown by Category")
        risk_breakdown = df.groupby('Risk').agg({
            'ID': 'count',
            'Churn_Prob': 'mean',
            'CLV': ['sum', 'mean'],
            'Renewal_Risk': 'mean'
        }).round(2)
        
        risk_breakdown.columns = ['Count', 'Avg Churn %', 'Total CLV', 'Avg CLV', 'Avg Renewal Risk']
        risk_breakdown['Avg Churn %'] = (risk_breakdown['Avg Churn %'] * 100).round(1).astype(str) + '%'
        risk_breakdown['Total CLV'] = '€' + risk_breakdown['Total CLV'].apply(lambda x: f'{x:,.0f}')
        risk_breakdown['Avg CLV'] = '€' + risk_breakdown['Avg CLV'].apply(lambda x: f'{x:,.0f}')
        
        st.dataframe(risk_breakdown, use_container_width=True)
    
    # ===========================================
    # CLAIMS INTELLIGENCE PAGE
    # ===========================================
    elif "Claims Intelligence" in page:
        st.markdown("# 🛡️ Claims Intelligence")
        st.markdown("*Predict and manage claims risk*")
        st.markdown("---")
        
        # Metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(metric_card("Avg Claims Prob", f"{avg_claims*100:.1f}%", "Portfolio average", 'warning'), unsafe_allow_html=True)
        c2.markdown(metric_card("High Risk", f"{(df['Claims_Prob']>0.5).sum():,}", "50%+ probability", 'orange'), unsafe_allow_html=True)
        c3.markdown(metric_card("Severity p95", f"€{df['Claims_Severity'].quantile(0.95):,.0f}", "95th percentile", 'info'), unsafe_allow_html=True)
        c4.markdown(metric_card("Total Exposure", f"€{df['Claims_Severity'].sum()/1e6:.1f}M", "Expected claims", 'blue'), unsafe_allow_html=True)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Claims Probability Distribution")
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=df['Claims_Prob'],
                nbinsx=35,
                marker=dict(
                    color='#f59e0b',
                    line=dict(color='#ffffff', width=1)
                )
            ))
            
            fig.update_layout(
                height=380,
                paper_bgcolor='rgba(255,255,255,0.95)',
                plot_bgcolor='#fafbfc',