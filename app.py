"""
🏢 Insurance Customer Success Platform - PRODUCTION VERSION
==============================================================
Streamlit app integrating all 8 models from Auto_Analysis_Notebook.ipynb
with enhanced RAG capabilities for intelligent customer management

Models Integrated:
- Model 1: Customer Retention (Churn Risk) - ROC AUC 0.715
- Model 2: Claims Frequency - ROC AUC 0.923 ⭐
- Model 3: Claims Severity - Segment-based estimation
- Model 4: Customer Lifetime Value - €244 avg, €25.8M portfolio
- Model 5: Renewal Risk Scoring - 25.9% high-risk flagged
- Model 6: Pricing Optimization - 14% underpriced identified
- Model 7: Customer Journey (Segmentation) - 4 strategic segments
- Model 8: Channel Attribution - Agent ROI 752% vs Broker 297%

Author: Auto Analysis Team
Date: December 2025
Version: 2.0 (Production)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import sys
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Customer Success Platform | Insurance Analytics",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for professional look
st.markdown("""
<style>
    /* Premium Glassmorphism Theme */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif;
        background: radial-gradient(circle at top right, #1e1e2f 0%, #121212 100%);
        color: #e0e0e0;
    }
    
    [data-testid="stHeader"] {
        background: rgba(0,0,0,0);
    }
    
    /* Unique Card Design */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 1rem;
        transition: transform 0.3s ease;
    }
    
    .glass-card:hover {
        transform: scale(1.02);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Custom Sidebar */
    [data-testid="stSidebar"] {
        background-color: rgba(15, 15, 25, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        letter-spacing: -1px;
        background: linear-gradient(90deg, #667eea, #764ba2, #6b8dd6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
        text-align: left;
    }
    
    .sub-header {
        font-size: 1.2rem;
        color: #888;
        font-weight: 400;
        margin-bottom: 3rem;
        text-align: left;
    }
    
    /* Metrics Override */
    [data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        font-weight: 700 !important;
    }
    
    /* Risk Levels - More subtle & professional */
    .risk-pill {
        padding: 4px 12px;
        border-radius: 50px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    
    .pill-critical { background: rgba(255, 68, 68, 0.2); color: #ff4444; border: 1px solid #ff4444; }
    .pill-high { background: rgba(255, 153, 51, 0.2); color: #ff9933; border: 1px solid #ff9933; }
    .pill-low { background: rgba(0, 200, 81, 0.2); color: #00c851; border: 1px solid #00c851; }

    /* Hide standard Streamlit elements for uniqueness */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# DATA LOADING & CACHING
# =============================================================================

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_data():
    """Load model predictions from Auto_Analysis_Notebook export"""
    try:
        data_path = Path('model_outputs/rag_model_predictions.csv')
        
        if not data_path.exists():
            st.error(f"❌ Data file not found: {data_path}")
            st.info("💡 Please run the export cell in Auto_Analysis_Notebook.ipynb first")
            st.code("""
# In your notebook, run:
df_export = df.copy()
df_export.to_csv('model_outputs/rag_model_predictions.csv', index=False)
print('✅ Data exported successfully!')
            """)
            return None
            
        df = pd.read_csv(data_path)
        
        # Data validation
        required_cols = ['ID', 'Churn_Probability', 'Claims_Probability', 
                        'Customer_Lifetime_Value', 'Customer_Segment']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            st.error(f"❌ Missing required columns: {missing}")
            return None
        
        # Add computed fields if not present
        if 'Churn_Risk_Category' not in df.columns:
            df['Churn_Risk_Category'] = pd.cut(
                df['Churn_Probability'],
                bins=[0, 0.3, 0.5, 0.7, 1.0],
                labels=['Low', 'Moderate', 'High', 'Critical']
            )
        
        if 'CLV_Category' not in df.columns:
            df['CLV_Category'] = pd.cut(
                df['Customer_Lifetime_Value'],
                bins=[-float('inf'), 0, 200, 400, 600, float('inf')],
                labels=['Negative', 'Low', 'Medium', 'High', 'Very High']
            )
        
        # Add priority score
        df['Priority_Score'] = (
            df['Churn_Probability'] * 0.4 +
            (df['Customer_Lifetime_Value'] / df['Customer_Lifetime_Value'].max()) * 0.3 +
            df['Claims_Probability'] * 0.3
        )
        
        return df
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None

@st.cache_resource
def load_enhanced_faiss():
    """Load RAG system with enhanced predictions"""
    try:
        from langchain_community.vectorstores import FAISS
        from langchain_community.embeddings import HuggingFaceEmbeddings
        
        # Try both possible locations
        possible_paths = [
            Path('project_structure/enhanced_faiss_index'),
            Path('enhanced_faiss_index')
        ]
        
        index_path = None
        for path in possible_paths:
            if path.exists():
                index_path = path
                break
        
        if index_path is None:
            return None, "Index directory not found. Please run project_structure/rag.ipynb Steps 1-6."
        
        # Load with embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        faiss_db = FAISS.load_local(
            str(index_path),
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
        return faiss_db, None
    except Exception as e:
        return None, str(e)

@st.cache_data
def calculate_portfolio_metrics(df):
    """Calculate comprehensive portfolio metrics"""
    total = len(df)
    
    metrics = {
        # Customer counts
        'total_customers': total,
        'active_customers': total,  # Assuming all are active
        
        # Churn metrics
        'critical_churn': len(df[df['Churn_Probability'] > 0.7]),
        'high_churn': len(df[df['Churn_Probability'] > 0.5]),
        'churn_rate_avg': df['Churn_Probability'].mean(),
        
        # Value metrics
        'total_clv': df['Customer_Lifetime_Value'].sum(),
        'avg_clv': df['Customer_Lifetime_Value'].mean(),
        'median_clv': df['Customer_Lifetime_Value'].median(),
        'negative_clv_count': len(df[df['Customer_Lifetime_Value'] < 0]),
        'negative_clv_total': df[df['Customer_Lifetime_Value'] < 0]['Customer_Lifetime_Value'].sum(),
        
        # Risk metrics
        'high_claims_risk': len(df[df['Claims_Probability'] > 0.5]),
        'expected_claims_cost': df['Expected_Claims_Cost'].sum() if 'Expected_Claims_Cost' in df.columns else 0,
        'underpriced_policies': len(df[df['Pricing_Adequacy'] < 1.0]) if 'Pricing_Adequacy' in df.columns else int(total * 0.14),
        
        # Segment distribution
        'protect_count': len(df[df['Customer_Segment'] == 'PROTECT']),
        'develop_count': len(df[df['Customer_Segment'] == 'DEVELOP']),
        'manage_count': len(df[df['Customer_Segment'] == 'MANAGE']),
        'exit_count': len(df[df['Customer_Segment'] == 'EXIT']),
        
        # Channel performance (from research insights)
        'agent_roi': 752,
        'broker_roi': 297,
        'agent_clv': 1278,
        'broker_clv': 795,
        
        # At-risk value
        'at_risk_clv': df[df['Churn_Probability'] > 0.5]['Customer_Lifetime_Value'].sum(),
        'critical_risk_clv': df[df['Churn_Probability'] > 0.7]['Customer_Lifetime_Value'].sum(),
    }
    
    return metrics

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_risk_badge(prob, metric_type='churn'):
    """Generate HTML badge for risk level"""
    if metric_type == 'churn':
        if prob > 0.7:
            return f'<div class="risk-critical">Critical ({prob:.1%})</div>'
        elif prob > 0.5:
            return f'<div class="risk-high">High ({prob:.1%})</div>'
        elif prob > 0.3:
            return f'<div class="risk-moderate">Moderate ({prob:.1%})</div>'
        else:
            return f'<div class="risk-low">Low ({prob:.1%})</div>'
    else:  # claims
        if prob > 0.6:
            return f'<div class="risk-high">High ({prob:.1%})</div>'
        elif prob > 0.4:
            return f'<div class="risk-moderate">Moderate ({prob:.1%})</div>'
        else:
            return f'<div class="risk-low">Low ({prob:.1%})</div>'

def get_segment_badge(segment):
    """Generate HTML badge for customer segment"""
    badges = {
        'PROTECT': '<div class="segment-protect">🛡️ PROTECT</div>',
        'DEVELOP': '<div class="segment-develop">📈 DEVELOP</div>',
        'MANAGE': '<div class="segment-manage">⚙️ MANAGE</div>',
        'EXIT': '<div class="segment-exit">🚪 EXIT</div>'
    }
    return badges.get(segment, f'<div class="segment-develop">{segment}</div>')

def get_recommendation(customer):
    """Generate AI-powered recommendation based on all 8 models"""
    segment = customer['Customer_Segment']
    churn = customer['Churn_Probability']
    claims = customer['Claims_Probability']
    clv = customer['Customer_Lifetime_Value']
    tenure = customer.get('Seniority', 0)
    
    # Critical interventions (URGENT)
    if segment == 'PROTECT' and churn > 0.7:
        return {
            'priority': '🚨 URGENT',
            'action': 'Immediate Executive Intervention',
            'details': f'High-value customer (€{clv:.0f} CLV) at critical churn risk. Schedule C-level call within 24 hours. Offer: VIP loyalty program, premium discount (up to 15%), dedicated account manager.',
            'expected_impact': f'Saving this customer protects €{clv * 3:.0f} in 3-year value',
            'timeline': 'Next 24 hours',
            'color': 'critical'
        }
    
    if segment == 'PROTECT' and churn > 0.5:
        return {
            'priority': '⚠️ HIGH',
            'action': 'Retention Campaign - High Priority',
            'details': f'Valued customer showing churn signals. Personal outreach within 7 days. Offer: 10% loyalty discount, policy review, enhanced coverage options.',
            'expected_impact': f'€{clv * 2.5:.0f} at stake over next 2 years',
            'timeline': 'Within 7 days',
            'color': 'high'
        }
    
    # Growth opportunities
    if segment == 'DEVELOP' and claims < 0.3 and tenure > 2:
        return {
            'priority': '💎 OPPORTUNITY',
            'action': 'Cross-Sell / Upsell Campaign',
            'details': f'Low-risk, stable customer ready for growth. Current CLV: €{clv:.0f}. Offer: Multi-policy discount (home, life insurance), premium tier upgrade, refer-a-friend bonus.',
            'expected_impact': f'Potential CLV increase to €{clv * 1.5:.0f} (+50%)',
            'timeline': 'Next renewal cycle',
            'color': 'opportunity'
        }
    
    # Risk mitigation
    if claims > 0.6:
        pricing_adequate = customer.get('Pricing_Adequacy', 1.0) >= 1.0
        if not pricing_adequate:
            return {
                'priority': '⚠️ RISK',
                'action': 'Pricing Correction Required',
                'details': f'High claims risk ({claims:.1%}) with inadequate pricing. Expected claims: €{customer.get("Expected_Claims_Cost", 0):.0f}. Action: Premium adjustment at renewal or add higher deductible.',
                'expected_impact': 'Protect profitability, reduce expected loss by 20%',
                'timeline': 'At renewal',
                'color': 'caution'
            }
        else:
            return {
                'priority': '👁️ MONITOR',
                'action': 'Claims Risk Management',
                'details': f'High claims probability ({claims:.1%}). Pricing adequate. Provide: Safe driving tips, telematics offer, defensive driving course discount.',
                'expected_impact': 'Reduce claims frequency by 10-15%',
                'timeline': 'Ongoing',
                'color': 'monitor'
            }
    
    # Early tenure risk
    if tenure <= 3 and churn > 0.4:
        return {
            'priority': '🎯 FOCUS',
            'action': 'Early Tenure Engagement',
            'details': f'Customer in critical Years 1-3 period (26.5% avg churn). Increase touchpoints: welcome call, 3-month check-in, 6-month policy review, first renewal incentive.',
            'expected_impact': 'Reduce early-stage churn by 30%',
            'timeline': 'Monthly touchpoints',
            'color': 'focus'
        }
    
    # EXIT segment
    if segment == 'EXIT':
        return {
            'priority': '🔄 REASSESS',
            'action': 'Portfolio Optimization',
            'details': f'Negative CLV customer (€{clv:.0f}). High acquisition cost not recovered. Consider: No retention investment, natural attrition, or significant pricing correction.',
            'expected_impact': 'Reallocate resources to high-value customers',
            'timeline': 'At renewal',
            'color': 'low'
        }
    
    # Default
    return {
        'priority': '✅ ROUTINE',
        'action': 'Standard Service Protocol',
        'details': f'Customer stable with {segment} profile. Continue regular service, automated renewal reminders, annual policy review.',
        'expected_impact': 'Maintain satisfaction and retention',
        'timeline': 'Standard schedule',
        'color': 'routine'
    }

# =============================================================================
# MAIN APP
# =============================================================================

def main():
    """Main application orchestrator"""
    
    # Header with branding
    st.markdown('<h1 class="main-header">🎯 Customer Success Platform</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Insurance Analytics | 8 Models | €25.8M Portfolio</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load data
    df = load_data()
    if df is None:
        st.stop()
    
    # Calculate metrics
    metrics = calculate_portfolio_metrics(df)
    
    # Sidebar navigation
    st.sidebar.image("https://via.placeholder.com/200x80/667eea/ffffff?text=Insurance+Co", use_container_width=True)
    st.sidebar.markdown("---")
    
    st.sidebar.header("🧭 Navigation")
    page = st.sidebar.radio(
        "",
        [
            "📊 Executive Command Center",
            "👥 Customer 360° Intelligence",
            "🎯 Priority Action Center",
            "🤖 AI Customer Assistant",
            "📈 Model Performance Hub",
            "💡 Strategic Insights"
        ],
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.header("🔍 Global Filters")
    
    # Segment filter
    segments = ['All'] + sorted(df['Customer_Segment'].dropna().unique().tolist())
    selected_segment = st.sidebar.selectbox("📦 Customer Segment", segments)
    
    # Risk filter
    risk_levels = ['All', 'Critical', 'High', 'Moderate', 'Low']
    selected_risk = st.sidebar.selectbox("⚠️ Churn Risk", risk_levels)
    
    # Channel filter
    if 'Distribution_channel' in df.columns:
        channels = ['All'] + sorted(df['Distribution_channel'].dropna().unique().tolist())
        selected_channel = st.sidebar.selectbox("🤝 Channel", channels)
    else:
        selected_channel = 'All'
    
    # Apply filters
    filtered_df = df.copy()
    if selected_segment != 'All':
        filtered_df = filtered_df[filtered_df['Customer_Segment'] == selected_segment]
    if selected_risk != 'All':
        filtered_df = filtered_df[filtered_df['Churn_Risk_Level'] == selected_risk]
    if selected_channel != 'All' and 'Distribution_channel' in df.columns:
        filtered_df = filtered_df[filtered_df['Distribution_channel'] == selected_channel]
    
    st.sidebar.markdown("---")
    st.sidebar.metric("🎯 Filtered Customers", f"{len(filtered_df):,}")
    st.sidebar.metric("📊 Total Portfolio", f"{len(df):,}")
    st.sidebar.metric("💰 Filtered CLV", f"€{filtered_df['Customer_Lifetime_Value'].sum()/1e6:.2f}M")
    
    # Quick stats
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚡ Quick Stats")
    st.sidebar.metric("Critical Risk", f"{metrics['critical_churn']:,}", 
                     f"{metrics['critical_churn']/metrics['total_customers']*100:.1f}%")
    st.sidebar.metric("At-Risk Value", f"€{metrics['at_risk_clv']/1e6:.2f}M")
    
    # Route to pages
    if page == "📊 Executive Command Center":
        show_executive_dashboard(filtered_df, metrics)
    elif page == "👥 Customer 360° Intelligence":
        show_customer_intelligence(filtered_df)
    elif page == "🎯 Priority Action Center":
        show_action_center(filtered_df, metrics)
    elif page == "🤖 AI Customer Assistant":
        show_smart_search(df)
    elif page == "📈 Model Performance Hub":
        show_model_performance(df)
    elif page == "💡 Strategic Insights":
        show_strategic_insights(df, metrics)

# =============================================================================
# PAGE 1: EXECUTIVE COMMAND CENTER
# =============================================================================

def show_executive_dashboard(df, metrics):
    """Modern, intuitive landing page with high-impact visualizations"""
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    with col1:
        st.header("🏢 Portfolio Health Index")
        st.markdown("Global view of retention, revenue, and actuarial risk.")
    with col2:
        st.metric("Total Portfolio CLV", f"€{metrics['total_clv']/1e6:.1f}M", f"€{metrics['avg_clv']:.0f} avg")
    st.markdown('</div>', unsafe_allow_html=True)

    # Informative Layout Row 1: The Core Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="glass-card">
            <p class="metric-label">Retention Signal</p>
            <p class="metric-value" style="color:#00c851;">{100 - metrics['churn_rate_avg']*100:.1f}%</p>
            <p style="font-size:0.8rem; color:#888;">{metrics['total_customers'] - metrics['high_churn']} Stable Policies</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="glass-card">
            <p class="metric-label">Danger Zone Exposure</p>
            <p class="metric-value" style="color:#ffbb33;">€{metrics['at_risk_clv']/1e3:.0f}K</p>
            <p style="font-size:0.8rem; color:#888;">{metrics['high_churn']} At-Risk Customers</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="glass-card">
            <p class="metric-label">Claims Frequency</p>
            <p class="metric-value" style="color:#33b5e5;">{metrics['high_claims_risk'] / metrics['total_customers'] * 100:.1f}%</p>
            <p style="font-size:0.8rem; color:#888;">High Risk Flagged</p>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="glass-card">
            <p class="metric-label">Strategic ROI</p>
            <p class="metric-value" style="color:#667eea;">752%</p>
            <p style="font-size:0.8rem; color:#888;">Agent Channel Peak</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 2: Advanced Visualizations
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("🌋 Risk-Value Landscape")
        st.markdown("Mapping customer density: Where is the money vs. the risk?")
        
        # 2D Density Plot for a unique look
        fig = px.density_heatmap(
            df.sample(min(10000, len(df))), 
            x="Customer_Lifetime_Value", 
            y="Churn_Probability",
            nbinsx=30, nbinsy=30,
            color_continuous_scale="Viridis",
            labels={'Customer_Lifetime_Value': 'Value (€)', 'Churn_Probability': 'Risk (%)'}
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color="#888",
            height=400,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("📊 Segment Efficiency")
        st.markdown("Portfolio composition by strategic value segments.")
        
        # Unique Sunburst plot instead of Pie
        fig = px.sunburst(
            df, 
            path=['Distribution_channel', 'Customer_Segment'], 
            values='Customer_Lifetime_Value',
            color='Customer_Segment',
            color_discrete_map={'PROTECT': '#00C851', 'DEVELOP': '#33b5e5', 'MANAGE': '#ffbb33', 'EXIT': '#ff4444'}
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=400,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Row 3: Actuarial Insight
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📑 The 'Danger Zone' Analytics (Years 1-3)")
    st.markdown("Analyzing why the first 1,000 days are critical for retention.")
    
    # Line chart showing Risk vs. Tenure
    tenure_stats = df.groupby('Seniority').agg({'Churn_Probability': 'mean', 'ID': 'count'}).reset_index()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(x=tenure_stats['Seniority'], y=tenure_stats['Churn_Probability'], name="Avg Churn Risk", line=dict(color='#ff4444', width=3)),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Bar(x=tenure_stats['Seniority'], y=tenure_stats['ID'], name="Customer Volume", marker_color='rgba(102, 126, 234, 0.3)'),
        secondary_y=True,
    )
    
    fig.add_vrect(x0=0, x1=3, fillcolor="red", opacity=0.1, layer="below", line_width=0, annotation_text="DANGER ZONE")
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color="#888",
        height=400,
        showlegend=False,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# PAGE 2: CUSTOMER 360° INTELLIGENCE
# =============================================================================

def show_customer_intelligence(df):
    """Detailed customer intelligence with 360° view"""
    
    st.header("👥 Customer 360° Intelligence")
    st.markdown("Complete customer profiles with AI-powered insights")
    
    # Customer search
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        customer_id = st.selectbox(
            "🔍 Select Customer",
            options=sorted(df['ID'].unique()),
            format_func=lambda x: f"Customer #{x}"
        )
    
    with col2:
        if st.button("🎲 Random", type="secondary", use_container_width=True):
            customer_id = np.random.choice(df['ID'].unique())
            st.rerun()
    
    with col3:
        if st.button("🔝 Top CLV", type="secondary", use_container_width=True):
            customer_id = df.nlargest(1, 'Customer_Lifetime_Value')['ID'].iloc[0]
            st.rerun()
    
    # Get customer
    customer = df[df['ID'] == customer_id].iloc[0]
    
    # Customer header card
    st.markdown("---")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        risk_class = {
            'Low': 'risk-low', 'Moderate': 'risk-moderate',
            'High': 'risk-high', 'Critical': 'risk-critical'
        }.get(customer['Churn_Risk_Level'], 'risk-moderate')
        st.markdown(f'<div class="{risk_class}">{customer["Churn_Risk_Level"]} Risk</div>', 
                   unsafe_allow_html=True)
    
    with col2:
        segment_class = {
            'PROTECT': 'segment-protect', 'DEVELOP': 'segment-develop',
            'MANAGE': 'segment-manage', 'EXIT': 'segment-exit'
        }.get(customer['Customer_Segment'], 'segment-manage')
        st.markdown(f'<div class="{segment_class}">{customer["Customer_Segment"]}</div>', 
                   unsafe_allow_html=True)
    
    with col3:
        st.metric("CLV", f"€{customer['Customer_Lifetime_Value']:.0f}")
    
    with col4:
        st.metric("Premium", f"€{customer['Premium']:.2f}")
    
    with col5:
        st.metric("Seniority", f"{customer['Seniority']} yrs")
    
    with col6:
        st.metric("Claims History", f"{customer['N_claims_history']:.0f}")
    
    # Detailed sections
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🎯 Risk Assessment")
        st.progress(customer['Churn_Probability'], text=f"Churn Risk: {customer['Churn_Probability']:.1%}")
        st.progress(customer['Claims_Probability'], text=f"Claims Risk: {customer['Claims_Probability']:.1%}")
        st.progress(customer['Renewal_Risk_Score'], text=f"Renewal Risk: {customer['Renewal_Risk_Score']:.1%}")
        
        st.write(f"**Churn Level:** {customer['Churn_Risk_Level']}")
        st.write(f"**Claims Level:** {customer['Claims_Risk_Level']}")
    
    with col2:
        st.markdown("### 💰 Value & Economics")
        st.write(f"**CLV:** €{customer['Customer_Lifetime_Value']:.2f}")
        st.write(f"**CLV Segment:** {customer['CLV_Segment']}")
        st.write(f"**Premium:** €{customer['Premium']:.2f}")
        st.write(f"**Expected Claims Cost:** €{customer['Expected_Claims_Cost']:.2f}")
        st.write(f"**Pricing Adequacy:** {customer['Pricing_Adequacy']:.2f}x")
        
        if customer['Is_Underpriced'] == 1:
            st.error("⚠️ Policy is underpriced")
        else:
            st.success("✅ Pricing adequate")
    
    with col3:
        st.markdown("### 📊 Policy Details")
        st.write(f"**Type:** {customer['Type_risk']}")
        st.write(f"**Area:** {customer['Area']}")
        st.write(f"**Channel:** {customer['Distribution_channel']}")
        st.write(f"**Payment:** {customer['Payment']}")
        st.write(f"**Vehicle Value:** €{customer['Value_vehicle']:.0f}")
        st.write(f"**Second Driver:** {customer['Second_driver']}")
    
    # AI-powered recommendation
    st.markdown("---")
    st.markdown("### 🤖 AI-Powered Recommendation")
    
    recommendation = get_recommendation(customer)
    
    priority_colors = {
        'high': '🔴', 'opportunity': '💎', 'caution': '⚠️',
        'monitor': '👁️', 'focus': '🎯', 'low': '🔵', 'routine': '✅'
    }
    
    rec_color = priority_colors.get(recommendation['color'], '✅')
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f"### {rec_color} {recommendation['priority']}")
        st.markdown(f"**Action:** {recommendation['action']}")
        st.markdown(f"**Timeline:** {recommendation['timeline']}")
    
    with col2:
        st.info(recommendation['details'])
        st.success(f"**Expected Impact:** {recommendation['expected_impact']}")
    
    # Comparison charts
    st.markdown("---")
    st.subheader("📊 Portfolio Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df['Churn_Probability'], name='Portfolio', 
                                   marker_color='lightblue', opacity=0.7))
        fig.add_vline(x=customer['Churn_Probability'], line_dash="dash", line_color="red",
                     annotation_text=f"This Customer: {customer['Churn_Probability']:.1%}")
        fig.update_layout(title="Churn Probability Distribution", height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df['Customer_Lifetime_Value'], name='Portfolio',
                                   marker_color='lightgreen', opacity=0.7))
        fig.add_vline(x=customer['Customer_Lifetime_Value'], line_dash="dash", line_color="red",
                     annotation_text=f"This Customer: €{customer['Customer_Lifetime_Value']:.0f}")
        fig.update_layout(title="CLV Distribution", height=300)
        st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# PAGE 3: PRIORITY ACTION CENTER
# =============================================================================

def show_action_center(df, metrics):
    """Prioritized action lists with exportable data"""
    
    st.header("🎯 Priority Action Center")
    st.markdown("Actionable customer lists prioritized by risk and value")
    
    # Quick metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🚨 Critical", f"{metrics['critical_churn']:,}")
    with col2:
        st.metric("💎 PROTECT", f"{metrics['protect_count']:,}")
    with col3:
        st.metric("📈 DEVELOP", f"{metrics['develop_count']:,}")
    with col4:
        st.metric("⚠️ High Claims", f"{metrics['high_claims_risk']:,}")
    
    # Action tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🚨 Critical Interventions",
        "💎 PROTECT Retention",
        "📈 DEVELOP Growth",
        "⚠️ Claims Risk Management"
    ])
    
    with tab1:
        st.subheader("🚨 Critical Churn Risk - Immediate Action")
        
        critical = df[df['Churn_Risk_Level'] == 'Critical'].sort_values(
            'Customer_Lifetime_Value', ascending=False
        )
        
        st.metric("Customers", len(critical))
        st.metric("Total CLV at Risk", f"€{critical['Customer_Lifetime_Value'].sum()/1e6:.2f}M")
        
        if len(critical) > 0:
            display_df = critical[['ID', 'Customer_Segment', 'Churn_Probability', 
                                  'Customer_Lifetime_Value', 'Premium', 'Seniority']].head(100)
            
            st.dataframe(
                display_df.style.format({
                    'Churn_Probability': '{:.1%}',
                    'Customer_Lifetime_Value': '€{:.0f}',
                    'Premium': '€{:.2f}'
                }),
                use_container_width=True,
                height=400
            )
            
            csv = critical.to_csv(index=False)
            st.download_button(
                "📥 Download Critical Risk List",
                csv,
                f"critical_risk_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv",
                type="primary"
            )
        else:
            st.success("✅ No customers in critical risk!")
    
    with tab2:
        st.subheader("💎 PROTECT Segment - High-Value Retention")
        
        protect = df[df['Customer_Segment'] == 'PROTECT'].sort_values(
            'Customer_Lifetime_Value', ascending=False
        )
        
        st.metric("Customers", len(protect))
        st.metric("Total Value", f"€{protect['Customer_Lifetime_Value'].sum()/1e6:.2f}M")
        
        if len(protect) > 0:
            display_df = protect[['ID', 'Churn_Risk_Level', 'Customer_Lifetime_Value',
                                 'Premium', 'Seniority', 'CLV_Segment']].head(100)
            
            st.dataframe(
                display_df.style.format({
                    'Customer_Lifetime_Value': '€{:.0f}',
                    'Premium': '€{:.2f}'
                }),
                use_container_width=True,
                height=400
            )
            
            csv = protect.to_csv(index=False)
            st.download_button(
                "📥 Download PROTECT List",
                csv,
                f"protect_segment_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv"
            )
    
    with tab3:
        st.subheader("📈 DEVELOP Segment - Growth Potential")
        
        develop = df[df['Customer_Segment'] == 'DEVELOP'].sort_values(
            'Customer_Lifetime_Value', ascending=False
        )
        
        st.metric("Customers", len(develop))
        potential_value = (develop['Customer_Lifetime_Value'].mean() * 1.5 * len(develop)) - develop['Customer_Lifetime_Value'].sum()
        st.metric("Growth Potential (50% uplift)", f"€{potential_value/1e6:.2f}M")
        
        if len(develop) > 0:
            display_df = develop[['ID', 'Customer_Lifetime_Value', 'Premium',
                                 'Seniority', 'Type_risk', 'CLV_Segment']].head(100)
            
            st.dataframe(
                display_df.style.format({
                    'Customer_Lifetime_Value': '€{:.0f}',
                    'Premium': '€{:.2f}'
                }),
                use_container_width=True,
                height=400
            )
            
            csv = develop.to_csv(index=False)
            st.download_button(
                "📥 Download DEVELOP List",
                csv,
                f"develop_segment_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv"
            )
    
    with tab4:
        st.subheader("⚠️ High Claims Risk - Monitoring Required")
        
        high_claims = df[df['Claims_Risk_Level'].isin(['High', 'Very High'])].sort_values(
            'Claims_Probability', ascending=False
        )
        
        st.metric("Customers", len(high_claims))
        st.metric("Expected Claims Cost", f"€{high_claims['Expected_Claims_Cost'].sum()/1e6:.2f}M")
        
        if len(high_claims) > 0:
            display_df = high_claims[['ID', 'Claims_Risk_Level', 'Claims_Probability',
                                     'Expected_Claims_Cost', 'Pricing_Adequacy',
                                     'N_claims_history']].head(100)
            
            st.dataframe(
                display_df.style.format({
                    'Claims_Probability': '{:.1%}',
                    'Expected_Claims_Cost': '€{:.2f}',
                    'Pricing_Adequacy': '{:.2f}x'
                }),
                use_container_width=True,
                height=400
            )
            
            csv = high_claims.to_csv(index=False)
            st.download_button(
                "📥 Download High Claims Risk List",
                csv,
                f"high_claims_risk_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv"
            )

# =============================================================================
# PAGE 4: INTELLIGENT CUSTOMER ASSISTANT (RAG + OLLAMA)
# =============================================================================

def query_ollama(prompt, context=""):
    """Query Ollama LLM for natural language response with enhanced prompting"""
    try:
        import requests
        
        # Build enhanced prompt for insurance domain
        system_prompt = """You are an expert insurance analyst helping customer success teams. 
        Analyze customer data and provide actionable insights focused on:
        - Churn prevention and retention strategies
        - Value optimization and upselling opportunities  
        - Risk assessment and pricing adequacy
        - Personalized action recommendations
        
        Be concise, specific, and action-oriented. Focus on business impact."""
        
        full_prompt = f"""{system_prompt}

{context}

Agent Question: {prompt}

Provide a clear, actionable analysis:"""
        
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': 'llama2',
                'prompt': full_prompt,
                'stream': False,
                'options': {
                    'temperature': 0.7,
                    'num_predict': 600,
                    'top_p': 0.9,
                    'top_k': 40
                }
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json().get('response', '').strip()
            return result if result else None
        return None
    except Exception as e:
        return None

def generate_analysis_fallback(customer_data, query_intent):
    """Generate intelligent structured analysis without Ollama"""
    
    if len(customer_data) == 0:
        return """
## ❌ No Matching Customers Found

**Try these search strategies:**
- Use broader criteria (e.g., 'high risk' instead of 'critical risk')
- Remove some filters to expand results
- Try quick start buttons for common searches
- Use specific attributes: 'urban', 'agent channel', 'high value'

**Popular queries:**
- "Show customers with churn risk above 50%"
- "Find PROTECT segment customers"
- "Display underpriced policies with high claims risk"
        """
    
    customer = customer_data.iloc[0]
    num_customers = len(customer_data)
    
    # Analyze query intent
    query_lower = query_intent.lower()
    focus_area = "retention" if any(word in query_lower for word in ['churn', 'retention', 'leave', 'urgent']) else \
                 "value" if any(word in query_lower for word in ['value', 'clv', 'protect', 'best']) else \
                 "risk" if any(word in query_lower for word in ['risk', 'claims', 'underpriced']) else \
                 "growth" if any(word in query_lower for word in ['upsell', 'growth', 'opportunity', 'cross-sell']) else \
                 "general"
    
    # Build context-aware analysis
    analysis = f"""
## 🎯 Analysis Results

**Query:** {query_intent}
**Found:** {num_customers} matching customer{'s' if num_customers != 1 else ''}
**Focus:** {focus_area.title()} Strategy

---

### 📊 Top Match: Customer {customer['ID']}

**Segment:** {customer['Customer_Segment']} | **Churn Risk:** {customer['Churn_Probability']:.1%} ({customer['Churn_Risk_Level']}) | **CLV:** €{customer['Customer_Lifetime_Value']:.2f}

"""

    # Context-aware insights
    if focus_area == "retention":
        analysis += f"""
#### 🚨 Retention Priority Analysis

This customer shows **{customer['Churn_Risk_Level'].lower()} churn risk** ({customer['Churn_Probability']:.1%} probability). 

**Key Risk Factors:**
- Claims history: {customer['N_claims_history']} claims (ratio: {customer['R_Claims_history']:.2f})
- Seniority: {customer['Seniority']} years
- Channel: {customer['Distribution_channel']}
- Renewal risk score: {customer['Renewal_Risk_Score']:.3f}

**Immediate Actions:**
1. **Call within 48 hours** - Personal outreach from senior agent
2. **Review pricing** - Current premium €{customer['Premium']:.2f}, adequacy {customer['Pricing_Adequacy']:.2f}x
3. **Loyalty incentives** - Offer renewal discount or added benefits
4. **Address service gaps** - Review claims experience and satisfaction

**Expected Impact:** Reduce churn probability by 20-30%, protect €{customer['Customer_Lifetime_Value']:.2f} CLV
"""
    elif focus_area == "value":
        analysis += f"""
#### 💎 High-Value Customer Protection

This customer represents **€{customer['Customer_Lifetime_Value']:.2f} lifetime value** in the {customer['CLV_Segment']} tier.

**Value Profile:**
- Premium: €{customer['Premium']:.2f}/year
- Vehicle value: €{customer['Value_vehicle']:.0f}
- Claims efficiency: {customer['Claims_Probability']:.1%} probability
- Pricing adequacy: {customer['Pricing_Adequacy']:.2f}x (profitable)

**Protection Strategy:**
1. **VIP treatment** - Dedicated account manager
2. **Proactive service** - Annual policy review call
3. **Exclusive benefits** - Premium customer perks
4. **Loyalty program** - Long-term value recognition

**Expected Impact:** Increase retention 95%+, potential CLV growth 15-20% through upselling
"""
    elif focus_area == "risk":
        analysis += f"""
#### ⚠️ Risk & Pricing Assessment

This policy shows **{customer['Claims_Risk_Level']} claims risk** ({customer['Claims_Probability']:.1%} probability).

**Risk Indicators:**
- Expected claims cost: €{customer['Expected_Claims_Cost']:.2f}
- Current premium: €{customer['Premium']:.2f}/year
- Pricing adequacy: {customer['Pricing_Adequacy']:.2f}x {'(UNDERPRICED ⚠️)' if customer['Pricing_Adequacy'] < 1.0 else '(adequate)'}
- Claims history: {customer['N_claims_history']} claims
- Vehicle type: {customer['Type_risk']}

**Risk Management Actions:**
1. **{'Premium adjustment' if customer['Pricing_Adequacy'] < 1.0 else 'Maintain pricing'}** - {'Increase to match risk profile' if customer['Pricing_Adequacy'] < 1.0 else 'Current pricing is adequate'}
2. **Claims prevention** - Share safe driving tips, telematics discount offer
3. **Coverage review** - Ensure appropriate deductibles and limits
4. **Monitoring** - Flag for quarterly risk reassessment

**Expected Impact:** {'Improve profitability 20-30%' if customer['Pricing_Adequacy'] < 1.0 else 'Maintain healthy margin'}, reduce loss ratio
"""
    elif focus_area == "growth":
        analysis += f"""
#### 📈 Growth & Upsell Opportunities

This customer shows strong potential for portfolio expansion.

**Opportunity Profile:**
- Current premium: €{customer['Premium']:.2f}/year
- CLV: €{customer['Customer_Lifetime_Value']:.2f} (room to grow)
- Seniority: {customer['Seniority']} years (established relationship)
- Claims: {customer['N_claims_history']} (stable)
- Churn risk: {customer['Churn_Probability']:.1%} (low)

**Growth Strategies:**
1. **Cross-sell** - Home, life, or multi-vehicle insurance bundles
2. **Premium upgrade** - Enhanced coverage options
3. **Family expansion** - Household member policies
4. **Commercial add-on** - Business insurance if applicable

**Expected Impact:** 40-60% CLV increase, stronger customer relationship, improved retention
"""
    else:
        # General analysis
        rec = get_recommendation(customer)
        analysis += f"""
#### 📋 Customer Profile

**Risk Assessment:**
- Churn: {customer['Churn_Probability']:.1%} ({customer['Churn_Risk_Level']})
- Claims: {customer['Claims_Probability']:.1%} ({customer['Claims_Risk_Level']})
- Renewal risk: {customer['Renewal_Risk_Score']:.3f}

**Value Metrics:**
- CLV: €{customer['Customer_Lifetime_Value']:.2f} ({customer['CLV_Segment']})
- Premium: €{customer['Premium']:.2f}/year
- Pricing adequacy: {customer['Pricing_Adequacy']:.2f}x

**Policy Details:**
- Type: {customer['Type_risk']} | Area: {customer['Area']}
- Channel: {customer['Distribution_channel']}
- Seniority: {customer['Seniority']} years
- Claims: {customer['N_claims_history']}

**Recommended Action:**
{rec['priority']} - {rec['action']}

{rec['details']}

**Impact:** {rec['expected_impact']} | **Timeline:** {rec['timeline']}
"""
    
    return analysis

def perform_fallback_search(df, query, segments=None, risk_levels=None, max_results=5):
    """Fallback search using keyword matching when RAG fails"""
    query_lower = query.lower()
    
    # Start with full dataframe
    filtered_df = df.copy()
    
    # Apply segment filter
    if segments:
        filtered_df = filtered_df[filtered_df['Customer_Segment'].isin(segments)]
    
    # Apply risk filter
    if risk_levels:
        filtered_df = filtered_df[filtered_df['Churn_Risk_Level'].isin(risk_levels)]
    
    # Keyword-based scoring
    scores = pd.Series(0, index=filtered_df.index)
    
    # Churn/retention keywords
    if any(word in query_lower for word in ['churn', 'retention', 'leave', 'urgent', 'critical', 'risk']):
        scores += filtered_df['Churn_Probability'] * 100
    
    # Value keywords
    if any(word in query_lower for word in ['value', 'clv', 'protect', 'best', 'high-value', 'premium']):
        scores += (filtered_df['Customer_Lifetime_Value'] / filtered_df['Customer_Lifetime_Value'].max()) * 50
    
    # Claims/risk keywords
    if any(word in query_lower for word in ['claims', 'underpriced', 'risk', 'pricing']):
        scores += filtered_df['Claims_Probability'] * 80
        scores += (1 - filtered_df['Pricing_Adequacy'].clip(upper=2)) * 40
    
    # Growth keywords
    if any(word in query_lower for word in ['upsell', 'growth', 'opportunity', 'cross-sell', 'upgrade']):
        scores += (filtered_df['Seniority'] > 1).astype(int) * 30
        scores += ((filtered_df['Churn_Probability'] < 0.3).astype(int)) * 20
    
    # Onboarding/new customer keywords
    if any(word in query_lower for word in ['new', 'onboard', 'first year', 'recent']):
        scores += (filtered_df['Seniority'] <= 1).astype(int) * 100
    
    # Urban/Rural
    if 'urban' in query_lower:
        scores += (filtered_df['Area'] == 'Urban').astype(int) * 50
    if 'rural' in query_lower:
        scores += (filtered_df['Area'] == 'Rural').astype(int) * 50
    
    # Channel
    if 'agent' in query_lower:
        scores += (filtered_df['Distribution_channel'] == 'Agent').astype(int) * 40
    if 'broker' in query_lower:
        scores += (filtered_df['Distribution_channel'] == 'Broker').astype(int) * 40
    
    # Segment-specific
    if 'protect' in query_lower:
        scores += (filtered_df['Customer_Segment'] == 'PROTECT').astype(int) * 60
    if 'develop' in query_lower:
        scores += (filtered_df['Customer_Segment'] == 'DEVELOP').astype(int) * 60
    
    # Get top results
    filtered_df['search_score'] = scores
    results = filtered_df.nlargest(max_results, 'search_score')
    
    return results.drop(columns=['search_score'])

def show_smart_search(df):
    """Intelligent conversational assistant with RAG + Ollama"""
    
    st.header("🤖 Intelligent Customer Assistant")
    st.markdown("Ask questions in natural language - I'll find and analyze customers for you")
    
    # Initialize session state for conversation
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    # Check systems
    faiss_db, rag_error = load_enhanced_faiss()
    
    # System status
    col1, col2, col3 = st.columns(3)
    with col1:
        if faiss_db:
            # Count customers in RAG index
            try:
                total_docs = faiss_db.index.ntotal
                st.success(f"✅ RAG: {total_docs:,} Records")
            except:
                st.success("✅ RAG System Online")
        else:
            st.error("❌ RAG Offline")
            if rag_error:
                st.caption(f"⚠️ {rag_error[:100]}")
    
    with col2:
        ollama_status = query_ollama("test", "system check")
        if ollama_status:
            st.success("✅ AI Assistant Online")
        else:
            st.warning("⚠️ Smart Fallback Mode")
            st.caption("Using rule-based analysis")
    
    with col3:
        st.info(f"📊 {len(df):,} Total Customers")
    
    st.markdown("---")
    
    # Conversational interface
    st.subheader("💬 Chat with the Assistant")
    
    # Quick start buttons
    st.markdown("**Quick Starts:**")
    quick_options = {
        "🚨 Find urgent retention cases": "Show me customers who need immediate attention to prevent churn",
        "💎 Identify high-value customers": "Find customers with high lifetime value that we should protect",
        "📈 Growth opportunities": "Show me customers ready for upselling or cross-selling",
        "⚠️ Risk assessment": "Find customers with high claims risk and underpriced policies",
        "🎯 New customer onboarding": "Show me customers in their first year who need engagement"
    }
    
    cols = st.columns(3)
    for i, (label, query) in enumerate(quick_options.items()):
        with cols[i % 3]:
            if st.button(label, key=f"quick_{i}", use_container_width=True):
                st.session_state.current_query = query
                st.rerun()
    
    st.markdown("---")
    
    # Main query input
    user_query = st.text_area(
        "Describe what you're looking for:",
        value=st.session_state.get('current_query', ''),
        placeholder="💬 Examples:\n• Show me urban customers with high churn risk and good value\n• Find underpriced policies with multiple claims in the last year\n• Who are my PROTECT segment customers that need attention?\n• Customers paying high premiums but showing low loyalty\n• Find growth opportunities in the broker channel\n\n✨ Tip: Be specific about risk, value, location, or segment!",
        height=140,
        help="🤖 Natural language search powered by RAG + AI. I'll find relevant customers and provide actionable insights tailored to your query."
    )
    
    # Advanced options
    with st.expander("🎛️ Refine Search Criteria (Optional)"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            segments = st.multiselect("Segment", ['PROTECT', 'DEVELOP', 'MANAGE', 'EXIT'], 
                                     help="Filter by customer segment")
        with col2:
            risk_levels = st.multiselect("Risk Level", ['Low', 'Moderate', 'High', 'Critical'],
                                        help="Filter by churn risk")
        with col3:
            num_results = st.slider("Max Results", 1, 20, 5, 
                                   help="Number of customers to analyze")
    
    # Search button
    col1, col2 = st.columns([4, 1])
    with col1:
        search_button = st.button("🔍 Find & Analyze Customers", type="primary", use_container_width=True)
    with col2:
        if st.button("🔄 Clear", use_container_width=True):
            st.session_state.conversation_history = []
            st.session_state.current_query = ''
            st.rerun()
    
    # Process query
    if search_button and user_query:
        
        if not faiss_db:
            st.error("❌ RAG system not available. Please run project_structure/rag.ipynb Steps 1-6 to enable intelligent search.")
            st.info("**Fallback:** Use the Customer Intelligence page to search by ID or use sidebar filters.")
            return
        
        with st.spinner("🤔 Searching database and analyzing customers..."):
            
            # Build filter
            filter_dict = {}
            if segments:
                filter_dict['customer_segment'] = {'$in': segments}
            if risk_levels:
                filter_dict['churn_risk_level'] = {'$in': risk_levels}
            
            # Search RAG
            try:
                if filter_dict:
                    results = faiss_db.similarity_search(user_query, k=num_results, filter=filter_dict)
                else:
                    results = faiss_db.similarity_search(user_query, k=num_results)
                
                if len(results) == 0:
                    st.warning("No customers found matching your criteria. Try broadening your search.")
                    return
                
                # Extract customer IDs from results - check both 'ID' and 'customer_id' keys
                customer_ids = []
                for doc in results:
                    cid = doc.metadata.get('ID') or doc.metadata.get('customer_id')
                    if cid:
                        customer_ids.append(cid)
                
                if not customer_ids:
                    st.warning("⚠️ No customer IDs found in search results. Using fallback search.")
                    # Fallback: Use dataframe search based on query keywords
                    matching_customers = perform_fallback_search(df, user_query, segments, risk_levels, num_results)
                else:
                    matching_customers = df[df['ID'].isin(customer_ids)]
                
                if len(matching_customers) == 0:
                    st.warning("⚠️ No matching customers found in database. Try different criteria.")
                    return
                
                st.success(f"✅ Found {len(matching_customers)} matching customers")
                
                # Store in conversation history
                st.session_state.conversation_history.append({
                    'query': user_query,
                    'results': matching_customers,
                    'timestamp': datetime.now()
                })
                
                st.markdown("---")
                
                # Generate AI response
                st.subheader("🤖 AI Analysis")
                
                # Prepare context for Ollama (safely access first customer)
                first_customer = matching_customers.iloc[0]
                context = f"""You are a Senior Insurance Customer Success Strategist. 
                Use your domain knowledge:
                - Years 1-3 are the 'Danger Zone' (26.5% churn risk).
                - Agent channel has 2.5x higher ROI than Broker channel.
                - Pricing adequacy < 1.0 indicates a loss-making policy regardless of premium.

                Analyze these customers based on the query: "{user_query}"
                
                Found {len(matching_customers)} matches. Top case data:
                - ID: {first_customer['ID']} | Segment: {first_customer['Customer_Segment']}
                - Churn: {first_customer['Churn_Probability']:.1%} ({first_customer['Churn_Risk_Level']})
                - CLV: €{first_customer['Customer_Lifetime_Value']:.0f} | Category: {first_customer['CLV_Segment']}
                - Tenure: {first_customer['Seniority']} years | Claims: {first_customer['N_claims_history']}
                - Channel: {first_customer.get('Distribution_channel', 'Unknown')}

                Provide a professional, executive-level analysis:
                1. Strategic Alignment: Why do these customers matter?
                2. Risk/Value Assessment: Focus on the 'Danger Zone' or 'Channel ROI' if applicable.
                3. Prescriptive Actions: Specific interventions with expected ROI impact.
                """
                
                # Try Ollama first
                try:
                    ai_response = query_ollama(
                        f"Analyze these customers and provide actionable insights for an insurance agent.",
                        context
                    )
                    
                    if ai_response:
                        st.markdown("### 🎯 AI-Powered Insights")
                        st.info(ai_response)
                    else:
                        st.markdown("### 📊 Structured Analysis")
                        fallback_analysis = generate_analysis_fallback(matching_customers, user_query)
                        st.markdown(fallback_analysis)
                except Exception as e:
                    st.warning(f"⚠️ AI analysis error: {str(e)}")
                    st.markdown("### 📊 Structured Analysis")
                    try:
                        fallback_analysis = generate_analysis_fallback(matching_customers, user_query)
                        st.markdown(fallback_analysis)
                    except Exception as e2:
                        st.error(f"Analysis generation failed: {str(e2)}")
                
                # Show detailed customer cards
                st.markdown("---")
                st.subheader(f"📋 Detailed Customer Profiles ({len(matching_customers)} found)")
                
                for idx, (_, customer) in enumerate(matching_customers.head(5).iterrows(), 1):
                    with st.expander(f"Customer #{customer['ID']} - {customer['Customer_Segment']} Segment", expanded=(idx==1)):
                        
                        # Top metrics
                        col1, col2, col3, col4, col5 = st.columns(5)
                        
                        with col1:
                            risk_color = {
                                'Low': '🟢', 'Moderate': '🟡', 
                                'High': '🟠', 'Critical': '🔴'
                            }.get(customer['Churn_Risk_Level'], '⚪')
                            st.metric("Churn Risk", 
                                     f"{risk_color} {customer['Churn_Risk_Level']}",
                                     f"{customer['Churn_Probability']:.1%}")
                        
                        with col2:
                            st.metric("CLV", f"€{customer['Customer_Lifetime_Value']:.0f}",
                                     customer['CLV_Segment'])
                        
                        with col3:
                            st.metric("Premium", f"€{customer['Premium']:.2f}",
                                     "Underpriced" if customer['Is_Underpriced'] == 1 else "Adequate")
                        
                        with col4:
                            st.metric("Seniority", f"{customer['Seniority']} yrs",
                                     f"{customer['N_claims_history']:.0f} claims")
                        
                        with col5:
                            segment_emoji = {
                                'PROTECT': '🛡️', 'DEVELOP': '📈',
                                'MANAGE': '⚙️', 'EXIT': '🚪'
                            }.get(customer['Customer_Segment'], '📦')
                            st.metric("Segment", 
                                     f"{segment_emoji} {customer['Customer_Segment']}")
                        
                        # Detailed info
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**📊 Risk Profile**")
                            st.progress(customer['Churn_Probability'], 
                                      text=f"Churn: {customer['Churn_Probability']:.1%}")
                            st.progress(customer['Claims_Probability'], 
                                      text=f"Claims: {customer['Claims_Probability']:.1%}")
                            st.write(f"**Renewal Risk:** {customer['Renewal_Risk_Score']:.3f}")
                        
                        with col2:
                            st.markdown("**📋 Policy Details**")
                            st.write(f"**Type:** {customer['Type_risk']}")
                            st.write(f"**Area:** {customer['Area']}")
                            st.write(f"**Channel:** {customer['Distribution_channel']}")
                            st.write(f"**Vehicle:** €{customer['Value_vehicle']:.0f}")
                        
                        # Recommendation
                        rec = get_recommendation(customer)
                        st.markdown("**🎯 Recommended Action**")
                        
                        priority_colors = {
                            'critical': '🔴', 'high': '🟠', 'opportunity': '💎',
                            'caution': '⚠️', 'monitor': '👁️', 'focus': '🎯',
                            'low': '🔵', 'routine': '✅'
                        }
                        priority_icon = priority_colors.get(rec['color'], '📌')
                        
                        st.info(f"{priority_icon} **{rec['action']}** - {rec['timeline']}")
                        st.write(rec['details'])
                        st.success(f"💡 {rec['expected_impact']}")
                
                # Export option
                if len(matching_customers) > 0:
                    st.markdown("---")
                    csv = matching_customers.to_csv(index=False)
                    st.download_button(
                        "📥 Download Results as CSV",
                        csv,
                        f"customer_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv",
                        type="secondary",
                        use_container_width=True
                    )
                
            except Exception as e:
                st.error(f"Search error: {str(e)}")
                st.info("Try simplifying your query or adjusting the filters.")
    
    # Show conversation history
    if st.session_state.conversation_history:
        st.markdown("---")
        st.subheader("📚 Recent Searches")
        
        for i, conv in enumerate(reversed(st.session_state.conversation_history[-5:]), 1):
            with st.expander(f"{i}. {conv['query'][:100]}..." if len(conv['query']) > 100 else f"{i}. {conv['query']}"):
                st.write(f"**Time:** {conv['timestamp'].strftime('%H:%M:%S')}")
                st.write(f"**Results:** {len(conv['results'])} customers found")
                if st.button(f"Rerun this search", key=f"rerun_{i}"):
                    st.session_state.current_query = conv['query']
                    st.rerun()
    
    # Help section
    st.markdown("---")
    with st.expander("❓ How to Use This Assistant"):
        st.markdown("""
        ### 💡 Tips for Better Results
        
        **Natural Language Queries:**
        - "Show me customers who might churn soon"
        - "Find high-value customers in urban areas"
        - "Who should I call today for retention?"
        - "Customers with multiple claims and low premiums"
        
        **Specific Criteria:**
        - Mention segments: PROTECT, DEVELOP, MANAGE, EXIT
        - Specify risk: low risk, high churn, critical
        - Include demographics: urban, rural, new customers
        - Reference values: high CLV, underpriced, platinum tier
        
        **What You Get:**
        - ✅ AI-powered analysis of why these customers match
        - ✅ Risk assessment and value metrics
        - ✅ Specific action recommendations with timeline
        - ✅ Expected impact of recommended actions
        - ✅ Exportable customer list
        
        **Pro Tips:**
        - Use quick start buttons for common scenarios
        - Refine with optional filters for precision
        - Export results to share with your team
        - Check recent searches to revisit previous queries
        """)
        
        st.markdown("### 🎯 Example Queries by Use Case")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Retention Focus:**
            - Critical risk customers with high CLV
            - PROTECT segment showing churn signals
            - Customers in first 3 years at risk
            - High value policies about to renew
            """)
        
        with col2:
            st.markdown("""
            **Growth & Revenue:**
            - DEVELOP segment ready for upsell
            - Low-risk customers with long tenure
            - Underpriced policies we can adjust
            - Customers with single products (cross-sell)
            """)

# =============================================================================
# PAGE 5: MODEL PERFORMANCE HUB
# =============================================================================

def show_model_performance(df):
    """Model performance analytics"""
    
    st.header("📈 Model Performance Hub")
    st.markdown("Analytics and monitoring for all 8 predictive models")
    
    # Model summary table
    st.subheader("🎯 Model Summary")
    
    models = pd.DataFrame({
        "Model": [
            "1️⃣ Customer Retention (Churn)",
            "2️⃣ Claims Frequency",
            "3️⃣ Claim Severity",
            "4️⃣ Customer Lifetime Value",
            "5️⃣ Renewal Risk",
            "6️⃣ Pricing Optimization",
            "7️⃣ Customer Segmentation",
            "8️⃣ Channel Attribution"
        ],
        "Type": [
            "Classification",
            "Classification",
            "Regression",
            "Regression",
            "Composite",
            "Business Logic",
            "Rule-Based",
            "Attribution"
        ],
        "Performance": [
            "AUC: 0.715",
            "AUC: 0.923 ⭐",
            "Segment-based",
            "€244 avg CLV",
            "25.9% high-risk",
            "14% underpriced",
            "4 segments",
            "Agent: 752% ROI"
        ],
        "Key Driver": [
            "R_Claims_history",
            "R_Claims_history",
            "Premium",
            "Premium + Seniority",
            "Churn + Claims",
            "Expected Cost",
            "CLV + Risk",
            "Distribution Channel"
        ]
    })
    
    st.dataframe(models, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Prediction distributions
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Churn Probability Distribution")
        fig = px.histogram(df, x='Churn_Probability', nbins=50, 
                          color_discrete_sequence=['#667eea'])
        fig.add_vline(x=df['Churn_Probability'].mean(), line_dash="dash",
                     annotation_text=f"Mean: {df['Churn_Probability'].mean():.1%}")
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Claims Probability Distribution")
        fig = px.histogram(df, x='Claims_Probability', nbins=50,
                          color_discrete_sequence=['#764ba2'])
        fig.add_vline(x=df['Claims_Probability'].mean(), line_dash="dash",
                     annotation_text=f"Mean: {df['Claims_Probability'].mean():.1%}")
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # CLV analysis
    st.markdown("---")
    st.subheader("💰 Customer Lifetime Value Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total CLV", f"€{df['Customer_Lifetime_Value'].sum()/1e6:.1f}M")
    with col2:
        st.metric("Mean CLV", f"€{df['Customer_Lifetime_Value'].mean():.0f}")
    with col3:
        st.metric("Median CLV", f"€{df['Customer_Lifetime_Value'].median():.0f}")
    with col4:
        st.metric("Std Dev", f"€{df['Customer_Lifetime_Value'].std():.0f}")
    
    fig = px.box(df, x='Customer_Segment', y='Customer_Lifetime_Value',
                color='Customer_Segment',
                color_discrete_map={'PROTECT': '#00C851', 'DEVELOP': '#33b5e5',
                                   'MANAGE': '#ffbb33', 'EXIT': '#ff4444'})
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Data quality
    st.markdown("---")
    st.subheader("✅ Data Quality & Coverage")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        st.metric("Completeness", f"{completeness:.1f}%")
    with col3:
        st.metric("Features", len(df.columns))
    with col4:
        st.metric("Predictions", "8 Models")

# =============================================================================
# PAGE 6: STRATEGIC INSIGHTS
# =============================================================================

def show_strategic_insights(df, metrics):
    """Business Case & ROI Dashboard - High Merit Version"""
    
    st.header("💡 Business Case & ROI Analysis")
    st.markdown("Quantifying the financial impact of the proposed retention and risk strategies.")
    
    # ROI Summary Cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="insight-card">
            <div class="metric-label">Preserved Value (Annual)</div>
            <div class="metric-value" style="color: #00C851;">€427k - €598k</div>
            <p style="font-size: 0.8rem; color: #666;">From targeted churn prevention in high-risk, high-value segments.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="insight-card">
            <div class="metric-label">Efficiency Savings</div>
            <div class="metric-value" style="color: #33b5e5;">€2.26M</div>
            <p style="font-size: 0.8rem; color: #666;">By redirecting uniform retention spending to data-driven prioritization.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div class="insight-card">
            <div class="metric-label">Total Annual ROI</div>
            <div class="metric-value" style="color: #764ba2;">3,386%</div>
            <p style="font-size: 0.8rem; color: #666;">Based on €70k implementation cost vs €2.37M periodic value creation.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📊 Strategic Channel Economics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Channel ROI Bar Chart
        channel_data = pd.DataFrame({
            'Channel': ['Agent', 'Broker'],
            'ROI (%)': [752, 297],
            'Avg CLV (€)': [1278, 795]
        })
        
        fig = px.bar(
            channel_data, 
            x='Channel', 
            y='ROI (%)',
            text='ROI (%)',
            color='Channel',
            color_discrete_map={'Agent': '#667eea', 'Broker': '#764ba2'},
            title="Channel ROI Comparison"
        )
        fig.update_traces(texttemplate='%{text}%', textposition='outside')
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.markdown("""
        ### 🧐 The Agent Advantage
        Analysis of 105,555 policies reveals that **Agent-sourced customers** deliver 2.5× superior return compared to Broker channels.
        
        *   **Tenure:** 60% longer (8 vs 5 years avg)
        *   **Claims:** 14% lower costs
        *   **Churn:** 4.7% lower annual rate
        
        **Strategic Recommendation:** Redirect 40% of broker acquisition budget toward agent relationship development.
        """)

    st.markdown("---")
    st.subheader("🧮 Live ROI Simulator")
    st.markdown("Adjust the parameters to see the projected value of the Customer Success Platform.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        success_rate = st.slider("Intervention Success Rate (%)", 5, 50, 25, help="Percentage of high-risk customers successfully retained via the platform.")
        # implementation_cost slider
        custom_budget = st.slider("Annual Platform Budget (€)", 10000, 200000, 70000)
        
    with col2:
        # Calculate dynamic ROI
        crit_customers = metrics['critical_churn']
        avg_clv = metrics['avg_clv']
        saved_customers = crit_customers * (success_rate / 100)
        preserved_value = saved_customers * avg_clv
        
        net_value = preserved_value - custom_budget
        roi = (net_value / custom_budget) * 100 if custom_budget > 0 else 0
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = roi,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Projected ROI (%)", 'font': {'size': 24}},
            gauge = {
                'axis': {'range': [None, 5000], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "#667eea"},
                'steps': [
                    {'range': [0, 500], 'color': "#ff4444"},
                    {'range': [500, 1500], 'color': "#ffbb33"},
                    {'range': [1500, 5000], 'color': "#00C851"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 3386
                }
            }
        ))
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
        
        st.success(f"💡 Based on a **{success_rate}%** success rate, the platform will preserve **€{preserved_value:,.0f}** in annual portfolio value.")

# =============================================================================
# RUN APP
# =============================================================================

if __name__ == "__main__":
    main()
# Professional Footer
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>🏢 <b>Insurance Customer Success Platform v2.0</b> | Created for Highest Merit Execution</p>
    <p>© 2025 Valerie Jerono - Research Methodology Project | Strathmore University</p>
</div>
""", unsafe_allow_html=True)
