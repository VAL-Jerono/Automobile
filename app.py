"""
Insurance Analytics Platform - Human-Friendly Edition
Each page answers a specific business question with clear visualizations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import logging
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Insurance Analytics", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif; }
[data-testid="stAppViewContainer"] { background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%); color: #1e293b; }
[data-testid="stSidebarContent"] { background: #ffffff !important; border-right: 1px solid #e2e8f0; }
.question-header { font-size: 2rem; font-weight: 800; color: #0f172a; margin: 1rem 0 0.5rem 0; letter-spacing: -0.02em; }
.answer-text { font-size: 1.1rem; color: #475569; line-height: 1.6; margin-bottom: 2rem; }
.metric-card { background: white; border-radius: 12px; padding: 1.5rem; box-shadow: 0 2px 8px rgba(0,0,0,0.04); border: 1px solid #e2e8f0; transition: all 0.2s; }
.metric-card:hover { box-shadow: 0 4px 16px rgba(0,0,0,0.08); transform: translateY(-2px); }
.big-number { font-size: 2.5rem; font-weight: 800; margin: 0.5rem 0; }
.metric-label { font-size: 0.875rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600; }
.insight-box { background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border-left: 4px solid #0ea5e9; padding: 1rem 1.25rem; border-radius: 8px; margin: 1rem 0; }
.warning-box { background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); border-left: 4px solid #f59e0b; padding: 1rem 1.25rem; border-radius: 8px; margin: 1rem 0; }
.success-box { background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); border-left: 4px solid #10b981; padding: 1rem 1.25rem; border-radius: 8px; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)

def process_dataframe(df):
    col_map = {'policy_id': 'ID', 'churn_probability': 'Churn_Prob', 'claims_probability': 'Claims_Prob', 'claims_severity': 'Claims_Severity', 'customer_lifetime_value': 'CLV', 'customer_segment': 'Segment', 'journey_quadrant': 'Journey', 'pricing_adequacy_flag': 'Underpriced', 'renewal_risk_score': 'Renewal_Risk', 'is_high_renewal_risk': 'High_Renewal_Risk'}
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    if 'Churn_Prob' in df.columns:
        df['Risk_Category'] = pd.cut(df['Churn_Prob'], bins=[0, 0.3, 0.6, 0.85, 1.1], labels=['Low Risk', 'Medium Risk', 'High Risk', 'Critical Risk'])
    if 'CLV' in df.columns:
        df['Value_Tier'] = pd.qcut(df['CLV'], q=4, labels=['Bronze', 'Silver', 'Gold', 'Platinum'])
    return df

@st.cache_data(ttl=3600)
def load_data():
    project_path = Path(__file__).parent
    if project_path.exists():
        sys.path.insert(0, str(project_path))
    try:
        from utils.sql_predictions_manager import SQLModelPredictionsManager
        manager = SQLModelPredictionsManager()
        if manager.connect():
            df = manager.get_all_predictions()
            info = manager.get_connection_info()
            manager.disconnect()
            if df is not None and not df.empty:
                return process_dataframe(df), f"Live Database ({info})"
    except Exception as e:
        logger.warning(f"SQL connection failed: {e}")
    csv_paths = [project_path / "model_outputs" / "rag_model_predictions.csv", project_path / "rag_model_predictions.csv", Path("model_outputs/rag_model_predictions.csv")]
    for path in csv_paths:
        if path.exists():
            try:
                df = pd.read_csv(path)
                return process_dataframe(df), f"CSV File ({path.name})"
            except Exception as e:
                logger.error(f"CSV loading error: {e}")
    return None, "No data source available"

COLORS = {'critical': '#ef4444', 'high': '#f97316', 'medium': '#f59e0b', 'low': '#10b981', 'primary': '#3b82f6', 'secondary': '#8b5cf6', 'accent': '#06b6d4'}

def create_metric_card(label, value, subtitle="", color='primary'):
    return f'<div class="metric-card"><div class="metric-label">{label}</div><div class="big-number" style="color: {COLORS.get(color, COLORS["primary"])}">{value}</div>{f"<div style=\"color: #64748b; font-size: 0.9rem;\">{subtitle}</div>" if subtitle else ""}</div>'

df, data_source = load_data()

with st.sidebar:
    st.markdown("### 🛡️ Insurance Analytics")
    st.markdown("---")
    pages = {"📊 Dashboard": "Which customers need attention NOW?", "❌ Who Will Leave?": "Who is most likely to churn?", "💰 Who Is Worth Most?": "Where is our revenue concentrated?", "🚨 Who Will File Claims?": "Who represents the highest claims risk?", "🎯 How Do We Prioritize?": "Which customers need what action?", "📈 Are We Pricing Right?": "Where are we losing money?", "🔍 Custom Analysis": "Ask your own questions"}
    page = st.radio("Navigate to:", list(pages.keys()), format_func=lambda x: x, label_visibility="collapsed")
    st.markdown("---")
    st.markdown(f"**Data Source:** {data_source}")
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

def main():
    if df is None or df.empty:
        st.error("⚠️ No data available")
        st.info("Please ensure the data file is accessible")
        if st.button("🔄 Retry Loading"):
            st.rerun()
        st.stop()
    
    total_customers = len(df)
    critical_customers = len(df[df['Risk_Category'] == 'Critical Risk'])
    high_value = len(df[df['Value_Tier'] == 'Platinum'])
    total_clv = df['CLV'].sum()
    avg_churn = df['Churn_Prob'].mean()
    high_risk_customers = len(df[df['Churn_Prob'] > 0.6])
    
    if page == "📊 Dashboard":
        st.markdown('<div class="question-header">Which customers need immediate attention?</div>', unsafe_allow_html=True)
        st.markdown('<div class="answer-text">Here\'s your portfolio at a glance, highlighting the most urgent priorities.</div>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(create_metric_card("CRITICAL ALERTS", f"{critical_customers:,}", f"{critical_customers/total_customers*100:.1f}% of portfolio", 'critical'), unsafe_allow_html=True)
        with col2:
            st.markdown(create_metric_card("HIGH RISK", f"{high_risk_customers:,}", "Churn probability > 60%", 'high'), unsafe_allow_html=True)
        with col3:
            st.markdown(create_metric_card("PREMIUM CUSTOMERS", f"{high_value:,}", "Top 25% by value", 'low'), unsafe_allow_html=True)
        with col4:
            st.markdown(create_metric_card("PORTFOLIO VALUE", f"€{total_clv/1e6:.1f}M", f"€{total_clv/total_customers:,.0f} avg/customer", 'primary'), unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("### 🎯 Key Insights")
        critical_value = df[df['Risk_Category'] == 'Critical Risk']['CLV'].sum()
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'<div class="warning-box"><strong>⚠️ Revenue at Risk</strong><br/>€{critical_value/1e6:.2f}M in lifetime value from critical risk customers. This represents {critical_value/total_clv*100:.1f}% of your total portfolio.</div>', unsafe_allow_html=True)
        with col2:
            underpriced = df['Underpriced'].sum() if 'Underpriced' in df.columns else 0
            st.markdown(f'<div class="insight-box"><strong>💡 Pricing Opportunity</strong><br/>{underpriced:,} customers ({underpriced/total_customers*100:.1f}%) are currently underpriced. Review pricing to optimize margins.</div>', unsafe_allow_html=True)
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📊 Risk Distribution")
            risk_counts = df['Risk_Category'].value_counts()
            fig = go.Figure()
            fig.add_trace(go.Bar(x=risk_counts.index, y=risk_counts.values, marker=dict(color=['#10b981', '#f59e0b', '#f97316', '#ef4444'], line=dict(color='white', width=2)), text=risk_counts.values, textposition='outside', textfont=dict(size=14, weight='bold')))
            fig.update_layout(height=350, paper_bgcolor='white', plot_bgcolor='#f8fafc', margin=dict(t=20, b=60, l=60, r=20), xaxis=dict(title="Risk Level", gridcolor='#e2e8f0'), yaxis=dict(title="Number of Customers", gridcolor='#e2e8f0'), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown("### 💎 Value Distribution")
            value_counts = df['Value_Tier'].value_counts()
            fig = go.Figure()
            fig.add_trace(go.Bar(x=value_counts.index, y=value_counts.values, marker=dict(color=['#cd7f32', '#c0c0c0', '#ffd700', '#e5e4e2'], line=dict(color='white', width=2)), text=value_counts.values, textposition='outside', textfont=dict(size=14, weight='bold')))
            fig.update_layout(height=350, paper_bgcolor='white', plot_bgcolor='#f8fafc', margin=dict(t=20, b=60, l=60, r=20), xaxis=dict(title="Value Tier", gridcolor='#e2e8f0'), yaxis=dict(title="Number of Customers", gridcolor='#e2e8f0'), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("### 🚨 Top 10 Priority Customers")
        st.markdown("*Critical risk customers with highest value - act immediately*")
        priority = df[df['Risk_Category'] == 'Critical Risk'].nlargest(10, 'CLV')
        display_df = priority[['ID', 'Churn_Prob', 'CLV', 'Claims_Prob', 'Segment']].copy()
        display_df['Churn_Prob'] = display_df['Churn_Prob'].apply(lambda x: f"{x*100:.1f}%")
        display_df['CLV'] = display_df['CLV'].apply(lambda x: f"€{x:,.0f}")
        display_df['Claims_Prob'] = display_df['Claims_Prob'].apply(lambda x: f"{x*100:.1f}%")
        st.dataframe(display_df, use_container_width=True, height=400)
    
    elif page == "❌ Who Will Leave?":
        st.markdown('<div class="question-header">Who is most likely to churn?</div>', unsafe_allow_html=True)
        st.markdown('<div class="answer-text">Identify customers at risk of leaving so you can take preventive action.</div>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        very_high_churn = len(df[df['Churn_Prob'] > 0.8])
        high_churn = len(df[(df['Churn_Prob'] > 0.6) & (df['Churn_Prob'] <= 0.8)])
        at_risk_value = df[df['Churn_Prob'] > 0.6]['CLV'].sum()
        with col1:
            st.markdown(create_metric_card("VERY HIGH RISK", f"{very_high_churn:,}", "> 80% churn probability", 'critical'), unsafe_allow_html=True)
        with col2:
            st.markdown(create_metric_card("HIGH RISK", f"{high_churn:,}", "60-80% churn probability", 'high'), unsafe_allow_html=True)
        with col3:
            st.markdown(create_metric_card("VALUE AT RISK", f"€{at_risk_value/1e6:.1f}M", "From high-risk customers", 'medium'), unsafe_allow_html=True)
        with col4:
            st.markdown(create_metric_card("AVG CHURN RATE", f"{avg_churn*100:.1f}%", "Portfolio average", 'primary'), unsafe_allow_html=True)
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📉 Churn Probability Distribution")
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=df['Churn_Prob'], nbinsx=50, marker=dict(color='#ef4444', line=dict(color='white', width=1)), opacity=0.8))
            fig.update_layout(height=350, paper_bgcolor='white', plot_bgcolor='#f8fafc', margin=dict(t=20, b=60, l=60, r=20), xaxis=dict(title="Churn Probability", tickformat='.0%', gridcolor='#e2e8f0'), yaxis=dict(title="Number of Customers", gridcolor='#e2e8f0'), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown("### 🎯 Churn by Customer Segment")
            segment_churn = df.groupby('Segment')['Churn_Prob'].mean().sort_values(ascending=False).head(10)
            fig = go.Figure()
            fig.add_trace(go.Bar(y=segment_churn.index, x=segment_churn.values, orientation='h', marker=dict(color=segment_churn.values, colorscale='Reds', line=dict(color='white', width=1)), text=[f"{v*100:.1f}%" for v in segment_churn.values], textposition='outside'))
            fig.update_layout(height=350, paper_bgcolor='white', plot_bgcolor='#f8fafc', margin=dict(t=20, b=60, l=140, r=80), xaxis=dict(title="Average Churn Probability", tickformat='.0%', gridcolor='#e2e8f0'), yaxis=dict(title=""), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("### 📊 Risk vs Value Analysis")
        fig = px.scatter(df.sample(min(5000, len(df)), random_state=42), x='Churn_Prob', y='CLV', color='Risk_Category', color_discrete_map={'Low Risk': '#10b981', 'Medium Risk': '#f59e0b', 'High Risk': '#f97316', 'Critical Risk': '#ef4444'}, opacity=0.6, size='Claims_Severity', hover_data=['Segment'])
        fig.update_layout(height=400, paper_bgcolor='white', plot_bgcolor='#f8fafc', margin=dict(t=20, b=60, l=60, r=20), xaxis=dict(title="Churn Probability", tickformat='.0%', gridcolor='#e2e8f0'), yaxis=dict(title="Customer Lifetime Value (€)", tickformat=',d', gridcolor='#e2e8f0'))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("### 🚨 Top 20 Churn Risks")
        st.markdown("*Customers most likely to leave - prioritize retention efforts*")
        top_churners = df.nlargest(20, 'Churn_Prob')[['ID', 'Churn_Prob', 'CLV', 'Segment', 'Renewal_Risk']].copy()
        top_churners['Churn_Prob'] = top_churners['Churn_Prob'].apply(lambda x: f"{x*100:.1f}%")
        top_churners['CLV'] = top_churners['CLV'].apply(lambda x: f"€{x:,.0f}")
        top_churners['Renewal_Risk'] = top_churners['Renewal_Risk'].apply(lambda x: f"{x:.2f}")
        st.dataframe(top_churners, use_container_width=True, height=500)
    
    elif page == "💰 Who Is Worth Most?":
        st.markdown('<div class="question-header">Where is our revenue concentrated?</div>', unsafe_allow_html=True)
        st.markdown('<div class="answer-text">Understand which customers drive the most value and protect your revenue base.</div>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        platinum_value = df[df['Value_Tier'] == 'Platinum']['CLV'].sum()
        top_10_pct_value = df.nlargest(int(len(df)*0.1), 'CLV')['CLV'].sum()
        avg_clv = df['CLV'].mean()
        median_clv = df['CLV'].median()
        with col1:
            st.markdown(create_metric_card("TOP 25% VALUE", f"€{platinum_value/1e6:.1f}M", f"{platinum_value/total_clv*100:.1f}% of portfolio", 'primary'), unsafe_allow_html=True)
        with col2:
            st.markdown(create_metric_card("TOP 10% VALUE", f"€{top_10_pct_value/1e6:.1f}M", f"{top_10_pct_value/total_clv*100:.1f}% of portfolio", 'secondary'), unsafe_allow_html=True)
        with col3:
            st.markdown(create_metric_card("AVERAGE CLV", f"€{avg_clv:,.0f}", "Per customer", 'accent'), unsafe_allow_html=True)
        with col4:
            st.markdown(create_metric_card("MEDIAN CLV", f"€{median_clv:,.0f}", "Middle customer value", 'low'), unsafe_allow_html=True)
        st.markdown("---")
        st.markdown(f'<div class="insight-box"><strong>💡 Revenue Concentration</strong><br/>Your top 10% of customers represent €{top_10_pct_value/1e6:.1f}M ({top_10_pct_value/total_clv*100:.1f}% of total value). Focus retention efforts here for maximum impact.</div>', unsafe_allow_html=True)
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 💎 Value by Tier")
            tier_stats = df.groupby('Value_Tier').agg({'CLV': 'sum', 'ID': 'count'}).reset_index()
            tier_stats.columns = ['Tier', 'Total_Value', 'Count']
            fig = go.Figure()
            fig.add_trace(go.Bar(x=tier_stats['Tier'], y=tier_stats['Total_Value'], marker=dict(color=['#cd7f32', '#c0c0c0', '#ffd700', '#e5e4e2'], line=dict(color='white', width=2)), text=[f"€{v/1e6:.1f}M" for v in tier_stats['Total_Value']], textposition='outside', textfont=dict(size=12, weight='bold')))
            fig.update_layout(height=350, paper_bgcolor='white', plot_bgcolor='#f8fafc', margin=dict(t=20, b=60, l=60, r=20), xaxis=dict(title="Value Tier", gridcolor='#e2e8f0'), yaxis=dict(title="Total Value (€)", tickformat=',d', gridcolor='#e2e8f0'), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown("### 📊 Top Segments by Value")
            segment_value = df.groupby('Segment')['CLV'].sum().sort_values(ascending=False).head(10)
            fig = go.Figure()
            fig.add_trace(go.Bar(y=segment_value.index, x=segment_value.values, orientation='h', marker=dict(color=segment_value.values, colorscale='Viridis', line=dict(color='white', width=1)), text=[f"€{v/1e6:.1f}M" for v in segment_value.values], textposition='outside'))
            fig.update_layout(height=350, paper_bgcolor='white', plot_bgcolor='#f8fafc', margin=dict(t=20, b=60, l=140, r=100), xaxis=dict(title="Total Value (€)", tickformat=',d', gridcolor='#e2e8f0'), yaxis=dict(title=""), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("### 📈 Customer Value Distribution")
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df['CLV'], nbinsx=60, marker=dict(color='#3b82f6', line=dict(color='white', width=1)), opacity=0.8))
        fig.update_layout(height=350, paper_bgcolor='white', plot_bgcolor='#f8fafc', margin=dict(t=20, b=60, l=60, r=20), xaxis=dict(title="Customer Lifetime Value (€)", tickformat=',d', gridcolor='#e2e8f0'), yaxis=dict(title="Number of Customers", gridcolor='#e2e8f0'), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("### 🏆 Top 20 Most Valuable Customers")
        top_value = df.nlargest(20, 'CLV')[['ID', 'CLV', 'Churn_Prob', 'Segment', 'Risk_Category']].copy()
        top_value['CLV'] = top_value['CLV'].apply(lambda x: f"€{x:,.0f}")
        top_value['Churn_Prob'] = top_value['Churn_Prob'].apply(lambda x: f"{x*100:.1f}%")
        st.dataframe(top_value, use_container_width=True, height=500)
    
    elif page == "🚨 Who Will File Claims?":
        st.markdown('<div class="question-header">Who represents the highest claims risk?</div>', unsafe_allow_html=True)
        st.markdown('<div class="answer-text">Predict and prepare for potential claims to manage reserves and pricing.</div>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        high_claims_risk = len(df[df['Claims_Prob'] > 0.5])
        avg_claims_prob = df['Claims_Prob'].mean()
        total_exposure = df['Claims_Severity'].sum()
        avg_severity = df['Claims_Severity'].mean()
        with col1:
            st.markdown(create_metric_card("HIGH CLAIMS RISK", f"{high_claims_risk:,}", "> 50% probability", 'critical'), unsafe_allow_html=True)
        with col2:
            st.markdown(create_metric_card("AVG CLAIMS PROB", f"{avg_claims_prob*100:.1f}%", "Portfolio average", 'medium'), unsafe_allow_html=True)
        with col3:
            st.markdown(create_metric_card("TOTAL EXPOSURE", f"€{total_exposure/1e6:.1f}M", "Expected claims cost", 'high'), unsafe_allow_html=True)
        with col4:
            st.markdown(create_metric_card("AVG SEVERITY", f"€{avg_severity:,.0f}", "Per claim", 'primary'), unsafe_allow_html=True)
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📊 Claims Probability Distribution")
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=df['Claims_Prob'], nbinsx=50, marker=dict(color='#f97316', line=dict(color='white', width=1)), opacity=0.8))
            fig.update_layout(height=350, paper_bgcolor='white', plot_bgcolor='#f8fafc', margin=dict(t=20, b=60, l=60, r=20), xaxis=dict(title="Claims Probability", tickformat='.0%', gridcolor='#e2e8f0'), yaxis=dict(title="Number of Customers", gridcolor='#e2e8f0'), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown("### 💥 Claims Severity Distribution")
            fig = go.Figure()
            fig.add_trace(go.Box(y=df.sample(min(5000, len(df)), random_state=7)['Claims_Severity'], marker=dict(color='#06b6d4'), boxmean='sd'))
            fig.update_layout(height=350, paper_bgcolor='white', plot_bgcolor='#f8fafc', margin=dict(t=20, b=60, l=60, r=20), showlegend=False, yaxis=dict(title="Expected Claim Amount (€)", tickformat=',d', gridcolor='#e2e8f0'))
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("### 📊 Probability vs Severity")
        import plotly.figure_factory as ff
        sample_df = df.sample(min(4000, len(df)), random_state=9)
        # Use a 2D density heatmap for better readability
        fig = ff.create_2d_density(
            x=sample_df['Claims_Prob'],
            y=sample_df['Claims_Severity'],
            colorscale='Blues',
            hist_color='rgba(16,185,129,0.2)',
            point_size=2,
            title='Probability vs Severity Density'
        )
        fig.update_layout(
            height=400,
            paper_bgcolor='white',
            plot_bgcolor='#f8fafc',
            margin=dict(t=20, b=60, l=60, r=20),
            xaxis=dict(title="Claims Probability", tickformat='.0%', gridcolor='#e2e8f0'),
            yaxis=dict(title="Expected Claim Amount (€)", tickformat=',d', gridcolor='#e2e8f0')
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("### 🚨 Top 20 Claims Risks")
        st.markdown("*Customers most likely to file high-value claims*")
        top_claims = df.nlargest(20, 'Claims_Prob')[['ID', 'Claims_Prob', 'Claims_Severity', 'CLV', 'Segment']].copy()
        top_claims['Claims_Prob'] = top_claims['Claims_Prob'].apply(lambda x: f"{x*100:.1f}%")
        top_claims['Claims_Severity'] = top_claims['Claims_Severity'].apply(lambda x: f"€{x:,.0f}")
        top_claims['CLV'] = top_claims['CLV'].apply(lambda x: f"€{x:,.0f}")
        st.dataframe(top_claims, use_container_width=True, height=500)
    
    elif page == "🎯 How Do We Prioritize?":
        st.markdown('<div class="question-header">Which customers need what action?</div>', unsafe_allow_html=True)
        st.markdown('<div class="answer-text">Strategic framework to prioritize actions based on customer value and risk.</div>', unsafe_allow_html=True)
        j_counts = df['Journey'].value_counts()
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(create_metric_card("PROTECT", f"{j_counts.get('Protect',0):,}", "High value, low risk", 'low'), unsafe_allow_html=True)
        with col2:
            st.markdown(create_metric_card("GROW", f"{j_counts.get('Grow',0):,}", "Low value, low risk", 'primary'), unsafe_allow_html=True)
        with col3:
            st.markdown(create_metric_card("RESCUE", f"{j_counts.get('Rescue',0):,}", "High value, high risk", 'critical'), unsafe_allow_html=True)
        with col4:
            st.markdown(create_metric_card("MONITOR", f"{j_counts.get('Monitor',0):,}", "Low value, high risk", 'medium'), unsafe_allow_html=True)
        st.markdown("---")
        journey_colors = {'Protect': '#10b981', 'Grow': '#3b82f6', 'Rescue': '#ef4444', 'Monitor': '#f59e0b'}
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📊 Journey Quadrant Distribution")
            fig = go.Figure()
            fig.add_trace(go.Bar(x=j_counts.index, y=j_counts.values, marker=dict(color=[journey_colors.get(j, '#6b7280') for j in j_counts.index], line=dict(color='white', width=2)), text=j_counts.values, textposition='outside', textfont=dict(size=14, weight='bold')))
            fig.update_layout(height=350, paper_bgcolor='white', plot_bgcolor='#f8fafc', margin=dict(t=20, b=60, l=60, r=20), showlegend=False, xaxis=dict(title="Journey Quadrant", gridcolor='#e2e8f0'), yaxis=dict(title="Number of Customers", gridcolor='#e2e8f0'))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown("### 💰 Value by Journey")
            journey_value = df.groupby('Journey')['CLV'].sum().sort_values(ascending=False)
            fig = go.Figure()
            fig.add_trace(go.Bar(x=journey_value.index, y=journey_value.values, marker=dict(color=[journey_colors.get(j, '#6b7280') for j in journey_value.index], line=dict(color='white', width=2)), text=[f"€{v/1e6:.1f}M" for v in journey_value.values], textposition='outside', textfont=dict(size=12, weight='bold')))
            fig.update_layout(height=350, paper_bgcolor='white', plot_bgcolor='#f8fafc', margin=dict(t=20, b=60, l=60, r=20), showlegend=False, xaxis=dict(title="Journey Quadrant", gridcolor='#e2e8f0'), yaxis=dict(title="Total Value (€)", tickformat=',d', gridcolor='#e2e8f0'))
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("### 📍 Strategic Positioning Map")
        fig = px.scatter(df.sample(min(3500, len(df)), random_state=5), x='Renewal_Risk', y='CLV', color='Journey', color_discrete_map=journey_colors, opacity=0.7, size='Claims_Severity', hover_data=['Segment', 'Churn_Prob'])
        fig.update_layout(height=400, paper_bgcolor='white', plot_bgcolor='#f8fafc', margin=dict(t=20, b=60, l=60, r=20), xaxis=dict(title="Renewal Risk Score", gridcolor='#e2e8f0'), yaxis=dict(title="Customer Lifetime Value (€)", tickformat=',d', gridcolor='#e2e8f0'))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("### 📋 Action Recommendations by Quadrant")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="success-box"><strong>🛡️ PROTECT Quadrant</strong><br/>High value, low risk customers. Action: Maintain satisfaction, offer loyalty rewards, prevent competitive poaching.</div>', unsafe_allow_html=True)
            st.markdown('<div class="warning-box"><strong>⚠️ RESCUE Quadrant</strong><br/>High value, high risk customers. Action: URGENT retention campaigns, personalized outreach, address pain points immediately.</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="insight-box"><strong>📈 GROW Quadrant</strong><br/>Low value, low risk customers. Action: Upsell opportunities, cross-sell products, increase engagement gradually.</div>', unsafe_allow_html=True)
            st.markdown('<div style="background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%); border-left: 4px solid #ef4444; padding: 1rem 1.25rem; border-radius: 8px; margin: 1rem 0;"><strong>👀 MONITOR Quadrant</strong><br/>Low value, high risk customers. Action: Review pricing adequacy, consider non-renewal, minimize acquisition costs.</div>', unsafe_allow_html=True)
    
    elif page == "📈 Are We Pricing Right?":
        st.markdown('<div class="question-header">Where are we losing money on pricing?</div>', unsafe_allow_html=True)
        st.markdown('<div class="answer-text">Identify underpriced policies and optimize your pricing strategy.</div>', unsafe_allow_html=True)
        if 'Underpriced' not in df.columns:
            st.warning("⚠️ Pricing adequacy data not available in current dataset")
            st.stop()
        underpriced_count = df['Underpriced'].sum()
        underpriced_value = df[df['Underpriced'] == 1]['CLV'].sum()
        underpriced_claims = df[df['Underpriced'] == 1]['Claims_Severity'].sum()
        properly_priced = len(df) - underpriced_count
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(create_metric_card("UNDERPRICED", f"{underpriced_count:,}", f"{underpriced_count/len(df)*100:.1f}% of portfolio", 'critical'), unsafe_allow_html=True)
        with col2:
            st.markdown(create_metric_card("VALUE AT RISK", f"€{underpriced_value/1e6:.1f}M", "From underpriced policies", 'high'), unsafe_allow_html=True)
        with col3:
            st.markdown(create_metric_card("CLAIMS EXPOSURE", f"€{underpriced_claims/1e6:.1f}M", "Expected from underpriced", 'medium'), unsafe_allow_html=True)
        with col4:
            st.markdown(create_metric_card("PROPERLY PRICED", f"{properly_priced:,}", f"{properly_priced/len(df)*100:.1f}% of portfolio", 'low'), unsafe_allow_html=True)
        st.markdown("---")
        st.markdown(f'<div class="warning-box"><strong>⚠️ Pricing Gap Alert</strong><br/>You have {underpriced_count:,} underpriced policies representing €{underpriced_value/1e6:.1f}M in value. Review these policies to optimize margins and improve profitability.</div>', unsafe_allow_html=True)
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📊 Pricing Adequacy Breakdown")
            pricing_counts = df['Underpriced'].value_counts()
            labels = ['Properly Priced', 'Underpriced']
            fig = go.Figure()
            fig.add_trace(go.Bar(x=labels, y=[pricing_counts.get(0, 0), pricing_counts.get(1, 0)], marker=dict(color=['#10b981', '#ef4444'], line=dict(color='white', width=2)), text=[pricing_counts.get(0, 0), pricing_counts.get(1, 0)], textposition='outside', textfont=dict(size=14, weight='bold')))
            fig.update_layout(height=350, paper_bgcolor='white', plot_bgcolor='#f8fafc', margin=dict(t=20, b=60, l=60, r=20), showlegend=False, xaxis=dict(gridcolor='#e2e8f0'), yaxis=dict(title="Number of Policies", gridcolor='#e2e8f0'))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown("### 💰 Value Distribution by Pricing")
            pricing_value = df.groupby('Underpriced')['CLV'].sum()
            labels = ['Properly Priced', 'Underpriced']
            fig = go.Figure()
            fig.add_trace(go.Bar(x=labels, y=[pricing_value.get(0, 0), pricing_value.get(1, 0)], marker=dict(color=['#10b981', '#ef4444'], line=dict(color='white', width=2)), text=[f"€{pricing_value.get(0, 0)/1e6:.1f}M", f"€{pricing_value.get(1, 0)/1e6:.1f}M"], textposition='outside', textfont=dict(size=12, weight='bold')))
            fig.update_layout(height=350, paper_bgcolor='white', plot_bgcolor='#f8fafc', margin=dict(t=20, b=60, l=60, r=20), showlegend=False, xaxis=dict(gridcolor='#e2e8f0'), yaxis=dict(title="Total Value (€)", tickformat=',d', gridcolor='#e2e8f0'))
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("### 📊 Underpriced Policies by Segment")
        underpriced_by_seg = df[df['Underpriced'] == 1].groupby('Segment').size().sort_values(ascending=False).head(10)
        fig = go.Figure()
        fig.add_trace(go.Bar(y=underpriced_by_seg.index, x=underpriced_by_seg.values, orientation='h', marker=dict(color='#ef4444', line=dict(color='white', width=1)), text=underpriced_by_seg.values, textposition='outside'))
        fig.update_layout(height=350, paper_bgcolor='white', plot_bgcolor='#f8fafc', margin=dict(t=20, b=60, l=140, r=60), xaxis=dict(title="Number of Underpriced Policies", gridcolor='#e2e8f0'), yaxis=dict(title=""), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("### 🚨 Top 20 Underpriced High-Value Policies")
        st.markdown("*Priority policies for pricing review*")
        underpriced_df = df[df['Underpriced'] == 1].nlargest(20, 'CLV')[['ID', 'CLV', 'Claims_Prob', 'Claims_Severity', 'Segment', 'Churn_Prob']].copy()
        underpriced_df['CLV'] = underpriced_df['CLV'].apply(lambda x: f"€{x:,.0f}")
        underpriced_df['Claims_Prob'] = underpriced_df['Claims_Prob'].apply(lambda x: f"{x*100:.1f}%")
        underpriced_df['Claims_Severity'] = underpriced_df['Claims_Severity'].apply(lambda x: f"€{x:,.0f}")
        underpriced_df['Churn_Prob'] = underpriced_df['Churn_Prob'].apply(lambda x: f"{x*100:.1f}%")
        st.dataframe(underpriced_df, use_container_width=True, height=500)
    
    elif page == "🔍 Custom Analysis":
        st.markdown('<div class="question-header">Ask your own questions</div>', unsafe_allow_html=True)
        st.markdown('<div class="answer-text">Use natural language to query your portfolio data.</div>', unsafe_allow_html=True)
        with st.expander("💡 Example Questions", expanded=True):
            st.markdown("- Show top 10 highest churn risk customers\n- Find high value customers with critical risk\n- List platinum segment with high churn probability\n- Show top 5 customers most likely to file claims\n- Find low risk high value customers for upselling")
        user_query = st.text_input("🔎 Ask a question about your portfolio", placeholder="e.g., Show top 5 critical churn customers with high CLV")
        if user_query:
            try:
                from scripts.rag.rag_system import InsuranceRAGSystem
                with st.spinner("🔍 Analyzing your portfolio..."):
                    rag = InsuranceRAGSystem(df=df)
                    results_df, explanation = rag.query(user_query)
                st.success(explanation)
                if len(results_df) > 0:
                    st.markdown("### 👥 Query Results")
                    display_df = results_df.copy()
                    if 'Churn_Prob' in display_df.columns:
                        display_df['Churn_Prob'] = display_df['Churn_Prob'].apply(lambda x: f"{float(x)*100:.1f}%")
                    if 'Claims_Prob' in display_df.columns:
                        display_df['Claims_Prob'] = display_df['Claims_Prob'].apply(lambda x: f"{float(x)*100:.1f}%")
                    if 'CLV' in display_df.columns:
                        display_df['CLV'] = display_df['CLV'].apply(lambda x: f"€{float(x):,.0f}")
                    if 'Claims_Severity' in display_df.columns:
                        display_df['Claims_Severity'] = display_df['Claims_Severity'].apply(lambda x: f"€{float(x):,.0f}")
                    st.dataframe(display_df, use_container_width=True, height=400)
                    csv = results_df.to_csv(index=False)
                    st.download_button("📥 Download Results as CSV", csv, "query_results.csv", "text/csv", use_container_width=True)
                else:
                    st.warning("No results found matching your query")
            except ImportError:
                st.warning("⚠️ RAG system not available")
                st.info("The natural language query system requires additional dependencies. Please install the full requirements.txt file.")
            except Exception as e:
                st.error(f"❌ Error processing query: {str(e)}")
        st.markdown("---")
        st.markdown("### 📊 Quick Filters")
        col1, col2, col3 = st.columns(3)
        with col1:
            risk_filter = st.multiselect("Risk Category", df['Risk_Category'].unique(), default=[])
        with col2:
            value_filter = st.multiselect("Value Tier", df['Value_Tier'].unique(), default=[])
        with col3:
            journey_filter = st.multiselect("Journey Quadrant", df['Journey'].unique(), default=[])
        if risk_filter or value_filter or journey_filter:
            filtered_df = df.copy()
            if risk_filter:
                filtered_df = filtered_df[filtered_df['Risk_Category'].isin(risk_filter)]
            if value_filter:
                filtered_df = filtered_df[filtered_df['Value_Tier'].isin(value_filter)]
            if journey_filter:
                filtered_df = filtered_df[filtered_df['Journey'].isin(journey_filter)]
            st.markdown(f"### 📋 Filtered Results ({len(filtered_df):,} customers)")
            display_df = filtered_df[['ID', 'Churn_Prob', 'CLV', 'Claims_Prob', 'Segment', 'Risk_Category', 'Value_Tier', 'Journey']].copy()
            display_df['Churn_Prob'] = display_df['Churn_Prob'].apply(lambda x: f"{x*100:.1f}%")
            display_df['CLV'] = display_df['CLV'].apply(lambda x: f"€{x:,.0f}")
            display_df['Claims_Prob'] = display_df['Claims_Prob'].apply(lambda x: f"{x*100:.1f}%")
            st.dataframe(display_df, use_container_width=True, height=500)
            csv = filtered_df.to_csv(index=False)
            st.download_button("📥 Download Filtered Data", csv, "filtered_portfolio.csv", "text/csv", use_container_width=True)

if __name__ == "__main__":
    main()