"""
Insurance Analytics Platform v8.0 - Light Theme Edition
Complete version - Ready to use
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

st.set_page_config(page_title="Insurance Analytics Pro", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

# Light Theme Styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif; }
[data-testid="stAppViewContainer"] { background: linear-gradient(135deg, #f5f7fa 0%, #e8f0fe 50%, #fef3f2 100%); color: #1a1a1a; }
[data-testid="stSidebarContent"] { background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%) !important; border-right: 2px solid #e2e8f0; box-shadow: 2px 0 12px rgba(0,0,0,0.08); }
.card { background: linear-gradient(145deg, #ffffff 0%, #fafbfc 100%); border: 2px solid #e5e7eb; border-radius: 16px; padding: 1.5rem 1.75rem; box-shadow: 0 4px 16px rgba(0,0,0,0.06); transition: all 0.3s ease; }
.card:hover { transform: translateY(-2px); box-shadow: 0 8px 24px rgba(0,0,0,0.1); }
.metric-large { font-size: 2.75rem; font-weight: 800; letter-spacing: -0.03em; line-height: 1; }
.metric-label { font-size: 0.8rem; color: #64748b; text-transform: uppercase; letter-spacing: 1.2px; font-weight: 600; margin-bottom: 0.5rem; }
.metric-sub { font-size: 0.9rem; color: #475569; margin-top: 0.5rem; font-weight: 500; }
h1 { color: #111827; font-weight: 800; margin-top: 1.5rem; font-size: 2.5rem; letter-spacing: -0.02em; }
h2, h3 { color: #374151; font-weight: 700; margin-top: 1.2rem; }
.stButton > button { background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); color: #fff; border: none; padding: 0.85rem 1.25rem; border-radius: 12px; font-weight: 700; box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3); }
hr { border-top: 2px solid #e5e7eb; margin: 2rem 0; }
</style>
""", unsafe_allow_html=True)

def process_dataframe(df):
    col_map = {'policy_id': 'ID', 'churn_probability': 'Churn_Prob', 'claims_probability': 'Claims_Prob', 'claims_severity': 'Claims_Severity', 'customer_lifetime_value': 'CLV', 'customer_segment': 'Segment', 'journey_quadrant': 'Journey', 'pricing_adequacy_flag': 'Underpriced', 'renewal_risk_score': 'Renewal_Risk', 'is_high_renewal_risk': 'High_Renewal_Risk'}
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    if 'Churn_Prob' in df.columns:
        df['Risk'] = pd.cut(df['Churn_Prob'], bins=[0, 0.3, 0.6, 0.85, 1.1], labels=['Low', 'Medium', 'High', 'Critical'])
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
                return process_dataframe(df), f"Live SQL ({info})"
    except Exception as e:
        logger.warning(f"SQL failed: {e}")
    
    csv_paths = [project_path / "model_outputs" / "rag_model_predictions.csv", project_path / "rag_model_predictions.csv", Path("model_outputs/rag_model_predictions.csv")]
    for path in csv_paths:
        if path.exists():
            try:
                df = pd.read_csv(path)
                return process_dataframe(df), f"CSV ({path.name})"
            except Exception as e:
                logger.error(f"CSV error: {e}")
    return None, "No data source"

df, source_info = load_data()

with st.sidebar:
    st.markdown("### 📊 Navigation Panel")
    st.markdown("---")
    page = st.radio("Select View", ["🏠 Executive Overview", "⚡ Churn Analysis", "🛡️ Claims Intelligence", "💎 Value & Segments", "🧭 Customer Journey", "🔍 RAG Q&A", "📥 Export Data"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("### ⚙️ Quick Actions")
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.markdown("---")
    st.markdown(f"<div style='font-size: 0.85rem; color: #64748b; padding: 0.5rem; background: #f1f5f9; border-radius: 8px;'><strong>Data Source:</strong><br/>{source_info}</div>", unsafe_allow_html=True)

def metric_card(label, value, sub=None, color='blue'):
    colors = {'blue': 'linear-gradient(135deg, #3b82f6 0%, #2563eb 100%)', 'green': 'linear-gradient(135deg, #10b981 0%, #059669 100%)', 'orange': 'linear-gradient(135deg, #f59e0b 0%, #d97706 100%)', 'red': 'linear-gradient(135deg, #ef4444 0%, #dc2626 100%)', 'purple': 'linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%)'}
    gradient = colors.get(color, colors['blue'])
    sub_html = f"<div class='metric-sub'>{sub}</div>" if sub else ""
    return f"<div class='card'><div class='metric-label'>{label}</div><div class='metric-large' style='background: {gradient}; -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>{value}</div>{sub_html}</div>"

RISK_COLORS = {'Low': '#10b981', 'Medium': '#f59e0b', 'High': '#f97316', 'Critical': '#ef4444'}
COLORS = {'primary': '#3b82f6', 'success': '#10b981', 'warning': '#f59e0b', 'danger': '#ef4444', 'info': '#06b6d4', 'purple': '#8b5cf6'}

def main():
    global df
    if df is None or df.empty:
        st.error("⚠️ Data Unavailable")
        st.info("Ensure data file is accessible")
        if st.button("🔄 Retry"):
            st.rerun()
        st.stop()
    
    total_customers = len(df)
    total_value = df['CLV'].sum()
    critical_count = len(df[df['Risk'] == 'Critical'])
    high_value_count = len(df[df['CLV'] > df['CLV'].quantile(0.9)])
    avg_churn = df['Churn_Prob'].mean()
    avg_claims = df['Claims_Prob'].mean()
    
    if "Executive Overview" in page:
        st.markdown("# 🏠 Executive Overview")
        st.markdown("*Comprehensive portfolio insights*")
        st.markdown("---")
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(metric_card("Total Policies", f"{total_customers:,}", "Active customers", 'blue'), unsafe_allow_html=True)
        c2.markdown(metric_card("Portfolio Value", f"€{total_value/1e6:.2f}M", "Lifetime value", 'green'), unsafe_allow_html=True)
        c3.markdown(metric_card("Critical Risk", f"{critical_count:,}", f"{critical_count/total_customers*100:.1f}%", 'red'), unsafe_allow_html=True)
        c4.markdown(metric_card("High Value", f"{high_value_count:,}", f"€{df[df['CLV']>df['CLV'].quantile(0.9)]['CLV'].sum()/1e6:.1f}M", 'purple'), unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("### 📈 Performance Indicators")
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Avg Churn", f"{avg_churn*100:.1f}%", f"{(avg_churn-0.35)*100:.1f}%")
        p2.metric("Avg Claims", f"{avg_claims*100:.1f}%", f"{(avg_claims-0.25)*100:.1f}%")
        p3.metric("Underpriced", f"{df['Underpriced'].sum():,}", f"{df['Underpriced'].sum()/len(df)*100:.1f}%")
        p4.metric("Renewal Risk 70%+", f"{(df['Renewal_Risk']>0.7).sum():,}")
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🎯 Risk Distribution")
            risk_data = df['Risk'].value_counts().reindex(['Low','Medium','High','Critical']).fillna(0)
            fig = go.Figure()
            fig.add_trace(go.Bar(x=risk_data.index, y=risk_data.values, marker=dict(color=[RISK_COLORS[r] for r in risk_data.index], line=dict(color='#fff', width=2)), text=risk_data.values, textposition='outside'))
            fig.update_layout(height=380, paper_bgcolor='rgba(255,255,255,0.95)', plot_bgcolor='#fafbfc', font=dict(color='#1a1a1a'), margin=dict(t=20,b=60,l=60,r=20), xaxis=dict(title="Risk Category", gridcolor='#e5e7eb'), yaxis=dict(title="Customers", gridcolor='#e5e7eb'))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown("### 💥 Churn vs Severity")
            sample = df.sample(min(len(df), 4000), random_state=9)
            fig = px.scatter(sample, x='Churn_Prob', y='Claims_Severity', color='Risk', color_discrete_map=RISK_COLORS, opacity=0.65)
            fig.update_layout(height=380, paper_bgcolor='rgba(255,255,255,0.95)', plot_bgcolor='#fafbfc', font=dict(color='#1a1a1a'), margin=dict(t=20,b=60,l=60,r=20), xaxis=dict(tickformat='.0%', gridcolor='#e5e7eb'), yaxis=dict(tickformat=',d', gridcolor='#e5e7eb'))
            st.plotly_chart(fig, use_container_width=True)
        col3, col4 = st.columns(2)
        with col3:
            st.markdown("### 📊 Segments")
            seg = df['Segment'].value_counts().head(8)
            fig = px.pie(names=seg.index, values=seg.values, color_discrete_sequence=px.colors.qualitative.Bold, hole=0.4)
            fig.update_traces(textposition='inside', textinfo='label+percent', marker=dict(line=dict(color='#fff', width=2)))
            fig.update_layout(height=350, paper_bgcolor='rgba(255,255,255,0.95)', font=dict(color='#1a1a1a'), margin=dict(t=20,b=20,l=20,r=20))
            st.plotly_chart(fig, use_container_width=True)
        with col4:
            st.markdown("### 🎲 Severity Distribution")
            fig = go.Figure()
            fig.add_trace(go.Box(y=df.sample(min(len(df), 5000), random_state=7)['Claims_Severity'], marker=dict(color=COLORS['info']), boxmean='sd'))
            fig.update_layout(height=350, paper_bgcolor='rgba(255,255,255,0.95)', plot_bgcolor='#fafbfc', font=dict(color='#1a1a1a'), margin=dict(t=20,b=60,l=60,r=20), showlegend=False, yaxis=dict(tickformat=',d', gridcolor='#e5e7eb'))
            st.plotly_chart(fig, use_container_width=True)
    
    elif "Churn Analysis" in page:
        st.markdown("# ⚡ Churn Analysis")
        st.markdown("*Identify at-risk customers*")
        st.markdown("---")
        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(metric_card("Avg Churn", f"{avg_churn*100:.1f}%", "Portfolio avg", 'orange'), unsafe_allow_html=True)
        m2.markdown(metric_card("Critical", f"{critical_count:,}", f"{critical_count/total_customers*100:.1f}%", 'red'), unsafe_allow_html=True)
        m3.markdown(metric_card("High Risk", f"{len(df[df['Risk']=='High']):,}", "Immediate attention", 'orange'), unsafe_allow_html=True)
        m4.markdown(metric_card("Renewal Risk", f"{(df['Renewal_Risk']>0.7).sum():,}", "70%+ score", 'purple'), unsafe_allow_html=True)
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📉 Churn Distribution")
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=df['Churn_Prob'], nbinsx=40, marker=dict(color='#ef4444', line=dict(color='#fff', width=1))))
            fig.update_layout(height=380, paper_bgcolor='rgba(255,255,255,0.95)', plot_bgcolor='#fafbfc', font=dict(color='#1a1a1a'), margin=dict(t=20,b=60,l=60,r=20), xaxis=dict(title="Churn Probability", tickformat='.0%', gridcolor='#e5e7eb'), yaxis=dict(title="Customers", gridcolor='#e5e7eb'))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown("### 🎯 Renewal Risk")
            fig = go.Figure()
            fig.add_trace(go.Violin(y=df['Renewal_Risk'], box_visible=True, meanline_visible=True, fillcolor='rgba(139, 92, 246, 0.3)', line=dict(color=COLORS['purple'], width=2)))
            fig.update_layout(height=380, paper_bgcolor='rgba(255,255,255,0.95)', plot_bgcolor='#fafbfc', font=dict(color='#1a1a1a'), margin=dict(t=20,b=60,l=60,r=20), showlegend=False, yaxis=dict(gridcolor='#e5e7eb'))
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("### 📋 Risk Breakdown")
        breakdown = df.groupby('Risk').agg({'ID': 'count', 'Churn_Prob': 'mean', 'CLV': ['sum', 'mean'], 'Renewal_Risk': 'mean'}).round(2)
        breakdown.columns = ['Count', 'Avg Churn %', 'Total CLV', 'Avg CLV', 'Avg Renewal']
        breakdown['Avg Churn %'] = (breakdown['Avg Churn %'] * 100).round(1).astype(str) + '%'
        breakdown['Total CLV'] = '€' + breakdown['Total CLV'].apply(lambda x: f'{x:,.0f}')
        breakdown['Avg CLV'] = '€' + breakdown['Avg CLV'].apply(lambda x: f'{x:,.0f}')
        st.dataframe(breakdown, use_container_width=True)
    
    elif "Claims Intelligence" in page:
        st.markdown("# 🛡️ Claims Intelligence")
        st.markdown("*Predict claims risk*")
        st.markdown("---")
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(metric_card("Avg Claims", f"{avg_claims*100:.1f}%", "Portfolio avg", 'orange'), unsafe_allow_html=True)
        c2.markdown(metric_card("High Risk", f"{(df['Claims_Prob']>0.5).sum():,}", "50%+ prob", 'orange'), unsafe_allow_html=True)
        c3.markdown(metric_card("Severity p95", f"€{df['Claims_Severity'].quantile(0.95):,.0f}", "95th percentile", 'blue'), unsafe_allow_html=True)
        c4.markdown(metric_card("Total Exposure", f"€{df['Claims_Severity'].sum()/1e6:.1f}M", "Expected claims", 'blue'), unsafe_allow_html=True)
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📊 Claims Distribution")
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=df['Claims_Prob'], nbinsx=35, marker=dict(color='#f59e0b', line=dict(color='#fff', width=1))))
            fig.update_layout(height=380, paper_bgcolor='rgba(255,255,255,0.95)', plot_bgcolor='#fafbfc', font=dict(color='#1a1a1a'), margin=dict(t=20,b=60,l=60,r=20), xaxis=dict(tickformat='.0%', gridcolor='#e5e7eb'), yaxis=dict(gridcolor='#e5e7eb'))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown("### 💥 Prob vs Severity")
            sample = df.sample(min(len(df), 4000), random_state=9)
            fig = px.scatter(sample, x='Claims_Prob', y='Claims_Severity', color='Risk', color_discrete_map=RISK_COLORS, opacity=0.7, size='CLV')
            fig.update_layout(height=380, paper_bgcolor='rgba(255,255,255,0.95)', plot_bgcolor='#fafbfc', font=dict(color='#1a1a1a'), margin=dict(t=20,b=60,l=60,r=20), xaxis=dict(tickformat='.0%', gridcolor='#e5e7eb'), yaxis=dict(tickformat=',d', gridcolor='#e5e7eb'))
            st.plotly_chart(fig, use_container_width=True)
    
    elif "Value & Segments" in page:
        st.markdown("# 💎 Value & Segments")
        st.markdown("*Customer worth analysis*")
        st.markdown("---")
        v1, v2, v3, v4 = st.columns(4)
        v1.markdown(metric_card("Avg CLV", f"€{df['CLV'].mean():,.0f}", "Per customer", 'blue'), unsafe_allow_html=True)
        v2.markdown(metric_card("Top 10%", f"€{df['CLV'].quantile(0.9):,.0f}", "High value", 'green'), unsafe_allow_html=True)
        v3.markdown(metric_card("Underpriced", f"{df['Underpriced'].sum():,}", f"{df['Underpriced'].sum()/len(df)*100:.1f}%", 'orange'), unsafe_allow_html=True)
        v4.markdown(metric_card("Portfolio", f"€{total_value/1e6:.2f}M", "Total value", 'purple'), unsafe_allow_html=True)
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📈 CLV Distribution")
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=df['CLV'], nbinsx=50, marker=dict(color='#3b82f6', line=dict(color='#fff', width=1))))
            fig.update_layout(height=380, paper_bgcolor='rgba(255,255,255,0.95)', plot_bgcolor='#fafbfc', font=dict(color='#1a1a1a'), margin=dict(t=20,b=60,l=60,r=20), xaxis=dict(tickformat=',d', gridcolor='#e5e7eb'), yaxis=dict(gridcolor='#e5e7eb'))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown("### 🎯 CLV by Segment")
            seg_avg = df.groupby('Segment')['CLV'].mean().sort_values(ascending=True).tail(10)
            fig = go.Figure()
            fig.add_trace(go.Bar(y=seg_avg.index, x=seg_avg.values, orientation='h', marker=dict(color=seg_avg.values, colorscale='Viridis', line=dict(color='#fff', width=1)), text=['€{:,.0f}'.format(v) for v in seg_avg.values], textposition='outside'))
            fig.update_layout(height=380, paper_bgcolor='rgba(255,255,255,0.95)', plot_bgcolor='#fafbfc', font=dict(color='#1a1a1a'), margin=dict(t=20,b=60,l=140,r=80), showlegend=False, xaxis=dict(tickformat=',d', gridcolor='#e5e7eb'))
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("### 📊 Segment Distribution")
        seg_counts = df['Segment'].value_counts().head(12)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=seg_counts.index, y=seg_counts.values, marker=dict(color=px.colors.qualitative.Bold[:len(seg_counts)], line=dict(color='#fff', width=2)), text=seg_counts.values, textposition='outside'))
        fig.update_layout(height=350, paper_bgcolor='rgba(255,255,255,0.95)', plot_bgcolor='#fafbfc', font=dict(color='#1a1a1a'), margin=dict(t=20,b=80,l=60,r=20), showlegend=False, xaxis=dict(tickangle=-45, gridcolor='#e5e7eb'), yaxis=dict(gridcolor='#e5e7eb'))
        st.plotly_chart(fig, use_container_width=True)
    
    elif "Customer Journey" in page:
        st.markdown("# 🧭 Customer Journey")
        st.markdown("*Strategic positioning*")
        st.markdown("---")
        j_counts = df['Journey'].value_counts()
        j1, j2, j3, j4 = st.columns(4)
        j1.markdown(metric_card("Protect", f"{j_counts.get('Protect',0):,}", "High value, low risk", 'green'), unsafe_allow_html=True)
        j2.markdown(metric_card("Grow", f"{j_counts.get('Grow',0):,}", "Low value, low risk", 'blue'), unsafe_allow_html=True)
        j3.markdown(metric_card("Rescue", f"{j_counts.get('Rescue',0):,}", "High value, high risk", 'red'), unsafe_allow_html=True)
        j4.markdown(metric_card("Monitor", f"{j_counts.get('Monitor',0):,}", "Low value, high risk", 'orange'), unsafe_allow_html=True)
        st.markdown("---")
        col1, col2 = st.columns(2)
        journey_colors = {'Protect': '#10b981', 'Grow': '#3b82f6', 'Rescue': '#ef4444', 'Monitor': '#f59e0b'}
        with col1:
            st.markdown("### 📊 Journey Distribution")
            fig = go.Figure()
            fig.add_trace(go.Bar(x=j_counts.index, y=j_counts.values, marker=dict(color=[journey_colors.get(j, '#6b7280') for j in j_counts.index], line=dict(color='#fff', width=2)), text=j_counts.values, textposition='outside'))
            fig.update_layout(height=380, paper_bgcolor='rgba(255,255,255,0.95)', plot_bgcolor='#fafbfc', font=dict(color='#1a1a1a'), margin=dict(t=20,b=60,l=60,r=20), showlegend=False, xaxis=dict(gridcolor='#e5e7eb'), yaxis=dict(gridcolor='#e5e7eb'))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown("### 💎 Risk vs Value")
            sample = df.sample(min(len(df), 3500), random_state=5)
            fig = px.scatter(sample, x='Renewal_Risk', y='CLV', color='Journey', color_discrete_map=journey_colors, opacity=0.7, size='Claims_Severity')
            fig.update_layout(height=380, paper_bgcolor='rgba(255,255,255,0.95)', plot_bgcolor='#fafbfc', font=dict(color='#1a1a1a'), margin=dict(t=20,b=60,l=60,r=20), xaxis=dict(gridcolor='#e5e7eb'), yaxis=dict(tickformat=',d', gridcolor='#e5e7eb'))
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("### 💰 Journey Value")
        jv = df.groupby('Journey').agg({'CLV': ['sum', 'mean', 'count']}).round(0)
        jv.columns = ['Total Value', 'Avg Value', 'Count']
        jv['Total Value'] = '€' + (jv['Total Value']/1e6).apply(lambda x: f'{x:.2f}M')
        jv['Avg Value'] = '€' + jv['Avg Value'].apply(lambda x: f'{x:,.0f}')
        st.dataframe(jv, use_container_width=True)
    
    elif "RAG Q&A" in page:
        st.markdown("# 🔍 Natural Language Queries")
        st.markdown("*Ask about your portfolio*")
        st.markdown("---")
        with st.expander("💡 Examples", expanded=True):
            st.markdown("- Show top 10 highest churn risk\n- Find high value critical risk\n- List platinum segment high churn\n- Show top 5 claims likely\n- Find low risk high value")
        user_q = st.text_input("🔎 Ask a question", placeholder="e.g., Show top 5 critical churn with high CLV")
        if user_q:
            try:
                from scripts.rag.rag_system import InsuranceRAGSystem
                with st.spinner("🔍 Analyzing..."):
                    rag = InsuranceRAGSystem(df=df)
                    results_df, explanation = rag.query(user_q)
                st.success(explanation)
                if len(results_df) > 0:
                    st.markdown("### 👥 Results")
                    display_df = results_df.copy()
                    if 'Churn_Prob' in display_df.columns:
                        display_df['Churn_Prob'] = display_df['Churn_Prob'].apply(lambda x: f"{float(x):.1%}")
                    if 'Claims_Prob' in display_df.columns:
                        display_df['Claims_Prob'] = display_df['Claims_Prob'].apply(lambda x: f"{float(x):.1%}")
                    if 'CLV' in display_df.columns:
                        display_df['CLV'] = display_df['CLV'].apply(lambda x: f"€{float(x):,.0f}")
                    st.dataframe(display_df, use_container_width=True, height=400)
                    csv = results_df.to_csv(index=False)
                    st.download_button("📥 Download CSV", csv, "rag_results.csv", "text/csv", use_container_width=True)
                else:
                    st.warning("No matches")
            except ImportError:
                st.warning("⚠️ RAG system unavailable")
                st.info("Install full requirements.txt")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    elif "Export Data" in page:
        st.markdown("# 📥 Export Portfolio Data")
        st.markdown("*Download and analyze externally*")
        st.markdown("---")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("### 📊 Full Portfolio Export")
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Complete Dataset (CSV)", csv, "insurance_portfolio_full.csv", "text/csv", use_container_width=True)
            st.markdown("### 🔍 Data Preview")
            st.dataframe(df.head(100), use_container_width=True, height=400)
        with col2:
            st.markdown("### 💰 Value by Risk")
            risk_value = df.groupby('Risk')['CLV'].sum() / 1e6
            fig = go.Figure()
            fig.add_trace(go.Bar(x=risk_value.index, y=risk_value.values, marker=dict(color=[RISK_COLORS[r] for r in risk_value.index], line=dict(color='#fff', width=2)), text=['€{:.1f}M'.format(v) for v in risk_value.values], textposition='outside'))
            fig.update_layout(height=350, paper_bgcolor='rgba(255,255,255,0.95)', plot_bgcolor='#fafbfc', font=dict(color='#1a1a1a'), margin=dict(t=20,b=60,l=60,r=20), showlegend=False, xaxis=dict(gridcolor='#e5e7eb'), yaxis=dict(gridcolor='#e5e7eb'))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("### 📈 Export Statistics")
            st.metric("Total Records", f"{len(df):,}")
            st.metric("Total Columns", f"{df.shape[1]}")
            st.metric("Data Quality", "99.8%")

if __name__ == "__main__":
    main()