"""
Insurance Analytics Platform v7.0
==================================
Clean, professional dashboard for insurance portfolio management.
Focuses on actionable insights and risk management.

- Simple, scannable landing page
- Intuitive sidebar navigation
- Real-time predictions from ML models
- Production-ready deployment

Author: Insurance Analytics Team
Date: January 2026
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

st.set_page_config(
    page_title="Insurance Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clean, modern styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');

    * { font-family: 'Space Grotesk', sans-serif; }

    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at 20% 20%, rgba(88, 28, 135, 0.15), transparent 25%),
                    radial-gradient(circle at 80% 10%, rgba(14, 165, 233, 0.18), transparent 30%),
                    radial-gradient(circle at 40% 80%, rgba(16, 185, 129, 0.12), transparent 28%),
                    linear-gradient(135deg, #0b1220 0%, #0f172a 45%, #0b1120 100%);
        color: #e2e8f0;
    }

    [data-testid="stSidebarContent"] {
        background: rgba(10, 14, 26, 0.9) !important;
        border-right: 1px solid rgba(148, 163, 184, 0.18);
        box-shadow: 4px 0 18px rgba(0,0,0,0.35);
    }

    .card {
        background: linear-gradient(145deg, rgba(31, 41, 55, 0.82), rgba(17, 24, 39, 0.92));
        border: 1px solid rgba(148, 163, 184, 0.25);
        border-radius: 14px;
        padding: 1.3rem 1.4rem;
        box-shadow: 0 12px 32px rgba(0,0,0,0.25);
    }

    .metric-large {
        font-size: 2.4rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        line-height: 1.05;
    }

    .metric-label {
        font-size: 0.78rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.35rem;
    }

    .badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 0.25rem 0.6rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        border: 1px solid rgba(255,255,255,0.12);
        background: rgba(255,255,255,0.06);
    }

    .status-critical { color: #f97373; }
    .status-high { color: #fb923c; }
    .status-medium { color: #facc15; }
    .status-low { color: #34d399; }

    h1, h2, h3 { margin-top: 1.2rem; margin-bottom: 0.8rem; font-weight: 700; letter-spacing: -0.01em; }

    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #22c55e 0%, #10b981 100%);
        color: #0b1120;
        border: none;
        padding: 0.72rem;
        border-radius: 10px;
        font-weight: 700;
        letter-spacing: 0.01em;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px) scale(1.01);
        box-shadow: 0 10px 24px rgba(16, 185, 129, 0.35);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA LOADING
# ============================================================================

def process_dataframe(df):
    """Standardize and process the predictions dataframe."""
    # Standardize column names
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

    # Only rename columns that exist
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Calculate risk categories if Churn_Prob exists
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
        Path("/app/Automobile/model_outputs/rag_model_predictions.csv") # Common Streamlit Cloud path
    ]

    tried_paths = []
    for path in csv_paths:
        path_str = str(path.absolute()) if path.is_absolute() else str(path)
        tried_paths.append(path_str)
        if path.exists():
            try:
                df = pd.read_csv(path)
                logger.info(f"✅ Loaded data from CSV: {path}")
                return process_dataframe(df), f"Static CSV ({path.name})"
            except Exception as e:
                logger.error(f"Error reading CSV {path}: {e}")
                return None, f"Error reading {path.name}: {str(e)}"

    return None, f"Checked: {', '.join(tried_paths)}"


# Load data globally for sidebar access
df, source_info = load_data()


# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

with st.sidebar:
    st.markdown("### 📊 Navigation")
    st.markdown("---")

    page = st.radio(
        "Select View",
        [
            "Flow",                # condensed story
            "Will they leave?",    # churn / renewal risk
            "Will they claim?",    # claims frequency & severity
            "What are they worth?",# CLV & segments
            "Where are they headed?", # journey quadrants
            "RAG Q&A",             # optional Q&A hook
            "Export"
        ],
        label_visibility="collapsed",
        key="nav"
    )

    st.markdown("---")
    st.markdown("### 🔧 Actions")

    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.markdown(f"<div style='font-size: 0.8rem; color: #64748b;'>Data Source: {source_info}</div>",
                unsafe_allow_html=True)
    st.markdown("<div style='font-size: 0.8rem; color: #64748b;'>v7.0 • Production Ready</div>",
                unsafe_allow_html=True)


# ============================================================================
# MAIN APPLICATION
# ============================================================================


def main():
    # Use data loaded globally
    global df, source_info

    if df is None or df.empty:
        st.error("⚠️ Data Source Unavailable")
        st.write(f"**Diagnostic Info:** {source_info}")
        st.markdown("""
        The application could not connect to the MySQL database or find a backup CSV file in the repository.

        ### 🔧 How to fix:
        1. **Start MySQL:** Open XAMPP Control Panel and start MySQL.
        2. **Populate Data:** Run the export script to sync data from the research notebook:
           ```bash
           python Automobile/scripts/database/export_predictions_to_sql.py
           ```
        3. **Fallback:** If you don't want to use MySQL, ensure `rag_model_predictions.csv` exists in the `Automobile/model_outputs/` folder.
        """)
        if st.button("🔄 Try Again"):
            st.rerun()
        st.stop()

    palette = {
        'bg': '#0b1120',
        'text': '#e2e8f0',
        'green': '#34d399',
        'amber': '#f59e0b',
        'orange': '#fb923c',
        'red': '#f97373',
        'blue': '#38bdf8'
    }

    total_customers = len(df)
    total_value = df['CLV'].sum()
    critical_count = len(df[df['Risk'] == 'Critical'])
    high_value_count = len(df[df['CLV'] > df['CLV'].quantile(0.9)])
    avg_churn = df['Churn_Prob'].mean()
    avg_claims_prob = df['Claims_Prob'].mean()

    def metric_card(label, value, sub=None, tone=None):
        color = palette.get(tone, palette['text']) if tone else palette['text']
        sub_html = f"<div style='font-size:0.9rem;color:#94a3b8'>{sub}</div>" if sub else ""
        return f"""
        <div class="card">
            <div class="metric-label">{label}</div>
            <div class="metric-large" style="color:{color}">{value}</div>
            {sub_html}
        </div>
        """

    # FLOW PAGE
    if page == "Flow":
        st.markdown("# 🔥 Customer Analytics Flow")
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(metric_card("Policies", f"{total_customers:,}", None, 'green'), unsafe_allow_html=True)
        c2.markdown(metric_card("Portfolio", f"€{total_value/1e6:.1f}M", "CLV total", 'blue'), unsafe_allow_html=True)
        c3.markdown(metric_card("Critical Risk", f"{critical_count:,}", f"{critical_count/total_customers*100:.1f}% flagged", 'red'), unsafe_allow_html=True)
        c4.markdown(metric_card("High Value", f"€{df[df['CLV']>df['CLV'].quantile(0.9)]['CLV'].sum()/1e6:.1f}M", f"{high_value_count:,} customers", 'green'), unsafe_allow_html=True)

        st.markdown("---")
        g1, g2 = st.columns(2)

        with g1:
            risk_data = df['Risk'].value_counts().reindex(['Low','Medium','High','Critical']).fillna(0)
            fig = go.Figure(go.Bar(
                x=risk_data.index,
                y=risk_data.values,
                marker_color=[palette['green'], palette['amber'], palette['orange'], palette['red']],
                text=risk_data.values,
                textposition='outside'
            ))
            fig.update_layout(height=360, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=palette['text']), margin=dict(t=20,b=40,l=40,r=10), showlegend=False)
            st.markdown("### Risk Mix")
            st.plotly_chart(fig, use_container_width=True)

        with g2:
            scatter = px.scatter(
                df.sample(min(len(df), 4000), random_state=42),
                x='Churn_Prob', y='CLV', color='Risk',
                color_discrete_map={'Low':palette['green'],'Medium':palette['amber'],'High':palette['orange'],'Critical':palette['red']},
                size='Claims_Severity', opacity=0.75,
                labels={'Churn_Prob':'Churn','CLV':'Value (€)'}
            )
            scatter.update_layout(height=360, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=palette['text']), margin=dict(t=20,b=40,l=10,r=10), legend_title_text='Risk')
            st.markdown("### Churn vs Value")
            st.plotly_chart(scatter, use_container_width=True)

        g3, g4 = st.columns(2)
        with g3:
            claims_fig = px.box(df.sample(min(len(df), 4000), random_state=7), y='Claims_Severity', points='suspectedoutliers', color_discrete_sequence=[palette['blue']])
            claims_fig.update_layout(height=320, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=palette['text']), margin=dict(t=10,b=30,l=10,r=10), showlegend=False)
            st.markdown("### Severity Spread")
            st.plotly_chart(claims_fig, use_container_width=True)

        with g4:
            seg = df['Segment'].value_counts()
            seg_fig = px.pie(names=seg.index, values=seg.values, color=seg.index,
                             color_discrete_sequence=[palette['green'], palette['amber'], palette['orange'], palette['blue']])
            seg_fig.update_layout(height=320, paper_bgcolor='rgba(0,0,0,0)', font=dict(color=palette['text']), showlegend=True, margin=dict(t=10,b=10,l=10,r=10))
            st.markdown("### Segments")
            st.plotly_chart(seg_fig, use_container_width=True)

    # RETENTION PAGE
    elif page == "Will they leave?":
        st.markdown("# ⚡ Will They Leave?")
        a1, a2, a3 = st.columns(3)
        a1.markdown(metric_card("Avg Churn", f"{avg_churn*100:.1f}%", None, 'orange'), unsafe_allow_html=True)
        a2.markdown(metric_card("Critical", f"{critical_count:,}", f"{critical_count/total_customers*100:.1f}%", 'red'), unsafe_allow_html=True)
        a3.markdown(metric_card("Renewal Risk 70%+", f"{(df['Renewal_Risk']>0.7).sum():,}", None, 'amber'), unsafe_allow_html=True)

        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            churn_hist = px.histogram(df, x='Churn_Prob', nbins=30, color_discrete_sequence=[palette['red']])
            churn_hist.update_layout(height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=palette['text']), margin=dict(t=20,b=40,l=40,r=10))
            st.markdown("### Churn Probability")
            st.plotly_chart(churn_hist, use_container_width=True)
        with c2:
            renewal = px.box(df, y='Renewal_Risk', color_discrete_sequence=[palette['amber']])
            renewal.update_layout(height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=palette['text']), margin=dict(t=20,b=40,l=10,r=10), showlegend=False)
            st.markdown("### Renewal Risk")
            st.plotly_chart(renewal, use_container_width=True)

    # CLAIMS PAGE
    elif page == "Will they claim?":
        st.markdown("# 🛡️ Will They Claim?")
        b1, b2, b3 = st.columns(3)
        b1.markdown(metric_card("Avg Claims Prob", f"{avg_claims_prob*100:.1f}%", None, 'amber'), unsafe_allow_html=True)
        b2.markdown(metric_card("High Claims Prob", f"{(df['Claims_Prob']>0.5).sum():,}", "Above 50%", 'orange'), unsafe_allow_html=True)
        b3.markdown(metric_card("Severity p95", f"€{df['Claims_Severity'].quantile(0.95):,.0f}", None, 'blue'), unsafe_allow_html=True)

        st.markdown("---")
        d1, d2 = st.columns(2)
        with d1:
            claims_hist = px.histogram(df, x='Claims_Prob', nbins=30, color_discrete_sequence=[palette['amber']])
            claims_hist.update_layout(height=340, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=palette['text']), margin=dict(t=20,b=40,l=40,r=10))
            st.markdown("### Claims Probability")
            st.plotly_chart(claims_hist, use_container_width=True)
        with d2:
            sev_scatter = px.scatter(df.sample(min(len(df), 5000), random_state=9), x='Claims_Prob', y='Claims_Severity', color='Risk',
                                     color_discrete_map={'Low':palette['green'],'Medium':palette['amber'],'High':palette['orange'],'Critical':palette['red']},
                                     opacity=0.7)
            sev_scatter.update_layout(height=340, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=palette['text']), margin=dict(t=20,b=40,l=10,r=10))
            st.markdown("### Probability vs Severity")
            st.plotly_chart(sev_scatter, use_container_width=True)

    # VALUE PAGE
    elif page == "What are they worth?":
        st.markdown("# 💎 What Are They Worth?")
        v1, v2, v3 = st.columns(3)
        v1.markdown(metric_card("Avg CLV", f"€{df['CLV'].mean():,.0f}", None, 'blue'), unsafe_allow_html=True)
        v2.markdown(metric_card("Top 10% CLV", f"€{df['CLV'].quantile(0.9):,.0f}", None, 'green'), unsafe_allow_html=True)
        v3.markdown(metric_card("Underpriced", f"{df['Underpriced'].sum():,}", "Loss-making risk", 'orange'), unsafe_allow_html=True)

        st.markdown("---")
        e1, e2 = st.columns(2)
        with e1:
            clv_hist = px.histogram(df, x='CLV', nbins=40, color_discrete_sequence=[palette['blue']])
            clv_hist.update_layout(height=340, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=palette['text']), margin=dict(t=20,b=40,l=40,r=10))
            st.markdown("### CLV Distribution")
            st.plotly_chart(clv_hist, use_container_width=True)
        with e2:
            seg_bar = px.bar(df.groupby('Segment')['CLV'].mean().reset_index(), x='Segment', y='CLV', color='Segment',
                             color_discrete_sequence=[palette['green'], palette['amber'], palette['orange'], palette['blue']])
            seg_bar.update_layout(height=340, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=palette['text']), margin=dict(t=20,b=40,l=10,r=10), showlegend=False)
            st.markdown("### CLV by Segment")
            st.plotly_chart(seg_bar, use_container_width=True)

    # JOURNEY PAGE
    elif page == "Where are they headed?":
        st.markdown("# 🧭 Where Are They Headed?")
        j_counts = df['Journey'].value_counts()
        j1, j2 = st.columns(2)
        j1.markdown(metric_card("Protect", f"{j_counts.get('Protect',0):,}", None, 'green'), unsafe_allow_html=True)
        j1.markdown(metric_card("Grow", f"{j_counts.get('Grow',0):,}", None, 'blue'), unsafe_allow_html=True)
        j2.markdown(metric_card("Rescue", f"{j_counts.get('Rescue',0):,}", None, 'orange'), unsafe_allow_html=True)
        j2.markdown(metric_card("Monitor", f"{j_counts.get('Monitor',0):,}", None, 'amber'), unsafe_allow_html=True)

        st.markdown("---")
        q1, q2 = st.columns([1,1])
        with q1:
            journey_fig = px.bar(j_counts.reset_index(), x='Journey', y='count', color='Journey',
                                 color_discrete_sequence=[palette['green'], palette['blue'], palette['orange'], palette['amber']])
            journey_fig.update_layout(height=340, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=palette['text']), margin=dict(t=20,b=40,l=40,r=10), showlegend=False, xaxis_title='', yaxis_title='Count')
            st.markdown("### Journey Quadrant Counts")
            st.plotly_chart(journey_fig, use_container_width=True)
        with q2:
            risk_value = px.scatter(df.sample(min(len(df), 4000), random_state=5), x='Renewal_Risk', y='CLV', color='Journey',
                                    color_discrete_sequence=[palette['green'], palette['blue'], palette['orange'], palette['amber']], opacity=0.75)
            risk_value.update_layout(height=340, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=palette['text']), margin=dict(t=20,b=40,l=10,r=10))
            st.markdown("### Risk vs Value")
            st.plotly_chart(risk_value, use_container_width=True)

    # RAG PAGE
    elif page == "RAG Q&A":
        st.markdown("# 🔎 RAG Q&A")
        st.markdown("Ask natural language questions about your customer portfolio. The system will query the database and provide insights.")

        # Example questions
        with st.expander("💡 Example Questions"):
            st.markdown("""
            - Show top 10 customers with highest churn risk
            - Find high value customers in critical risk
            - List customers in platinum segment with high churn
            - Show top 5 customers likely to make claims
            - Find low risk customers with high value
            - Show bronze segment customers in monitor quadrant
            - List top 20 customers by lifetime value
            """)

        user_q = st.text_input("Ask a question", placeholder="e.g., Show top 5 critical churn policies with high CLV")

        if user_q:
            try:
                # Try to import RAG system (optional feature)
                from scripts.rag.rag_system import InsuranceRAGSystem

                with st.spinner("🔍 Analyzing portfolio data..."):
                    # Pass the pre-loaded dataframe to the RAG system
                    # This avoids DB connection errors on Streamlit Cloud
                    rag = InsuranceRAGSystem(df=df)
                    results_df, explanation = rag.query(user_q)

                # Display explanation
                st.markdown("### 📊 Results")
                st.info(explanation)

                # Display results table
                if len(results_df) > 0:
                    st.markdown("### 👥 Customer Details")

                    # Format the dataframe for display
                    display_df = results_df.copy()

                    # Apply formatting based on column names (standardized by RAG system)
                    if 'Churn_Prob' in display_df.columns:
                        display_df['Churn_Prob'] = display_df['Churn_Prob'].apply(lambda x: f"{float(x):.1%}")
                    if 'Claims_Prob' in display_df.columns:
                        display_df['Claims_Prob'] = display_df['Claims_Prob'].apply(lambda x: f"{float(x):.1%}")
                    if 'CLV' in display_df.columns:
                        display_df['CLV'] = display_df['CLV'].apply(lambda x: f"€{float(x):,.0f}")

                    # Define pretty names for columns
                    pretty_cols = {
                        'ID': 'Policy ID',
                        'Churn_Prob': 'Churn Risk',
                        'Claims_Prob': 'Claims Risk',
                        'CLV': 'Lifetime Value',
                        'Segment': 'Segment',
                        'Journey': 'Journey'
                    }

                    # Rename only existing columns
                    display_df = display_df.rename(columns=pretty_cols)

                    st.dataframe(display_df, use_container_width=True)

                    # Add download button
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name=f"rag_query_results.csv",
                        mime="text/csv",
                    )
                else:
                    st.warning("No customers match your query criteria.")

            except ImportError as import_err:
                st.warning("⚠️ **RAG system not available** - Missing dependencies")
                st.info("""
                The natural language query feature requires additional packages not installed in this deployment.

                **To enable this feature:**
                1. Use full `requirements.txt` instead of `requirements-cloud.txt`
                2. This will increase build time but enable AI-powered queries

                **Alternative:** Use the filters in other sections to explore data.
                """)
                logger.warning(f"RAG system import failed: {import_err}")
            except Exception as e:
                st.error(f"❌ Error processing query: {str(e)}")
                logger.error(f"RAG query error: {e}", exc_info=True)

    # EXPORT PAGE
    elif page == "Export":
        st.markdown("# ⬇️ Export")

        col1, col2 = st.columns([2, 1])

        with col1:
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download predictions (CSV)",
                data=csv,
                file_name="model_predictions_export.csv",
                mime="text/csv",
                use_container_width=True
            )
            st.dataframe(df.head(200))

        with col2:
            st.markdown("### Value at Risk")
            risk_value = df.groupby('Risk')['CLV'].sum() / 1e6
            fig = go.Figure(data=[
                go.Bar(
                    x=risk_value.index,
                    y=risk_value.values,
                    marker_color=['#10b981', '#f59e0b', '#f97316', '#ef4444'],
                    text=[f'€{v:.1f}M' for v in risk_value.values],
                    textposition='outside'
                )
            ])
            fig.update_layout(
                height=350,
                showlegend=False,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e2e8f0'),
                xaxis=dict(title="", gridcolor='rgba(255,255,255,0.1)'),
                yaxis=dict(title="Value (€M)", gridcolor='rgba(255,255,255,0.1)'),
                margin=dict(t=10, b=30, l=40, r=10)
            )
            st.plotly_chart(fig, use_container_width=True)

    # PAGE: RISK ANALYSIS
    elif page == "Risk Analysis":
        st.markdown("# 🚨 Risk Analysis")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("### Risk Matrix")
            risk_counts = df['Risk'].value_counts()
            fig = go.Figure(data=[
                go.Pie(
                    labels=risk_counts.index,
                    values=risk_counts.values,
                    marker=dict(colors=['#10b981', '#f59e0b', '#f97316', '#ef4444']),
                    hole=0.3
                )
            ])
            fig.update_layout(
                height=400,
                showlegend=True,
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e2e8f0'),
                legend=dict(bgcolor='rgba(0,0,0,0)'),
                margin=dict(t=10, b=10, l=10, r=10)
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### Churn vs Value Scatter")
            sample_df = df.sample(min(3000, len(df)))
            fig = px.scatter(
                sample_df,
                x='Churn_Prob',
                y='CLV',
                color='Risk',
                size='Claims_Severity',
                color_discrete_map={
                    'Low': '#10b981',
                    'Medium': '#f59e0b',
                    'High': '#f97316',
                    'Critical': '#ef4444'
                },
                labels={'Churn_Prob': 'Churn Probability', 'CLV': 'Customer Value (€)'}
            )
            fig.update_layout(
                height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(15,23,42,0.6)',
                font=dict(color='#e2e8f0'),
                xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                legend=dict(bgcolor='rgba(0,0,0,0)'),
                margin=dict(t=10, b=30, l=40, r=10)
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("### Risk Breakdown")
        risk_table = df['Risk'].value_counts().reset_index()
        risk_table.columns = ['Risk Level', 'Count']
        risk_table['% of Portfolio'] = (risk_table['Count'] / len(df) * 100).round(1)
        risk_table['Avg CLV (€)'] = df.groupby('Risk').apply(lambda x: f"€{x['CLV'].mean():.0f}")
        st.dataframe(risk_table.set_index('Risk Level'), use_container_width=True)

    # PAGE: SEGMENTS
    elif page == "Segments":
        st.markdown("# 📈 Customer Segments")

        segment_data = df['Segment'].value_counts().head(15)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Segment Distribution")
            fig = px.pie(
                values=segment_data.values,
                names=segment_data.index,
                hole=0.4,
                color_discrete_sequence=px.colors.sequential.Blues_r
            )
            fig.update_layout(
                height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e2e8f0'),
                legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=10)),
                margin=dict(t=10, b=10, l=10, r=10)
            )
            fig.update_traces(textfont_size=10)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### Average Value by Segment")
            segment_avg = df.groupby('Segment')['CLV'].mean().sort_values(ascending=False).head(15)
            fig = go.Figure(data=[
                go.Bar(
                    y=segment_avg.index,
                    x=segment_avg.values,
                    orientation='h',
                    marker_color='#667eea',
                    text=[f'€{v:.0f}' for v in segment_avg.values],
                    textposition='outside'
                )
            ])
            fig.update_layout(
                height=400,
                showlegend=False,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e2e8f0'),
                xaxis=dict(title="Avg CLV (€)", gridcolor='rgba(255,255,255,0.1)'),
                yaxis=dict(title=""),
                margin=dict(t=10, b=30, l=120, r=10)
            )
            st.plotly_chart(fig, use_container_width=True)

    # PAGE: HIGH-RISK CUSTOMERS
    elif page == "High-Risk":
        st.markdown("# 🚨 Critical & High-Risk Customers")

        risk_filter = st.selectbox("Show:", ["Critical", "Critical + High", "All"])

        if risk_filter == "Critical":
            display_df = df[df['Risk'] == 'Critical']
        elif risk_filter == "Critical + High":
            display_df = df[df['Risk'].isin(['Critical', 'High'])]
        else:
            display_df = df

        display_df = display_df.sort_values('Churn_Prob', ascending=False)

        st.markdown(f"### {len(display_df):,} Customers")
        st.metric("Total Value at Risk", f"€{display_df['CLV'].sum()/1e6:.1f}M")

        st.dataframe(
            display_df[['ID', 'Segment', 'Churn_Prob', 'CLV', 'Risk']].head(500),
            use_container_width=True,
            height=500
        )

    # PAGE: EXPORT
    elif page == "Export":
        st.markdown("# 📥 Export Data")

        st.markdown("### Select Data to Download")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("📊 Full Portfolio", use_container_width=True):
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Full Dataset",
                    data=csv,
                    file_name="insurance_portfolio.csv",
                    mime="text/csv"
                )

        with col2:
            if st.button("🚨 Critical Only", use_container_width=True):
                critical = df[df['Risk'] == 'Critical']
                csv = critical.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Critical Customers",
                    data=csv,
                    file_name="critical_customers.csv",
                    mime="text/csv"
                )

        st.markdown("---")
        st.markdown("### Data Summary")
        summary = {
            'Total Customers': f"{len(df):,}",
            'Total Value': f"€{df['CLV'].sum()/1e6:.1f}M",
            'Critical Customers': f"{len(df[df['Risk'] == 'Critical']):,}",
            'Avg Churn Risk': f"{df['Churn_Prob'].mean()*100:.1f}%",
            'Data Records': f"{df.shape[0]:,}"
        }

        for key, value in summary.items():
            st.metric(key, value)


if __name__ == "__main__":
    main()