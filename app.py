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
    """Load predictions and join with raw demographic data."""
    project_path = Path(__file__).parent
    
    # 1. Load Raw Data for Demographics
    df_raw = None
    raw_path = project_path / "Motor_vehicle_insurance_data.csv"
    if raw_path.exists():
        try:
            df_raw = pd.read_csv(raw_path, sep=';')
            # Basic cleaning for raw data
            if 'Date_birth' in df_raw.columns:
                df_raw['Date_birth'] = pd.to_datetime(df_raw['Date_birth'], errors='coerce')
        except Exception as e:
            logger.warning(f"Could not load raw data: {e}")

    # 2. Load Predictions
    df_preds = None
    source = "None"
    
    # Try SQL first
    try:
        from utils.sql_predictions_manager import SQLModelPredictionsManager
        manager = SQLModelPredictionsManager()
        if manager.connect():
            df_preds = manager.get_all_predictions()
            source = "Live SQL"
            manager.disconnect()
    except Exception:
        pass

    # Try CSV if SQL failed
    if df_preds is None:
        csv_path = project_path / "model_outputs" / "rag_model_predictions.csv"
        if csv_path.exists():
            df_preds = pd.read_csv(csv_path)
            source = f"Static CSV ({csv_path.name})"

    if df_preds is None:
        return None, "No prediction data found."

    # Process and Join
    df_preds = process_dataframe(df_preds)
    
    if df_raw is not None:
        # Join predictions with raw data on ID
        df = pd.merge(df_preds, df_raw, on='ID', how='left')
        return df, f"{source} + Metadata"
    
    return df_preds, source


# Load data globally for sidebar access
df, source_info = load_data()


# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

with st.sidebar:
    st.markdown("### 📊 Journey")
    st.markdown("---")
    
    page = st.radio(
        "Select Phase",
        [
            "Life Cycle Flow",    
            "1. Acquisition (Get)", 
            "2. Maintenance (Keep)",
            "3. Churn & Risks (Exit)",
            "RAG Q&A",
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

    # LIFE CYCLE FLOW PAGE
    if page == "Life Cycle Flow":
        st.markdown("# 🔄 Customer Life Cycle Flow")
        
        # Strategic Column at top
        si1, si2 = st.columns([3, 1])
        with si1:
            st.markdown("""
            This dashboard follows the **Customer Journey** approach: 
            1. **Get** (Identify high-value acquisition targets)
            2. **Keep** (Monitor active portfolio health and pricing)
            3. **Exit** (Mitigate churn and claims impact)
            """)
        with si2:
            st.info("**Strategy:** Maximize portfolio value by shifting customers from 'Monitor' to 'Protect'.")

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(metric_card("Acquisition", f"{total_customers:,}", "Total active policies", 'green'), unsafe_allow_html=True)
        c2.markdown(metric_card("Portfolio Value", f"€{total_value/1e6:.1f}M", "Total CLV", 'blue'), unsafe_allow_html=True)
        c3.markdown(metric_card("Maintenance", f"{len(df[df['Risk'] == 'Low']):,}", "Low-risk stable base", 'green'), unsafe_allow_html=True)
        c4.markdown(metric_card("Exit Risk", f"{critical_count:,}", f"{(critical_count/total_customers)*100:.1f}% at-risk", 'red'), unsafe_allow_html=True)

        st.markdown("---")
        
        # Journey Map Visual
        st.markdown("### 🗺️ The Portfolio Journey Map")
        j_counts = df['Journey'].value_counts()
        
        # Improved journey display
        f1, f2, f3 = st.columns(3)
        with f1:
            st.markdown("#### 🟢 1. Acquisition (Get)")
            st.markdown("Focus on high-growth potential.")
            st.metric("Growth Targets", f"{j_counts.get('Grow', 0):,}")
        with f2:
            st.markdown("#### 🔵 2. Maintenance (Keep)")
            st.markdown("Ensure stability and pricing depth.")
            st.metric("Stable Base", f"{j_counts.get('Protect', 0):,}")
        with f3:
            st.markdown("#### 🔴 3. Retention (Exit)")
            st.markdown("Immediate churn intervention.")
            st.metric("Rescue Priority", f"{j_counts.get('Rescue', 0):,}")
            
        st.markdown("---")
        
        # High level scatter: Churn vs Value
        scatter = px.scatter(
            df.sample(min(len(df), 4000), random_state=42),
            x='Churn_Prob', y='CLV', color='Risk',
            color_discrete_map={'Low':palette['green'],'Medium':palette['amber'],'High':palette['orange'],'Critical':palette['red']},
            size='Claims_Severity', opacity=0.75,
            hover_data=['ID', 'Segment', 'Journey'],
            labels={'Churn_Prob':'Retention Risk','CLV':'Customer Value (€)'}
        )
        scatter.update_layout(height=450, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=palette['text']), margin=dict(t=20,b=40,l=10,r=10), legend_title_text='Risk Status')
        st.markdown("### 📊 Portfolio Value vs. Retention Risk")
        st.plotly_chart(scatter, use_container_width=True)

    # ACQUISITION PAGE
    elif page == "1. Acquisition (Get)":
        st.markdown("# 🎯 Acquisition Strategy: Who Are We Getting?")
        
        ast1, ast2 = st.columns([2, 1])
        with ast1:
            with st.expander("❓ Questions this answers", expanded=True):
                st.markdown("""
                - **Age Profile:** What age groups are we attracting?
                - **Risk Profile:** Are we heavy on high-risk vehicle types?
                - **Geography:** Where are our customers coming from?
                """)
        with ast2:
            st.warning("**Strategic Insight:** Young drivers (18-24) represent our longest acquisition horizon but carry higher initial premium sensitivity.")

        col1, col2 = st.columns(2)
        
        with col1:
            # Age Distribution (Notebook finding: peak around mid-30s)
            if 'Date_birth' in df.columns:
                ref_date = pd.Timestamp('2026-01-14')
                df['Age'] = (ref_date - df['Date_birth']).dt.days / 365.25
                age_data = df[(df['Age'] >= 18) & (df['Age'] <= 85)]['Age']
                
                age_fig = px.histogram(age_data, x='Age', nbins=40, 
                                     color_discrete_sequence=[palette['blue']],
                                     title="Demographic Snapshot: Age of Policyholders")
                age_fig.add_vrect(x0=18, x1=25, fillcolor="red", opacity=0.1, annotation_text="High Growth", annotation_position="top left")
                age_fig.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=palette['text']))
                st.plotly_chart(age_fig, use_container_width=True)

        with col2:
            # Vehicle Type Distribution
            if 'Type_risk' in df.columns:
                risk_counts = df['Type_risk'].value_counts().reset_index()
                risk_counts.columns = ['Vehicle Type', 'Count']
                
                risk_fig = px.bar(risk_counts, y='Vehicle Type', x='Count', orientation='h', 
                                 color='Count', color_continuous_scale='Blues',
                                 title="Targeting: Vehicle Type Acquisition")
                risk_fig.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=palette['text']), showlegend=False)
                st.plotly_chart(risk_fig, use_container_width=True)

        # Acquisition Channel and Geography
        col3, col4 = st.columns(2)
        with col3:
            if 'Area' in df.columns:
                area_counts = df['Area'].value_counts().reset_index()
                area_fig = px.pie(area_counts, names='Area', values='count', hole=0.5,
                                 color_discrete_sequence=[palette['blue'], palette['green']],
                                 title="Market Location: Urban vs Rural")
                area_fig.update_layout(height=350, font=dict(color=palette['text']))
                st.plotly_chart(area_fig, use_container_width=True)
        
        with col4:
            if 'Distribution_channel' in df.columns:
                chan_counts = df['Distribution_channel'].value_counts().reset_index()
                chan_fig = px.bar(chan_counts, x='Distribution_channel', y='count', 
                                 color='count', color_continuous_scale='Greens',
                                 title="Sales Channel Performance")
                chan_fig.update_layout(height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=palette['text']), showlegend=False)
                st.plotly_chart(chan_fig, use_container_width=True)

    # MAINTENANCE PAGE
    elif page == "2. Maintenance (Keep)":
        st.markdown("# 💎 Maintenance Strategy: Ensuring Portfolio Value")
        
        mst1, mst2 = st.columns([2, 1])
        with mst1:
            with st.expander("❓ Questions this answers", expanded=True):
                st.markdown("""
                - **Segmentation:** Which segments (Platinum/Gold) provide the stable value base?
                - **Pricing:** Are we maintaining premium adequacy against predicted claims?
                - **Value:** How is CLV distributed across the book?
                """)
        with mst2:
            st.success("**Strategic Insight:** Platinum segments yield 3x the CLV of Bronze. Maintenance focus should be on upselling Silver/Gold to Platinum.")

        m1, m2 = st.columns(2)
        with m1:
            # CLV by Segment
            seg_clv = df.groupby('Segment')['CLV'].mean().sort_values(ascending=False).reset_index()
            fig_clv = px.bar(seg_clv, x='Segment', y='CLV', color='CLV',
                            color_continuous_scale='Viridis',
                            title="Portfolio Value: Avg CLV by Segment")
            fig_clv.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=palette['text']))
            st.plotly_chart(fig_clv, use_container_width=True)

        with m2:
            # Premium Distribution
            if 'Premium' in df.columns:
                q99 = df['Premium'].quantile(0.99)
                prem_data = df[df['Premium'] <= q99]['Premium']
                fig_prem = px.histogram(prem_data, x='Premium', nbins=40, color_discrete_sequence=[palette['green']],
                                      title="Revenue Stream: Premium Amount Distribution")
                fig_prem.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=palette['text']))
                st.plotly_chart(fig_prem, use_container_width=True)

        # Pricing Adequacy
        st.markdown("### ⚖️ Pricing Strategy & Margin Analysis")
        p1, p2 = st.columns([2, 1])
        with p1:
            # Scatter of CLV vs Premium
            if 'Premium' in df.columns:
                scatter_p = px.scatter(df.sample(min(len(df), 3000), random_state=1), 
                                     x='Premium', y='CLV', color='Underpriced',
                                     color_discrete_map={0: palette['blue'], 1: palette['red']},
                                     hover_data=['ID', 'Segment'],
                                     opacity=0.6, title="Margin Health: Premium vs Customer Value")
                scatter_p.update_layout(height=450, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=palette['text']))
                st.plotly_chart(scatter_p, use_container_width=True)
        with p2:
            st.markdown("<br><br>", unsafe_allow_html=True)
            up_count = df['Underpriced'].sum()
            st.markdown(metric_card("Underpriced", f"{up_count:,}", f"{(up_count/total_customers)*100:.1f}% Margin Risk", 'red'), unsafe_allow_html=True)
            st.info("""
            **Priority Action:** 
            Policies in **Red** are underpriced relative to their predicted risk. 
            Initiate price adjustments or limit renewal discounts for these targets.
            """)

    # RETENTION / EXIT PAGE
    elif page == "3. Churn & Risks (Exit)":
        st.markdown("# ⚠️ Exit Strategy: Mitigating Churn & Claims")
        
        est1, est2 = st.columns([2, 1])
        with est1:
            with st.expander("❓ Questions this answers", expanded=True):
                st.markdown("""
                - **Churn Drivers:** Which age groups or segments are most likely to leave?
                - **Claim Spikes:** What is our exposure to high-severity predicted claims?
                - **Retention:** Who are the 'Rescue' targets we must prioritize?
                """)
        with est2:
            st.error("**Strategic Insight:** High-severity claims are clustered in the 'Critical' risk group. Prioritize renewal reviews for 'Rescue' quadrant immediately.")

        r1, r2 = st.columns(2)
        with r1:
            # Churn by Age Group
            if 'Age' in df.columns:
                age_bins = [18, 25, 35, 45, 55, 65, 100]
                age_labels = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
                df['Age_Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels)
                
                churn_age = df.groupby('Age_Group')['Churn_Prob'].mean().reset_index()
                fig_churn_age = px.bar(churn_age, x='Age_Group', y='Churn_Prob', color='Churn_Prob',
                                      color_continuous_scale='Reds',
                                      title="Churn Forecast: Exit Probability by Age")
                fig_churn_age.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=palette['text']))
                st.plotly_chart(fig_churn_age, use_container_width=True)
            else:
                # Fallback to simple churn prob histogram
                fig_churn = px.histogram(df, x='Churn_Prob', nbins=30, color_discrete_sequence=[palette['red']])
                fig_churn.update_layout(title="Churn Probability Distribution", height=380, font=dict(color=palette['text']))
                st.plotly_chart(fig_churn, use_container_width=True)

        with r2:
            # Claims Severity Analysis
            sev_box = px.box(df, y='Claims_Severity', x='Risk', color='Risk', 
                            color_discrete_map={'Low':palette['green'],'Medium':palette['amber'],'High':palette['orange'],'Critical':palette['red']},
                            title="Loss Exposure: Claims Severity by Risk Status")
            sev_box.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=palette['text']), showlegend=False)
            st.plotly_chart(sev_box, use_container_width=True)

        st.markdown("### 🚨 High Value Retention (Rescue) Priority List")
        rescue_targets = df[df['Journey'] == 'Rescue'].sort_values(by=['CLV', 'Churn_Prob'], ascending=[False, False]).head(10)
        
        if not rescue_targets.empty:
            display_rescue = rescue_targets[['ID', 'Segment', 'CLV', 'Churn_Prob', 'Claims_Prob']].copy()
            display_rescue['CLV'] = display_rescue['CLV'].apply(lambda x: f"€{x:,.0f}")
            display_rescue['Churn_Prob'] = display_rescue['Churn_Prob'].apply(lambda x: f"{x:.1%}")
            display_rescue['Claims_Prob'] = display_rescue['Claims_Prob'].apply(lambda x: f"{x:.1%}")
            st.dataframe(display_rescue, use_container_width=True)
            st.caption("Top 10 Strategy: Direct intervention recommended for these high-value/low-retention customers.")
        else:
            st.success("No critical rescue targets identified in current view.")

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
    
if __name__ == "__main__":
    main()
