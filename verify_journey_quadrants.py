"""
Deep dive into Journey Quadrant Logic
"""

from sql_predictions_manager import SQLModelPredictionsManager
import pandas as pd
import numpy as np

mgr = SQLModelPredictionsManager()
if mgr.connect():
    df = mgr.get_all_predictions()
    mgr.disconnect()
    
    print('=' * 80)
    print('JOURNEY QUADRANT DEEP DIVE')
    print('=' * 80)
    
    print('\n1. UNIQUE VALUES IN DATABASE:')
    print(f'   Unique quadrants: {df["journey_quadrant"].unique()}')
    print(f'\n   Distribution:')
    for quad, count in df['journey_quadrant'].value_counts().items():
        pct = count / len(df) * 100
        print(f'      {quad}: {count:,} ({pct:.1f}%)')
    
    print('\n2. RENEWAL RISK SCORE ANALYSIS:')
    print(df['renewal_risk_score'].describe())
    print(f'\n   Key percentiles:')
    print(f'      P25: {df["renewal_risk_score"].quantile(0.25):.4f}')
    print(f'      P50 (Median): {df["renewal_risk_score"].quantile(0.50):.4f}')
    print(f'      P75: {df["renewal_risk_score"].quantile(0.75):.4f}')
    
    print('\n3. CLV ANALYSIS:')
    print(df['customer_lifetime_value'].describe())
    median_clv = df['customer_lifetime_value'].median()
    print(f'\n   Median CLV: €{median_clv:.2f}')
    
    print('\n4. CHECKING QUADRANT LOGIC:')
    print('   Expected logic:')
    print('      - Protect: Low renewal risk (< median) + High CLV (>= median)')
    print('      - Grow: Low renewal risk (< median) + Low CLV (< median)')
    print('      - Rescue: High renewal risk (>= median) + High CLV (>= median)')
    print('      - Monitor: High renewal risk (>= median) + Low CLV (< median)')
    
    median_risk = df['renewal_risk_score'].median()
    
    print(f'\n   Actual thresholds being used:')
    print(f'      Median Renewal Risk: {median_risk:.4f}')
    print(f'      Median CLV: €{median_clv:.2f}')
    
    # Manual calculation
    print('\n5. MANUAL CALCULATION OF EXPECTED QUADRANTS:')
    low_risk = df['renewal_risk_score'] < median_risk
    high_clv = df['customer_lifetime_value'] >= median_clv
    
    protect_expected = (low_risk & high_clv).sum()
    grow_expected = (low_risk & ~high_clv).sum()
    rescue_expected = (~low_risk & high_clv).sum()
    monitor_expected = (~low_risk & ~high_clv).sum()
    
    print(f'   Expected if using median thresholds:')
    print(f'      Protect (Low Risk + High CLV): {protect_expected:,}')
    print(f'      Grow (Low Risk + Low CLV): {grow_expected:,}')
    print(f'      Rescue (High Risk + High CLV): {rescue_expected:,}')
    print(f'      Monitor (High Risk + Low CLV): {monitor_expected:,}')
    
    print('\n6. SAMPLE RECORDS FROM EACH ACTUAL QUADRANT:')
    for quad in df['journey_quadrant'].unique():
        subset = df[df['journey_quadrant'] == quad][['policy_id', 'customer_lifetime_value', 'renewal_risk_score', 'churn_probability']]
        print(f'\n   {quad} (n={len(subset):,}):')
        print(f'      CLV range: €{subset["customer_lifetime_value"].min():.2f} - €{subset["customer_lifetime_value"].max():.2f}')
        print(f'      Renewal risk range: {subset["renewal_risk_score"].min():.4f} - {subset["renewal_risk_score"].max():.4f}')
        print(f'      Sample:')
        print(subset.head(3).to_string(index=False))
    
    print('\n7. CONCLUSION:')
    print('   The issue is that renewal_risk_score might all be 0 or have very few')
    print('   non-zero values, causing all customers to fall into just 2 quadrants.')
    print(f'\n   Checking: How many have renewal_risk > 0?')
    print(f'      renewal_risk_score > 0: {(df["renewal_risk_score"] > 0).sum():,}')
    print(f'      renewal_risk_score == 0: {(df["renewal_risk_score"] == 0).sum():,}')
    
else:
    print('❌ Failed to connect to database')
