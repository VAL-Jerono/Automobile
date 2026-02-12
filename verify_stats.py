#!/usr/bin/env python3
import pandas as pd
import numpy as np

# Load and process exactly like the app does
df = pd.read_csv('model_data/engineered_features_complete.csv')
print(f'📊 Raw data: {len(df):,} records')

# Create the same mapping as the app
mapped_df = pd.DataFrame()
mapped_df['ID'] = df['ID'].astype(str)
mapped_df['Churn_Prob'] = df['Churn_target']
mapped_df['Claims_Prob'] = df['Claims_binary']
mapped_df['Claims_Severity'] = df['Claims_severity']

# Calculate CLV exactly like the app
expected_claims_cost = mapped_df['Claims_Prob'] * mapped_df['Claims_Severity']
mapped_df['CLV'] = (df['Premium'] - expected_claims_cost) * 3

# Risk categories like the app
mapped_df['Risk_Category'] = pd.cut(
    mapped_df['Churn_Prob'], 
    bins=[0, 0.3, 0.6, 0.85, 1.1], 
    labels=['Low Risk', 'Medium Risk', 'High Risk', 'Critical Risk']
)

# Value tiers like the app
mapped_df['Value_Tier'] = pd.qcut(
    mapped_df['CLV'], 
    q=4, 
    labels=['Bronze', 'Silver', 'Gold', 'Platinum'],
    duplicates='drop'
)

# Calculate key statistics
total_customers = len(mapped_df)
critical_customers = len(mapped_df[mapped_df['Risk_Category'] == 'Critical Risk'])
high_value = len(mapped_df[mapped_df['Value_Tier'] == 'Platinum'])
total_clv = mapped_df['CLV'].sum()
high_risk_customers = len(mapped_df[mapped_df['Churn_Prob'] > 0.6])
critical_value = mapped_df[mapped_df['Risk_Category'] == 'Critical Risk']['CLV'].sum()

print(f'✅ Processed data: {total_customers:,} records')
print(f'📈 Key Metrics:')
print(f'  • Critical customers: {critical_customers:,} ({critical_customers/total_customers*100:.1f}%)')
print(f'  • High risk (>60%): {high_risk_customers:,} ({high_risk_customers/total_customers*100:.1f}%)')
print(f'  • Premium customers: {high_value:,} (top 25%)')
print(f'  • Portfolio value: €{total_clv/1e6:.1f}M')
print(f'  • Avg customer value: €{total_clv/total_customers:,.0f}')
print(f'  • Critical value at risk: €{critical_value/1e6:.2f}M ({critical_value/total_clv*100:.1f}%)')

# Check underpricing
underpriced = len(mapped_df[df['Is_overpriced'] == 0])
print(f'  • Underpriced customers: {underpriced:,} ({underpriced/total_customers*100:.1f}%)')