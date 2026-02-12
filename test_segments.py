import pandas as pd
from pathlib import Path

# Load and check the real data
real_data_path = Path('model_data/engineered_features_complete.csv')
if real_data_path.exists():
    df = pd.read_csv(real_data_path)
    
    # Create segments using the same logic as the app
    churn_median = df['Churn_target'].median()
    clv = df['Premium'] * 3.5  # CLV approximation
    clv_median = clv.median()
    
    def get_segment(row):
        if row['Churn_target'] < churn_median and row['CLV'] >= clv_median:
            return 'NEW_CUSTOMER'
        elif row['Churn_target'] < churn_median and row['CLV'] < clv_median:
            return 'DEVELOPING'
        elif row['Churn_target'] >= churn_median and row['CLV'] >= clv_median:
            return 'ESTABLISHED'
        else:
            return 'LOYAL_VETERAN'
    
    df['CLV'] = clv
    df['Segment'] = df.apply(get_segment, axis=1)
    
    print('Segment distribution:')
    segment_counts = df['Segment'].value_counts()
    print(segment_counts)
    print()
    
    print('Value by segment:')
    segment_value = df.groupby('Segment')['CLV'].sum()
    for segment, value in segment_value.items():
        print(f'{segment}: €{value/1e6:.1f}M')
else:
    print('Data file not found')