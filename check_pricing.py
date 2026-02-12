#!/usr/bin/env python3
import pandas as pd

df = pd.read_csv('model_data/engineered_features_complete.csv')
print('Is_overpriced column values:')
print(df['Is_overpriced'].value_counts())
print('\nPercentage breakdown:')
overpriced_pct = df['Is_overpriced'].value_counts(normalize=True) * 100
print(overpriced_pct)

print(f'\nSummary:')
print(f'Overpriced (1): {(df["Is_overpriced"] == 1).sum():,} customers ({(df["Is_overpriced"] == 1).mean()*100:.1f}%)')
print(f'Not overpriced (0): {(df["Is_overpriced"] == 0).sum():,} customers ({(df["Is_overpriced"] == 0).mean()*100:.1f}%)')