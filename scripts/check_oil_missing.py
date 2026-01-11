import pandas as pd
import os

# Get project root directory (parent of scripts folder)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'Data')

# Check oil data missing months in detail
oil = pd.read_csv(os.path.join(DATA_DIR, 'Brent_Oil_Prices_2010_2025.csv'))
oil['Date'] = pd.to_datetime(oil['Date'], format='%d/%m/%Y', errors='coerce')
oil = oil.dropna(subset=['Date'])

expected_dates = pd.date_range(start='2010-01-01', end='2025-08-01', freq='MS')
expected_months = set(expected_dates.to_period('M'))
oil_months = set(oil['Date'].dt.to_period('M'))
oil_missing = sorted(expected_months - oil_months)

print(f"OIL DATA MISSING MONTHS: {len(oil_missing)} out of {len(expected_months)}")
print("\nMissing months:")
for month in oil_missing:
    print(f"  {month}")

# Check if there are consecutive missing months
print("\nChecking consecutive missing months...")
oil_sorted = sorted(oil['Date'].dt.to_period('M'))
gaps = []
for i in range(len(oil_sorted)-1):
    current = oil_sorted[i]
    next_month = oil_sorted[i+1]
    gap = (next_month - current).n
    if gap > 1:
        gaps.append((current, next_month, gap-1))

if gaps:
    print("\nGaps found:")
    for gap in gaps[:10]:  # Show first 10 gaps
        print(f"  Between {gap[0]} and {gap[1]}: {gap[2]} missing month(s)")

# Show sample of oil data to see pattern
print("\nOIL DATA SAMPLE (first 20 rows):")
print(oil[['Date', 'Brent_Oil_Price']].head(20))

print("\nOIL DATA SAMPLE (showing gaps):")
# Show rows where date changes significantly
oil_sorted_df = oil.sort_values('Date')
oil_sorted_df['Month'] = oil_sorted_df['Date'].dt.to_period('M')
oil_sorted_df['PrevMonth'] = oil_sorted_df['Month'].shift(1)
oil_sorted_df['MonthDiff'] = (oil_sorted_df['Month'] - oil_sorted_df['PrevMonth']).apply(lambda x: x.n if pd.notna(x) else 0)

print(oil_sorted_df[oil_sorted_df['MonthDiff'] > 1][['Date', 'Brent_Oil_Price', 'MonthDiff']].head(15))
