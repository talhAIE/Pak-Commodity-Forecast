"""
Comprehensive Data Analysis Script

Note: This script should be run from the project root directory.
Paths are relative to project root.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

# Get project root directory (parent of scripts folder)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'Data')

print("=" * 80)
print("COMPREHENSIVE DATA ANALYSIS")
print("=" * 80)

# 1. Read Export Data Files
print("\n1. EXPORT DATA FILES")
print("-" * 80)

rice = pd.read_excel(os.path.join(DATA_DIR, 'Pakistan_Exports_1006_Rice_2010_2025.xlsx'))
cotton = pd.read_excel(os.path.join(DATA_DIR, 'Pakistan_Exports_520512_cotton_2010_2025.xlsx'))
copper = pd.read_excel(os.path.join(DATA_DIR, 'Pakistan_Exports_7403_Copper__2010_2025.xlsx'))

print(f"\nRICE: {len(rice)} rows")
print(f"  - Date column: refPeriodId")
print(f"  - Export Value: fobvalue (primaryValue if fobvalue missing)")
print(f"  - Weight: netWgt")
print(f"  - First date: {rice['refPeriodId'].iloc[0]}")
print(f"  - Last date: {rice['refPeriodId'].iloc[-1]}")
print(f"  - Missing fobvalue: {rice['fobvalue'].isnull().sum()}")
print(f"  - Zero fobvalue: {(rice['fobvalue'] == 0).sum()}")
print(f"  - Missing netWgt: {rice['netWgt'].isnull().sum()}")

print(f"\nCOTTON: {len(cotton)} rows")
print(f"  - First date: {cotton['refPeriodId'].iloc[0]}")
print(f"  - Last date: {cotton['refPeriodId'].iloc[-1]}")
print(f"  - Missing fobvalue: {cotton['fobvalue'].isnull().sum()}")
print(f"  - Zero fobvalue: {(cotton['fobvalue'] == 0).sum()}")
print(f"  - Missing netWgt: {cotton['netWgt'].isnull().sum()}")

print(f"\nCOPPER: {len(copper)} rows")
print(f"  - First date: {copper['refPeriodId'].iloc[0]}")
print(f"  - Last date: {copper['refPeriodId'].iloc[-1]}")
print(f"  - Missing fobvalue: {copper['fobvalue'].isnull().sum()}")
print(f"  - Zero fobvalue: {(copper['fobvalue'] == 0).sum()}")
print(f"  - Missing netWgt: {copper['netWgt'].isnull().sum()}")

# Check for missing months
print("\n2. DATE COVERAGE ANALYSIS")
print("-" * 80)

# Convert refPeriodId to datetime for each commodity
rice_dates = pd.to_datetime(rice['refPeriodId'].astype(str), format='%Y%m%d')
cotton_dates = pd.to_datetime(cotton['refPeriodId'].astype(str), format='%Y%m%d')
copper_dates = pd.to_datetime(copper['refPeriodId'].astype(str), format='%Y%m%d')

# Create expected date range (2010-01-01 to 2025-08-01, monthly)
expected_dates = pd.date_range(start='2010-01-01', end='2025-08-01', freq='MS')
print(f"Expected months: {len(expected_dates)} (Jan 2010 to Aug 2025)")
print(f"  First: {expected_dates[0]}")
print(f"  Last: {expected_dates[-1]}")

# Find missing months for each commodity
rice_months = set(rice_dates.dt.to_period('M'))
cotton_months = set(cotton_dates.dt.to_period('M'))
copper_months = set(copper_dates.dt.to_period('M'))
expected_months = set(expected_dates.to_period('M'))

rice_missing = expected_months - rice_months
cotton_missing = expected_months - cotton_months
copper_missing = expected_months - copper_months

print(f"\nRICE missing months: {len(rice_missing)}")
if len(rice_missing) > 0 and len(rice_missing) <= 10:
    print(f"  {sorted(rice_missing)}")

print(f"\nCOTTON missing months: {len(cotton_missing)}")
if len(cotton_missing) > 0 and len(cotton_missing) <= 10:
    print(f"  {sorted(cotton_missing)}")

print(f"\nCOPPER missing months: {len(copper_missing)}")
if len(copper_missing) > 0 and len(copper_missing) <= 10:
    print(f"  {sorted(copper_missing)}")

# 3. External Data Sources
print("\n3. EXTERNAL DATA SOURCES")
print("-" * 80)

usd_pkr = pd.read_csv(os.path.join(DATA_DIR, 'USD_PKR_Exchange_Rate_2010-01-01_2025-12-31.csv'))
oil = pd.read_csv(os.path.join(DATA_DIR, 'Brent_Oil_Prices_2010_2025.csv'))
confidence = pd.read_csv(os.path.join(DATA_DIR, 'US_Consumer_Confidence_2010_2025.csv'))

# Clean date columns
usd_pkr['Date'] = pd.to_datetime(usd_pkr['Date'], format='%d/%m/%Y', errors='coerce')
oil['Date'] = pd.to_datetime(oil['Date'], format='%d/%m/%Y', errors='coerce')
confidence['Date'] = pd.to_datetime(confidence['Date'], format='%d/%m/%Y', errors='coerce')

# Remove rows with missing dates
usd_pkr = usd_pkr.dropna(subset=['Date'])
oil = oil.dropna(subset=['Date'])
confidence = confidence.dropna(subset=['Date'])

print(f"\nUSD/PKR Exchange Rate: {len(usd_pkr)} rows")
print(f"  - Date range: {usd_pkr['Date'].min()} to {usd_pkr['Date'].max()}")
print(f"  - Missing values: {usd_pkr['USD_PKR_Rate'].isnull().sum()}")
print(f"  - Unique dates: {usd_pkr['Date'].nunique()}")

print(f"\nBrent Oil Price: {len(oil)} rows")
print(f"  - Date range: {oil['Date'].min()} to {oil['Date'].max()}")
print(f"  - Missing values: {oil['Brent_Oil_Price'].isnull().sum()}")
print(f"  - Unique dates: {oil['Date'].nunique()}")

# Check for missing months in oil data
oil_months = set(oil['Date'].dt.to_period('M'))
oil_missing = expected_months - oil_months
print(f"  - Missing months: {len(oil_missing)}")
if len(oil_missing) > 0 and len(oil_missing) <= 10:
    print(f"    {sorted(oil_missing)}")

print(f"\nUS Consumer Confidence: {len(confidence)} rows")
print(f"  - Date range: {confidence['Date'].min()} to {confidence['Date'].max()}")
print(f"  - Missing values: {confidence['US_Consumer_Confidence_Index'].isnull().sum()}")
print(f"  - Unique dates: {confidence['Date'].nunique()}")

conf_months = set(confidence['Date'].dt.to_period('M'))
conf_missing = expected_months - conf_months
print(f"  - Missing months: {len(conf_missing)}")
if len(conf_missing) > 0 and len(conf_missing) <= 10:
    print(f"    {sorted(conf_missing)}")

# 4. Sample Data Values
print("\n4. SAMPLE DATA VALUES")
print("-" * 80)
print("\nRICE Sample:")
print(rice[['refPeriodId', 'fobvalue', 'netWgt']].head(10))
print("\nCOTTON Sample:")
print(cotton[['refPeriodId', 'fobvalue', 'netWgt']].head(10))
print("\nCOPPER Sample:")
print(copper[['refPeriodId', 'fobvalue', 'netWgt']].head(10))

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
