"""
Data Merging and Preprocessing Script
Combines all export data with external drivers and handles missing values

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
print("DATA MERGING AND PREPROCESSING")
print("=" * 80)

# CREATE MASTER DATE INDEX (2010-01-01 to 2025-08-01, monthly)
print("\n1. Creating master date index...")
master_dates = pd.date_range(start='2010-01-01', end='2025-08-01', freq='MS')
print(f"   Created {len(master_dates)} monthly dates")
print(f"   From {master_dates[0]} to {master_dates[-1]}")


# LOAD AND PROCESS EXPORT DATA

print("\n2. Loading export data files...")

def load_export_data(filepath, commodity_name, hs_code):
    """Load and process export data from Excel file"""
    df = pd.read_excel(filepath)
    
    # Convert refPeriodId to datetime
    df['Date'] = pd.to_datetime(df['refPeriodId'].astype(str), format='%Y%m%d')
    
    # Extract key columns
    export_df = pd.DataFrame({
        'Date': df['Date'],
        'Commodity': commodity_name,
        'HS_Code': hs_code,
        'Export_Value_USD': df['fobvalue'].fillna(df.get('primaryValue', 0)),  # Use fobvalue, fallback to primaryValue
        'Weight_kg': df['netWgt'].fillna(0)  # Fill missing weights with 0
    })
    
    # Ensure Export_Value_USD is not negative
    export_df['Export_Value_USD'] = export_df['Export_Value_USD'].clip(lower=0) #set any negative value to 0
    export_df['Weight_kg'] = export_df['Weight_kg'].clip(lower=0)
    
    return export_df

# Load all three commodities
rice = load_export_data(os.path.join(DATA_DIR, 'Pakistan_Exports_1006_Rice_2010_2025.xlsx'), 'Rice', 1006)
cotton = load_export_data(os.path.join(DATA_DIR, 'Pakistan_Exports_520512_cotton_2010_2025.xlsx'), 'Cotton Yarn', 520512)
copper = load_export_data(os.path.join(DATA_DIR, 'Pakistan_Exports_7403_Copper__2010_2025.xlsx'), 'Copper', 7403)

print(f"   Rice: {len(rice)} rows")
print(f"   Cotton: {len(cotton)} rows")
print(f"   Copper: {len(copper)} rows")

# Combine all export data
exports = pd.concat([rice, cotton, copper], ignore_index=True)
print(f"   Total export records: {len(exports)}")


# HANDLE MISSING MONTHS FOR EXPORTS (Fill with 0 = No Trade)
print("\n3. Handling missing months for exports...")

# Create complete date-commodity combinations
commodities = ['Rice', 'Cotton Yarn', 'Copper']
hs_codes = [1006, 520512, 7403]

# Create full date-commodity grid
date_commodity_grid = []
for date in master_dates:
    for i, commodity in enumerate(commodities):
        date_commodity_grid.append({
            'Date': date,
            'Commodity': commodity,
            'HS_Code': hs_codes[i]
        })

full_grid = pd.DataFrame(date_commodity_grid)

# Merge with actual export data
exports_complete = full_grid.merge(
    exports,
    on=['Date', 'Commodity', 'HS_Code'],
    how='left'
)

# Fill missing export values with 0 (no trade occurred)
exports_complete['Export_Value_USD'] = exports_complete['Export_Value_USD'].fillna(0)
exports_complete['Weight_kg'] = exports_complete['Weight_kg'].fillna(0)

print(f"   Complete export data: {len(exports_complete)} rows (should be {len(master_dates) * 3})")
print(f"   Missing months filled with 0 (no trade): {(exports_complete['Export_Value_USD'] == 0).sum()} rows")


# LOAD AND PROCESS EXTERNAL DATA SOURCES

print("\n4. Loading external data sources...")

# USD/PKR Exchange Rate
usd_pkr = pd.read_csv(os.path.join(DATA_DIR, 'USD_PKR_Exchange_Rate_2010-01-01_2025-12-31.csv'))
usd_pkr['Date'] = pd.to_datetime(usd_pkr['Date'], format='%d/%m/%Y', errors='coerce')
usd_pkr = usd_pkr.dropna(subset=['Date'])
usd_pkr = usd_pkr.rename(columns={'USD_PKR_Rate': 'USD_PKR'})
usd_pkr = usd_pkr[['Date', 'USD_PKR']]
print(f"   USD/PKR: {len(usd_pkr)} rows")

# Brent Oil Price
oil = pd.read_csv(os.path.join(DATA_DIR, 'Brent_Oil_Prices_2010_2025.csv'))
oil['Date'] = pd.to_datetime(oil['Date'], format='%d/%m/%Y', errors='coerce')
oil = oil.dropna(subset=['Date'])
oil = oil.rename(columns={'Brent_Oil_Price': 'Oil_Price'})
oil = oil[['Date', 'Oil_Price']]
print(f"   Oil Price: {len(oil)} rows (before interpolation)")

# US Consumer Confidence
confidence = pd.read_csv(os.path.join(DATA_DIR, 'US_Consumer_Confidence_2010_2025.csv'))
confidence['Date'] = pd.to_datetime(confidence['Date'], format='%d/%m/%Y', errors='coerce')
confidence = confidence.dropna(subset=['Date'])
confidence = confidence.rename(columns={'US_Consumer_Confidence_Index': 'US_Confidence'})
confidence = confidence[['Date', 'US_Confidence']]
print(f"   US Confidence: {len(confidence)} rows")


# CREATE COMPLETE EXTERNAL DATA FOR ALL DATES

print("\n5. Creating complete external data timeline...")

# Create base dataframe with all dates
external_data = pd.DataFrame({'Date': master_dates})

# Merge each external dataset
external_data = external_data.merge(usd_pkr, on='Date', how='left')
external_data = external_data.merge(oil, on='Date', how='left')
external_data = external_data.merge(confidence, on='Date', how='left')

print(f"   External data rows: {len(external_data)}")


#  HANDLE MISSING VALUES IN EXTERNAL DATA

print("\n6. Handling missing values in external data...")

print(f"   Missing values before imputation:")
print(f"     USD_PKR: {external_data['USD_PKR'].isnull().sum()}")
print(f"     Oil_Price: {external_data['Oil_Price'].isnull().sum()}")
print(f"     US_Confidence: {external_data['US_Confidence'].isnull().sum()}")

# USD/PKR and US_Confidence should have no missing values, but handle if any
external_data['USD_PKR'] = external_data['USD_PKR'].ffill().bfill() #forward fill and backward fill to handle missing values
external_data['US_Confidence'] = external_data['US_Confidence'].ffill().bfill()

# Oil Price: Use linear interpolation (average of continous price data)
external_data['Oil_Price'] = external_data['Oil_Price'].interpolate(method='linear')

# If interpolation fails at edges, use forward/backward fill
external_data['Oil_Price'] = external_data['Oil_Price'].ffill().bfill()

print(f"   Missing values after imputation:")
print(f"     USD_PKR: {external_data['USD_PKR'].isnull().sum()}")
print(f"     Oil_Price: {external_data['Oil_Price'].isnull().sum()}")
print(f"     US_Confidence: {external_data['US_Confidence'].isnull().sum()}")


# MERGE EXPORTS WITH EXTERNAL DATA

print("\n7. Merging exports with external data...")

# Merge exports with external data
final_dataset = exports_complete.merge(external_data, on='Date', how='left')

print(f"   Final dataset rows: {len(final_dataset)}")
print(f"   Expected rows: {len(master_dates) * 3}")


# DATA VALIDATION AND CLEANUP

print("\n8. Validating merged dataset...")

# Sort by Date and Commodity
final_dataset = final_dataset.sort_values(['Date', 'Commodity']).reset_index(drop=True)

# Verify all dates are present
date_check = final_dataset.groupby('Date').size()
if len(date_check) == len(master_dates):
    print("   [OK] All dates present")
else:
    print(f"   [WARNING] Missing dates: {len(master_dates) - len(date_check)}")

# Verify all commodities are present for each date
commodity_check = final_dataset.groupby('Date')['Commodity'].count()
if (commodity_check == 3).all():
    print("   [OK] All commodities present for each date")
else:
    print(f"   [WARNING] Missing commodities on some dates: {(commodity_check != 3).sum()} dates")

# Check for any remaining missing values
missing_check = final_dataset.isnull().sum()
if missing_check.sum() == 0:
    print("   [OK] No missing values in final dataset")
else:
    print(f"   [WARNING] Remaining missing values:\n{missing_check[missing_check > 0]}")

# Data type checks
print("\n9. Data type summary:")
print(final_dataset.dtypes)


# EXPORT CLEAN DATASET

print("\n10. Exporting clean dataset...")

# Export to CSV 
output_file = os.path.join(PROJECT_ROOT, 'merged_export_dataset_2010_2025.csv')
final_dataset.to_csv(output_file, index=False)
print(f"   [OK] Saved to: {output_file}")

# Also create a wide format version
print("\n11. Creating wide format version...")
wide_dataset = final_dataset.pivot_table(
    index='Date',
    columns='Commodity',
    values=['Export_Value_USD', 'Weight_kg'],
    aggfunc='first'
).reset_index()

# Flatten column names
wide_dataset.columns = [f"{col[1]}_{col[0]}" if col[1] else col[0] for col in wide_dataset.columns]
wide_dataset = wide_dataset.rename(columns={'Date_': 'Date'})

# Merge with external data
wide_dataset = wide_dataset.merge(external_data, on='Date', how='left')

wide_output_file = os.path.join(PROJECT_ROOT, 'merged_export_dataset_wide_2010_2025.csv')
wide_dataset.to_csv(wide_output_file, index=False)
print(f"   [OK] Saved to: {wide_output_file}")


# SUMMARY STATISTICS

print("\n" + "=" * 80)
print("DATASET SUMMARY")
print("=" * 80)

print("\nFinal Dataset Shape:")
print(f"  Rows: {len(final_dataset)}")
print(f"  Columns: {len(final_dataset.columns)}")

print("\nDate Range:")
print(f"  From: {final_dataset['Date'].min()}")
print(f"  To: {final_dataset['Date'].max()}")
print(f"  Total months: {final_dataset['Date'].nunique()}")

print("\nCommodities:")
print(final_dataset['Commodity'].value_counts())

print("\nExport Value Statistics (USD):")
print(final_dataset.groupby('Commodity')['Export_Value_USD'].describe())

print("\nExternal Drivers Statistics:")
print(external_data[['USD_PKR', 'Oil_Price', 'US_Confidence']].describe())

print("\nZero Export Values (No Trade):")
zero_exports = final_dataset[final_dataset['Export_Value_USD'] == 0]
print(f"  Total rows with zero exports: {len(zero_exports)}")
print(f"  By commodity:")
print(zero_exports['Commodity'].value_counts())

print("\n" + "=" * 80)
print("DATA PREPROCESSING COMPLETE!")
print("=" * 80)

# Display first few rows
print("\nFirst 10 rows of final dataset:")
print(final_dataset.head(10))
