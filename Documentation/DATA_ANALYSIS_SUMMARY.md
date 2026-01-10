# Comprehensive Data Analysis & Merging Strategy

## 📊 Dataset Overview

### Time Period
- **Start**: January 1, 2010
- **End**: August 1, 2025
- **Expected Months**: 188 months (Jan 2010 to Aug 2025)

---

## 📁 1. EXPORT DATA (Primary Targets)

### 1.1 Rice (HS Code: 1006)
- **Rows**: 188
- **Date Coverage**: Complete (188/188 months) ✅
- **Date Format**: `refPeriodId` (YYYYMMDD format, e.g., 20100101)
- **Key Columns**:
  - `fobvalue`: Export value in USD (target variable) - **No missing values**
  - `netWgt`: Weight in kg - **5 missing values**
- **Missing Data Strategy**: 
  - Missing `netWgt`: Forward fill or interpolate (not critical for forecasting)
  - Missing months: N/A (no missing months)

### 1.2 Cotton Yarn (HS Code: 520512)
- **Rows**: 188
- **Date Coverage**: Complete (188/188 months) ✅
- **Date Format**: `refPeriodId` (YYYYMMDD format)
- **Key Columns**:
  - `fobvalue`: Export value in USD (target variable) - **No missing values**
  - `netWgt`: Weight in kg - **No missing values**
- **Missing Data Strategy**: 
  - No missing data issues

### 1.3 Copper (HS Code: 7403)
- **Rows**: 182 (6 missing months)
- **Date Coverage**: 182/188 months ❌
- **Date Format**: `refPeriodId` (YYYYMMDD format)
- **Key Columns**:
  - `fobvalue`: Export value in USD (target variable) - **No missing values**
  - `netWgt`: Weight in kg - **No missing values**
- **Missing Months**:
  - 2010-06 (June 2010)
  - 2015-12 (December 2015)
  - 2016-03 (March 2016)
  - 2016-06 (June 2016)
  - 2016-07 (July 2016)
  - 2016-10 (October 2016)
- **Missing Data Strategy**: 
  - **Fill with 0** for `fobvalue` and `netWgt` (no trade occurred)
  - This represents actual business reality - no export in those months

---

## 🌍 2. EXTERNAL DATA SOURCES (Features/Drivers)

### 2.1 USD/PKR Exchange Rate
- **Rows**: 188
- **Date Coverage**: Complete (188/188 months) ✅
- **Date Format**: `DD/MM/YYYY` (e.g., 01/01/2010)
- **Missing Values**: None
- **Missing Data Strategy**: 
  - **No missing values** - ready to use
  - For future dates: Use forward fill or linear interpolation if needed

### 2.2 Brent Oil Price
- **Rows**: 161 (27 missing months) ⚠️
- **Date Coverage**: 161/188 months
- **Date Format**: `DD/MM/YYYY`
- **Missing Months**: 27 months scattered throughout the period
  - Notable gaps: 2010-08, 2011-05, 2012-01, 2012-04, 2012-07, 2013-09, 2013-12, 2014-06, 2015-02, 2015-03, 2015-11, 2016-05, 2017-01, 2017-10, 2018-04, 2018-07, 2019-09, 2019-12, 2020-03, 2020-11, 2021-08, 2022-05, 2023-01, 2023-10, 2024-09, 2024-12, 2025-06
- **Missing Data Strategy**:
  - **Method 1 (Recommended)**: Linear interpolation between known values
  - **Method 2**: Forward fill from previous known value
  - **Method 3**: Use monthly average (median of surrounding months)
  - **Rationale**: Oil prices are continuous and don't drop to zero, so interpolation is appropriate

### 2.3 US Consumer Confidence Index
- **Rows**: 188
- **Date Coverage**: Complete (188/188 months) ✅
- **Date Format**: `DD/MM/YYYY`
- **Missing Values**: None
- **Missing Data Strategy**: 
  - **No missing values** - ready to use

---

## 🔄 3. DATA MERGING STRATEGY

### Step 1: Create Master Date Index
```python
# Create complete monthly date range from 2010-01-01 to 2025-08-01
master_dates = pd.date_range(start='2010-01-01', end='2025-08-01', freq='MS')
# This ensures all 188 months are represented
```

### Step 2: Extract and Standardize Export Data
```python
# For each commodity (Rice, Cotton, Copper):
# 1. Convert refPeriodId to datetime
# 2. Extract: Date, Export_Value_USD (fobvalue), Weight_kg (netWgt)
# 3. Rename columns consistently
# 4. Fill missing months with 0 (for Copper only)
```

### Step 3: Standardize External Data
```python
# Convert all date formats to datetime
# Ensure all are monthly (first day of month)
# Create standard column names
```

### Step 4: Merge Strategy
```python
# Option A: Wide Format (One row per date, multiple columns)
# Columns: Date, Rice_Export, Cotton_Export, Copper_Export, 
#          USD_PKR, Oil_Price, US_Confidence

# Option B: Long Format (One row per commodity-date combination)
# Columns: Date, Commodity, Export_Value, Weight, USD_PKR, Oil_Price, US_Confidence
```

**Recommended**: **Option B (Long Format)** for multi-commodity modeling

---

## 🔧 4. MISSING DATA IMPUTATION RULES

### Export Data (Rice, Cotton, Copper)
- **Missing months**: Fill with `Export_Value = 0` and `Weight = 0`
  - Reason: Represents actual "no trade" scenario
  - Applies to: Copper (6 missing months)

### Rice netWgt (5 missing values)
- **Strategy**: Forward fill or interpolate
  - Reason: Weight data is supplementary, not critical
  - Alternative: Use export value to weight ratio from surrounding months

### Oil Price (27 missing months)
- **Primary Strategy**: **Linear Interpolation**
  ```python
  oil['Brent_Oil_Price'] = oil['Brent_Oil_Price'].interpolate(method='linear')
  ```
  - Reason: Oil prices are continuous; interpolation preserves trend
  
- **Alternative Strategies** (if interpolation fails):
  1. Forward fill (carry last known value)
  2. Backward fill (use next known value)
  3. Monthly median (robust to outliers)

### USD/PKR and US Confidence
- **No missing values** - no imputation needed

---

## 📋 5. FINAL MERGED DATASET STRUCTURE

### Recommended Structure (Long Format)

| Date       | Commodity | HS_Code | Export_Value_USD | Weight_kg | USD_PKR | Oil_Price | US_Confidence |
|------------|-----------|---------|------------------|-----------|---------|-----------|---------------|
| 2010-01-01 | Rice      | 1006    | 235850504.298    | 449937000 | 84.304  | 71.46     | 74.4         |
| 2010-01-01 | Cotton    | 520512  | 65946357.248     | 36905725  | 84.304  | 71.46     | 74.4         |
| 2010-01-01 | Copper    | 7403    | 442138.359       | 186278    | 84.304  | 71.46     | 74.4         |
| 2010-02-01 | Rice      | 1006    | 206046142.982    | 0         | 83.325  | 77.59     | 73.6         |
| ...        | ...       | ...     | ...              | ...       | ...     | ...       | ...          |

**Total Rows**: 188 months × 3 commodities = **564 rows**

### Alternative Structure (Wide Format)

| Date       | Rice_Export | Cotton_Export | Copper_Export | Rice_Weight | Cotton_Weight | Copper_Weight | USD_PKR | Oil_Price | US_Confidence |
|------------|-------------|---------------|---------------|-------------|---------------|---------------|---------|-----------|---------------|
| 2010-01-01 | 235850504   | 65946357      | 442138        | 449937000   | 36905725      | 186278        | 84.304  | 71.46     | 74.4          |
| 2010-02-01 | 206046143   | 42859445      | 684199        | 0           | 18416485      | 155600        | 83.325  | 77.59     | 73.6          |
| ...        | ...         | ...           | ...           | ...         | ...           | ...           | ...     | ...       | ...           |

**Total Rows**: **188 rows**

---

## ✅ 6. DATA QUALITY CHECKS

After merging, verify:
1. ✅ All 188 months are present
2. ✅ All 3 commodities are present for each month (or 0 if no trade)
3. ✅ External drivers are filled (oil price interpolated)
4. ✅ No duplicate dates
5. ✅ Date column is properly sorted
6. ✅ All numeric columns have appropriate data types
7. ✅ Export values ≥ 0
8. ✅ External drivers are reasonable (check for outliers)

---

## 🎯 7. RECOMMENDATIONS FOR MODELING

Given the data structure:
1. **Use Long Format** for multi-task learning (allows commodity embeddings)
2. **Handle missing oil prices** with interpolation (don't use 0)
3. **Keep missing export months as 0** (represents actual no-trade scenario)
4. **Consider feature engineering**:
   - Lag features (1, 3, 6, 12 months)
   - Rolling statistics (mean, std)
   - Seasonal features (month, quarter)
   - Commodity-specific interactions

---

## 📝 Next Steps

1. ✅ Create data preprocessing script
2. ✅ Implement merging logic
3. ✅ Apply missing data imputation
4. ✅ Validate merged dataset
5. ✅ Export clean dataset for modeling
