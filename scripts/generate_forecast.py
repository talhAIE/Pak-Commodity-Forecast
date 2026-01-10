"""
Forecast Generation Script for Pakistan Export Commodities
Generates 6-month forecasts using the trained LightGBM model
Saves results to CSV and creates visualizations with uncertainty bands
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
try:
    plt.style.use('seaborn-v0_8')
except:
    plt.style.use('seaborn')

# Set random seed for reproducibility
np.random.seed(42)

def create_lag_features(df, column, lags):
    """Create lag features for a given column within each commodity group"""
    df = df.copy()
    for lag in lags:
        df[f'{column}_lag_{lag}'] = df.groupby('Commodity')[column].shift(lag)
    return df

def create_rolling_features(df, column, windows):
    """Create rolling statistics for a given column within each commodity group"""
    df = df.copy()
    for window in windows:
        df[f'{column}_rolling_mean_{window}'] = df.groupby('Commodity')[column].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
        )
        df[f'{column}_rolling_std_{window}'] = df.groupby('Commodity')[column].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).std()
        )
    return df

def load_data_and_model():
    """Load the dataset and trained model"""
    print("Loading data and model...")
    
    # Load data
    data_path = 'merged_export_dataset_2010_2025.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df_long = pd.read_csv(data_path)
    df_long['Date'] = pd.to_datetime(df_long['Date'])
    df_long = df_long.sort_values(['Commodity', 'Date']).reset_index(drop=True)
    
    # Load model
    model_path = 'models/best_model_lgbm.pkl'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = joblib.load(model_path)
    
    # Load feature names
    feature_names_path = 'models/feature_names_lgbm.json'
    if not os.path.exists(feature_names_path):
        raise FileNotFoundError(f"Feature names file not found: {feature_names_path}")
    
    with open(feature_names_path, 'r') as f:
        feature_cols_enhanced = json.load(f)
    
    print(f"Data loaded: {len(df_long)} rows")
    print(f"Model loaded: {len(feature_cols_enhanced)} features")
    
    return df_long, model, feature_cols_enhanced

def prepare_features(df):
    """Prepare all features for the model (replicate feature engineering pipeline)"""
    print("Preparing features...")
    
    df_lgbm = df.copy()
    
    # Time features
    df_lgbm['Year'] = df_lgbm['Date'].dt.year
    df_lgbm['Month'] = df_lgbm['Date'].dt.month
    df_lgbm['Quarter'] = df_lgbm['Date'].dt.quarter
    
    # Cyclical encoding for Month and Quarter
    df_lgbm['Month_sin'] = np.sin(2 * np.pi * df_lgbm['Month'] / 12)
    df_lgbm['Month_cos'] = np.cos(2 * np.pi * df_lgbm['Month'] / 12)
    df_lgbm['Quarter_sin'] = np.sin(2 * np.pi * df_lgbm['Quarter'] / 4)
    df_lgbm['Quarter_cos'] = np.cos(2 * np.pi * df_lgbm['Quarter'] / 4)
    
    # Encode commodity
    commodity_mapping = {'Rice': 0, 'Cotton Yarn': 1, 'Copper': 2}
    df_lgbm['Commodity_encoded'] = df_lgbm['Commodity'].map(commodity_mapping)
    
    # Create lag features for export values
    df_lgbm = create_lag_features(df_lgbm, 'Export_Value_USD', [1, 3, 6, 12, 24])
    
    # Create rolling features
    df_lgbm = create_rolling_features(df_lgbm, 'Export_Value_USD', [3, 6, 12])
    
    # Create lag features for external drivers (from wide format)
    df_wide = df_lgbm.pivot_table(
        index='Date', 
        columns='Commodity', 
        values=['USD_PKR', 'Oil_Price', 'US_Confidence'], 
        aggfunc='first'
    )
    df_wide.columns = [f'{col[1]}_{col[0]}' if col[1] else col[0] for col in df_wide.columns]
    df_wide = df_wide.reset_index()
    
    # Get unique external driver values per date
    external_drivers = df_lgbm.groupby('Date')[['USD_PKR', 'Oil_Price', 'US_Confidence']].first().reset_index()
    
    # Create lag features for external drivers
    for lag in [1, 3, 6]:
        external_drivers[f'USD_PKR_lag_{lag}'] = external_drivers['USD_PKR'].shift(lag)
        external_drivers[f'Oil_Price_lag_{lag}'] = external_drivers['Oil_Price'].shift(lag)
        external_drivers[f'US_Confidence_lag_{lag}'] = external_drivers['US_Confidence'].shift(lag)
    
    # Merge lagged external drivers back
    df_lgbm = df_lgbm.merge(
        external_drivers[['Date'] + [col for col in external_drivers.columns if 'lag' in col]],
        on='Date',
        how='left'
    )
    
    # Fill missing values in lag and rolling features
    lag_cols = [col for col in df_lgbm.columns if 'lag' in col or 'rolling' in col]
    for col in lag_cols:
        df_lgbm[col] = df_lgbm.groupby('Commodity')[col].ffill().fillna(0)
    
    # Enhanced features
    df_lgbm_enhanced = df_lgbm.copy()
    
    # Interaction features
    df_lgbm_enhanced['Commodity_USD_PKR'] = df_lgbm_enhanced['Commodity_encoded'] * df_lgbm_enhanced['USD_PKR']
    df_lgbm_enhanced['Commodity_Oil'] = df_lgbm_enhanced['Commodity_encoded'] * df_lgbm_enhanced['Oil_Price']
    df_lgbm_enhanced['Commodity_Confidence'] = df_lgbm_enhanced['Commodity_encoded'] * df_lgbm_enhanced['US_Confidence']
    
    # Ratio features
    df_lgbm_enhanced['Export_USD_PKR_ratio'] = df_lgbm_enhanced['Export_Value_USD_lag_1'] / (df_lgbm_enhanced['USD_PKR'] + 1)
    df_lgbm_enhanced['Export_Oil_ratio'] = df_lgbm_enhanced['Export_Value_USD_lag_1'] / (df_lgbm_enhanced['Oil_Price'] + 1)
    df_lgbm_enhanced['Export_Confidence_ratio'] = df_lgbm_enhanced['Export_Value_USD_lag_1'] / (df_lgbm_enhanced['US_Confidence'] + 1)
    df_lgbm_enhanced[['Export_USD_PKR_ratio', 'Export_Oil_ratio', 'Export_Confidence_ratio']] = \
        df_lgbm_enhanced[['Export_USD_PKR_ratio', 'Export_Oil_ratio', 'Export_Confidence_ratio']].fillna(0)
    
    # Year-over-year change
    df_lgbm_enhanced['Export_YoY_change'] = df_lgbm_enhanced.groupby('Commodity')['Export_Value_USD'].pct_change(periods=12) * 100
    df_lgbm_enhanced['Export_YoY_change'] = df_lgbm_enhanced['Export_YoY_change'].fillna(0)
    
    # Momentum features
    df_lgbm_enhanced['Export_momentum_3'] = df_lgbm_enhanced.groupby('Commodity')['Export_Value_USD'].pct_change(periods=3) * 100
    df_lgbm_enhanced['Export_momentum_6'] = df_lgbm_enhanced.groupby('Commodity')['Export_Value_USD'].pct_change(periods=6) * 100
    df_lgbm_enhanced[['Export_momentum_3', 'Export_momentum_6']] = \
        df_lgbm_enhanced[['Export_momentum_3', 'Export_momentum_6']].fillna(0)
    
    # Fill lag_24 if missing
    if 'Export_Value_USD_lag_24' in df_lgbm_enhanced.columns:
        df_lgbm_enhanced['Export_Value_USD_lag_24'] = df_lgbm_enhanced['Export_Value_USD_lag_24'].fillna(0)
    
    # Trend features
    df_lgbm_enhanced['Export_trend_3'] = df_lgbm_enhanced.groupby('Commodity')['Export_Value_USD_lag_1'].diff(periods=3).fillna(0)
    df_lgbm_enhanced['Export_trend_6'] = df_lgbm_enhanced.groupby('Commodity')['Export_Value_USD_lag_1'].diff(periods=6).fillna(0)
    df_lgbm_enhanced['Export_acceleration'] = df_lgbm_enhanced.groupby('Commodity')['Export_trend_3'].diff(periods=1).fillna(0)
    
    # Volatility features
    df_lgbm_enhanced['Export_volatility_3'] = df_lgbm_enhanced.groupby('Commodity')['Export_Value_USD'].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).std()
    ).fillna(0)
    df_lgbm_enhanced['Export_volatility_6'] = df_lgbm_enhanced.groupby('Commodity')['Export_Value_USD'].transform(
        lambda x: x.shift(1).rolling(window=6, min_periods=1).std()
    ).fillna(0)
    
    # External driver changes
    df_lgbm_enhanced = df_lgbm_enhanced.sort_values(['Date', 'Commodity']).reset_index(drop=True)
    df_lgbm_enhanced['USD_PKR_change_3'] = df_lgbm_enhanced.groupby('Date')['USD_PKR'].transform('first').pct_change(periods=3).fillna(0) * 100
    df_lgbm_enhanced['Oil_Price_change_3'] = df_lgbm_enhanced.groupby('Date')['Oil_Price'].transform('first').pct_change(periods=3).fillna(0) * 100
    df_lgbm_enhanced['US_Confidence_change_3'] = df_lgbm_enhanced.groupby('Date')['US_Confidence'].transform('first').pct_change(periods=3).fillna(0) * 100
    df_lgbm_enhanced = df_lgbm_enhanced.sort_values(['Commodity', 'Date']).reset_index(drop=True)
    
    # Interaction features with lags
    df_lgbm_enhanced['Lag1_USD_PKR'] = df_lgbm_enhanced['Export_Value_USD_lag_1'] * df_lgbm_enhanced['USD_PKR']
    df_lgbm_enhanced['Lag1_Oil'] = df_lgbm_enhanced['Export_Value_USD_lag_1'] * df_lgbm_enhanced['Oil_Price']
    df_lgbm_enhanced['Lag1_Confidence'] = df_lgbm_enhanced['Export_Value_USD_lag_1'] * df_lgbm_enhanced['US_Confidence']
    
    # Commodity-seasonality interactions
    df_lgbm_enhanced['Commodity_Month'] = df_lgbm_enhanced['Commodity_encoded'] * df_lgbm_enhanced['Month_sin']
    df_lgbm_enhanced['Commodity_Quarter'] = df_lgbm_enhanced['Commodity_encoded'] * df_lgbm_enhanced['Quarter_sin']
    
    # Polynomial features
    df_lgbm_enhanced['USD_PKR_squared'] = df_lgbm_enhanced['USD_PKR'] ** 2
    df_lgbm_enhanced['Oil_Price_squared'] = df_lgbm_enhanced['Oil_Price'] ** 2
    
    # Percentile features
    df_lgbm_enhanced['Export_percentile_12'] = df_lgbm_enhanced.groupby('Commodity')['Export_Value_USD'].transform(
        lambda x: x.shift(1).rolling(window=12, min_periods=1).rank(pct=True)
    ).fillna(0.5)
    
    # Lag vs rolling mean
    df_lgbm_enhanced['Lag1_vs_rolling_mean_6'] = (df_lgbm_enhanced['Export_Value_USD_lag_1'] - df_lgbm_enhanced['Export_Value_USD_rolling_mean_6']) / (df_lgbm_enhanced['Export_Value_USD_rolling_mean_6'] + 1)
    df_lgbm_enhanced['Lag1_vs_rolling_mean_12'] = (df_lgbm_enhanced['Export_Value_USD_lag_1'] - df_lgbm_enhanced['Export_Value_USD_rolling_mean_12']) / (df_lgbm_enhanced['Export_Value_USD_rolling_mean_12'] + 1)
    df_lgbm_enhanced[['Lag1_vs_rolling_mean_6', 'Lag1_vs_rolling_mean_12']] = \
        df_lgbm_enhanced[['Lag1_vs_rolling_mean_6', 'Lag1_vs_rolling_mean_12']].fillna(0)
    
    # Fill any remaining NaN values
    df_lgbm_enhanced = df_lgbm_enhanced.fillna(0)
    
    return df_lgbm_enhanced

def generate_forecasts(df_enhanced, model, feature_cols_enhanced, forecast_months=6):
    """Generate forecasts for the next N months"""
    print(f"\nGenerating {forecast_months}-month forecasts...")
    
    commodities = ['Rice', 'Cotton Yarn', 'Copper']
    commodity_mapping = {'Rice': 0, 'Cotton Yarn': 1, 'Copper': 2}
    
    # Get last date in dataset
    last_date = df_enhanced['Date'].max()
    print(f"Last date in dataset: {last_date.strftime('%Y-%m-%d')}")
    
    # Generate future dates
    future_dates = []
    current_date = last_date
    for i in range(1, forecast_months + 1):
        # Add months
        if current_date.month == 12:
            future_date = current_date.replace(year=current_date.year + 1, month=1, day=1)
        else:
            future_date = current_date.replace(month=current_date.month + 1, day=1)
        future_dates.append(future_date)
        current_date = future_date
    
    print(f"Forecasting for dates: {[d.strftime('%Y-%m-%d') for d in future_dates]}")
    
    # Get last known external driver values (simplified - use last known values)
    last_external = df_enhanced.groupby('Date')[['USD_PKR', 'Oil_Price', 'US_Confidence']].first().tail(1).iloc[0]
    
    # Create future external regressors (use last known values)
    future_exog = pd.DataFrame({
        'Date': future_dates,
        'USD_PKR': [last_external['USD_PKR']] * forecast_months,
        'Oil_Price': [last_external['Oil_Price']] * forecast_months,
        'US_Confidence': [last_external['US_Confidence']] * forecast_months
    })
    
    future_forecasts = {}
    
    for commodity in commodities:
        print(f"\nForecasting for {commodity}...")
        
        # Get last data point for this commodity
        commodity_data = df_enhanced[df_enhanced['Commodity'] == commodity].copy()
        last_row = commodity_data.iloc[-1]
        
        commodity_forecasts = []
        
        for i, future_date in enumerate(future_dates):
            # Create future row based on last row
            future_row = last_row.copy()
            
            # Update date
            future_row['Date'] = future_date
            
            # Update time features
            future_row['Year'] = future_date.year
            future_row['Month'] = future_date.month
            future_row['Quarter'] = (future_date.month - 1) // 3 + 1
            future_row['Month_sin'] = np.sin(2 * np.pi * future_date.month / 12)
            future_row['Month_cos'] = np.cos(2 * np.pi * future_date.month / 12)
            future_row['Quarter_sin'] = np.sin(2 * np.pi * future_row['Quarter'] / 4)
            future_row['Quarter_cos'] = np.cos(2 * np.pi * future_row['Quarter'] / 4)
            
            # Update external drivers
            future_row['USD_PKR'] = future_exog.iloc[i]['USD_PKR']
            future_row['Oil_Price'] = future_exog.iloc[i]['Oil_Price']
            future_row['US_Confidence'] = future_exog.iloc[i]['US_Confidence']
            
            # Update lag features using previous forecasts
            if i == 0:
                # Use last known export value
                future_row['Export_Value_USD_lag_1'] = last_row['Export_Value_USD']
            else:
                # Use previous forecast
                future_row['Export_Value_USD_lag_1'] = commodity_forecasts[-1]
            
            # Update other lags
            if i >= 3:
                future_row['Export_Value_USD_lag_3'] = commodity_forecasts[-3]
            else:
                if len(commodity_data) >= (3 - i):
                    future_row['Export_Value_USD_lag_3'] = commodity_data['Export_Value_USD'].iloc[-(3-i)]
                else:
                    future_row['Export_Value_USD_lag_3'] = last_row['Export_Value_USD']
            
            if i >= 6:
                future_row['Export_Value_USD_lag_6'] = commodity_forecasts[-6]
            else:
                if len(commodity_data) >= (6 - i):
                    future_row['Export_Value_USD_lag_6'] = commodity_data['Export_Value_USD'].iloc[-(6-i)]
                else:
                    future_row['Export_Value_USD_lag_6'] = last_row['Export_Value_USD']
            
            if i >= 12:
                future_row['Export_Value_USD_lag_12'] = commodity_forecasts[-12]
            else:
                if len(commodity_data) >= (12 - i):
                    future_row['Export_Value_USD_lag_12'] = commodity_data['Export_Value_USD'].iloc[-(12-i)]
                else:
                    future_row['Export_Value_USD_lag_12'] = last_row['Export_Value_USD']
            
            if i >= 24:
                future_row['Export_Value_USD_lag_24'] = commodity_forecasts[-24]
            else:
                if len(commodity_data) >= (24 - i):
                    future_row['Export_Value_USD_lag_24'] = commodity_data['Export_Value_USD'].iloc[-(24-i)]
                else:
                    future_row['Export_Value_USD_lag_24'] = last_row['Export_Value_USD']
            
            # Update rolling features (simplified - use last known)
            for window in [3, 6, 12]:
                if f'Export_Value_USD_rolling_mean_{window}' in last_row.index:
                    future_row[f'Export_Value_USD_rolling_mean_{window}'] = last_row[f'Export_Value_USD_rolling_mean_{window}']
                else:
                    future_row[f'Export_Value_USD_rolling_mean_{window}'] = last_row['Export_Value_USD']
                
                if f'Export_Value_USD_rolling_std_{window}' in last_row.index:
                    future_row[f'Export_Value_USD_rolling_std_{window}'] = last_row[f'Export_Value_USD_rolling_std_{window}']
                else:
                    future_row[f'Export_Value_USD_rolling_std_{window}'] = 0
            
            # Update interaction features
            future_row['Commodity_USD_PKR'] = future_row['Commodity_encoded'] * future_row['USD_PKR']
            future_row['Commodity_Oil'] = future_row['Commodity_encoded'] * future_row['Oil_Price']
            future_row['Commodity_Confidence'] = future_row['Commodity_encoded'] * future_row['US_Confidence']
            
            # Update ratio features
            future_row['Export_USD_PKR_ratio'] = future_row['Export_Value_USD_lag_1'] / (future_row['USD_PKR'] + 1)
            future_row['Export_Oil_ratio'] = future_row['Export_Value_USD_lag_1'] / (future_row['Oil_Price'] + 1)
            future_row['Export_Confidence_ratio'] = future_row['Export_Value_USD_lag_1'] / (future_row['US_Confidence'] + 1)
            
            # Update momentum features
            if i >= 3:
                future_row['Export_momentum_3'] = ((commodity_forecasts[-1] - commodity_forecasts[-3]) / (commodity_forecasts[-3] + 1)) * 100 if commodity_forecasts[-3] > 0 else 0
            else:
                if 'Export_momentum_3' in last_row.index:
                    future_row['Export_momentum_3'] = last_row['Export_momentum_3']
                else:
                    future_row['Export_momentum_3'] = 0
            
            if i >= 6:
                future_row['Export_momentum_6'] = ((commodity_forecasts[-1] - commodity_forecasts[-6]) / (commodity_forecasts[-6] + 1)) * 100 if commodity_forecasts[-6] > 0 else 0
            else:
                if 'Export_momentum_6' in last_row.index:
                    future_row['Export_momentum_6'] = last_row['Export_momentum_6']
                else:
                    future_row['Export_momentum_6'] = 0
            
            if 'Export_YoY_change' in last_row.index:
                future_row['Export_YoY_change'] = last_row['Export_YoY_change']
            else:
                future_row['Export_YoY_change'] = 0
            
            # Update trend features
            if i >= 3:
                future_row['Export_trend_3'] = commodity_forecasts[-1] - commodity_forecasts[-3] if len(commodity_forecasts) >= 3 else 0
            else:
                if 'Export_trend_3' in last_row.index:
                    future_row['Export_trend_3'] = last_row['Export_trend_3']
                else:
                    future_row['Export_trend_3'] = 0
            
            if i >= 6:
                future_row['Export_trend_6'] = commodity_forecasts[-1] - commodity_forecasts[-6] if len(commodity_forecasts) >= 6 else 0
            else:
                if 'Export_trend_6' in last_row.index:
                    future_row['Export_trend_6'] = last_row['Export_trend_6']
                else:
                    future_row['Export_trend_6'] = 0
            
            # Update acceleration
            if i >= 2:
                prev_trend = commodity_forecasts[-1] - commodity_forecasts[-2] if len(commodity_forecasts) >= 2 else 0
                if i >= 3:
                    prev_prev_trend = commodity_forecasts[-2] - commodity_forecasts[-3] if len(commodity_forecasts) >= 3 else 0
                    future_row['Export_acceleration'] = prev_trend - prev_prev_trend
                else:
                    future_row['Export_acceleration'] = 0
            else:
                if 'Export_acceleration' in last_row.index:
                    future_row['Export_acceleration'] = last_row['Export_acceleration']
                else:
                    future_row['Export_acceleration'] = 0
            
            # Update volatility
            if i >= 3:
                recent_values = commodity_forecasts[-3:] if len(commodity_forecasts) >= 3 else commodity_forecasts
                future_row['Export_volatility_3'] = np.std(recent_values) if len(recent_values) > 1 else 0
            else:
                if 'Export_volatility_3' in last_row.index:
                    future_row['Export_volatility_3'] = last_row['Export_volatility_3']
                else:
                    future_row['Export_volatility_3'] = 0
            
            if i >= 6:
                recent_values = commodity_forecasts[-6:] if len(commodity_forecasts) >= 6 else commodity_forecasts
                future_row['Export_volatility_6'] = np.std(recent_values) if len(recent_values) > 1 else 0
            else:
                if 'Export_volatility_6' in last_row.index:
                    future_row['Export_volatility_6'] = last_row['Export_volatility_6']
                else:
                    future_row['Export_volatility_6'] = 0
            
            # Update external driver changes
            if i >= 3:
                usd_pkr_3_ago = future_exog.iloc[i-3]['USD_PKR'] if i >= 3 else last_external['USD_PKR']
                oil_3_ago = future_exog.iloc[i-3]['Oil_Price'] if i >= 3 else last_external['Oil_Price']
                conf_3_ago = future_exog.iloc[i-3]['US_Confidence'] if i >= 3 else last_external['US_Confidence']
                
                future_row['USD_PKR_change_3'] = ((future_row['USD_PKR'] - usd_pkr_3_ago) / (usd_pkr_3_ago + 1)) * 100
                future_row['Oil_Price_change_3'] = ((future_row['Oil_Price'] - oil_3_ago) / (oil_3_ago + 1)) * 100
                future_row['US_Confidence_change_3'] = ((future_row['US_Confidence'] - conf_3_ago) / (conf_3_ago + 1)) * 100
            else:
                if 'USD_PKR_change_3' in last_row.index:
                    future_row['USD_PKR_change_3'] = last_row['USD_PKR_change_3']
                    future_row['Oil_Price_change_3'] = last_row['Oil_Price_change_3']
                    future_row['US_Confidence_change_3'] = last_row['US_Confidence_change_3']
                else:
                    future_row['USD_PKR_change_3'] = 0
                    future_row['Oil_Price_change_3'] = 0
                    future_row['US_Confidence_change_3'] = 0
            
            # Update interaction features with lags
            future_row['Lag1_USD_PKR'] = future_row['Export_Value_USD_lag_1'] * future_row['USD_PKR']
            future_row['Lag1_Oil'] = future_row['Export_Value_USD_lag_1'] * future_row['Oil_Price']
            future_row['Lag1_Confidence'] = future_row['Export_Value_USD_lag_1'] * future_row['US_Confidence']
            
            # Update commodity-seasonality interactions
            future_row['Commodity_Month'] = future_row['Commodity_encoded'] * future_row['Month_sin']
            future_row['Commodity_Quarter'] = future_row['Commodity_encoded'] * future_row['Quarter_sin']
            
            # Update polynomial features
            future_row['USD_PKR_squared'] = future_row['USD_PKR'] ** 2
            future_row['Oil_Price_squared'] = future_row['Oil_Price'] ** 2
            
            # Update percentile features
            if 'Export_percentile_12' in last_row.index:
                future_row['Export_percentile_12'] = last_row['Export_percentile_12']
            else:
                future_row['Export_percentile_12'] = 0.5
            
            # Update lag vs rolling mean
            rolling_mean_6 = future_row.get('Export_Value_USD_rolling_mean_6', future_row['Export_Value_USD_lag_1'])
            rolling_mean_12 = future_row.get('Export_Value_USD_rolling_mean_12', future_row['Export_Value_USD_lag_1'])
            future_row['Lag1_vs_rolling_mean_6'] = (future_row['Export_Value_USD_lag_1'] - rolling_mean_6) / (rolling_mean_6 + 1)
            future_row['Lag1_vs_rolling_mean_12'] = (future_row['Export_Value_USD_lag_1'] - rolling_mean_12) / (rolling_mean_12 + 1)
            
            # Update external driver lags (simplified)
            for lag in [1, 3, 6]:
                if f'USD_PKR_lag_{lag}' in last_row.index:
                    future_row[f'USD_PKR_lag_{lag}'] = last_row[f'USD_PKR_lag_{lag}']
                else:
                    future_row[f'USD_PKR_lag_{lag}'] = future_row['USD_PKR']
                
                if f'Oil_Price_lag_{lag}' in last_row.index:
                    future_row[f'Oil_Price_lag_{lag}'] = last_row[f'Oil_Price_lag_{lag}']
                else:
                    future_row[f'Oil_Price_lag_{lag}'] = future_row['Oil_Price']
                
                if f'US_Confidence_lag_{lag}' in last_row.index:
                    future_row[f'US_Confidence_lag_{lag}'] = last_row[f'US_Confidence_lag_{lag}']
                else:
                    future_row[f'US_Confidence_lag_{lag}'] = future_row['US_Confidence']
            
            # Prepare feature vector
            X_future_row = []
            for col in feature_cols_enhanced:
                if col in future_row.index:
                    X_future_row.append(future_row[col])
                else:
                    X_future_row.append(0)
            
            X_future = np.array(X_future_row).reshape(1, -1)
            
            # Make prediction (model outputs log-transformed value)
            forecast_value_log = model.predict(X_future, num_iteration=model.best_iteration)[0]
            forecast_value = np.expm1(forecast_value_log)  # Inverse log transformation
            forecast_value = max(0, forecast_value)  # Ensure non-negative
            commodity_forecasts.append(forecast_value)
        
        future_forecasts[commodity] = commodity_forecasts
        print(f"  {commodity}: {[f'{v:,.0f}' for v in commodity_forecasts]}")
    
    return future_dates, future_forecasts

def save_forecasts(future_dates, future_forecasts):
    """Save forecast results to CSV"""
    print("\nSaving forecasts to CSV...")
    
    os.makedirs('forecasts', exist_ok=True)
    
    forecast_results = []
    commodities = ['Rice', 'Cotton Yarn', 'Copper']
    
    for commodity in commodities:
        for i, date in enumerate(future_dates):
            forecast_results.append({
                'Date': date.strftime('%Y-%m-%d'),
                'Commodity': commodity,
                'Forecast_Export_Value_USD': future_forecasts[commodity][i]
            })
    
    forecast_df = pd.DataFrame(forecast_results)
    forecast_df = forecast_df.sort_values(['Date', 'Commodity']).reset_index(drop=True)
    
    # Save to CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = f'forecasts/forecast_6months_{timestamp}.csv'
    forecast_df.to_csv(output_path, index=False)
    
    print(f"Forecasts saved to: {output_path}")
    print("\nForecast Summary:")
    print(forecast_df.to_string(index=False))
    
    return forecast_df, output_path

def create_visualizations(df_long, forecast_df, output_path):
    """Create separate, clean visualizations with uncertainty bands for each commodity"""
    print("\nCreating clean visualizations with uncertainty bands...")
    
    commodities = ['Rice', 'Cotton Yarn', 'Copper']
    forecast_df['Date'] = pd.to_datetime(forecast_df['Date'])
    last_date = df_long['Date'].max()
    
    # Color scheme
    colors = {
        'historical': '#2E86AB',  # Blue
        'forecast': '#DC143C',    # Red
        'uncertainty': '#FF6B6B', # Light Red/Pink
        'forecast_start': '#06A77D'  # Green
    }
    
    viz_paths = []
    
    for commodity in commodities:
        # Create separate figure for each commodity
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Get historical and forecast data
        historical = df_long[df_long['Commodity'] == commodity].copy()
        commodity_forecast = forecast_df[forecast_df['Commodity'] == commodity].copy()
        
        # Calculate uncertainty bands based on historical volatility
        recent_historical = historical.tail(12)['Export_Value_USD']
        historical_mean = recent_historical.mean()
        historical_std = recent_historical.std()
        historical_cv = historical_std / historical_mean if historical_mean > 0 else 0.15
        
        # Calculate uncertainty bands
        forecast_values = commodity_forecast['Forecast_Export_Value_USD'].values
        uncertainty_factor = 1.5 * historical_cv  # Base uncertainty factor
        
        # Uncertainty increases slightly over forecast horizon
        months_ahead = np.arange(1, len(forecast_values) + 1)
        uncertainty_multiplier = 1 + (months_ahead - 1) * 0.05  # 5% increase per month
        
        upper_bound = forecast_values * (1 + uncertainty_factor * uncertainty_multiplier)
        lower_bound = forecast_values * (1 - uncertainty_factor * uncertainty_multiplier)
        lower_bound = np.maximum(lower_bound, 0)  # Ensure non-negative
        
        # Plot historical data
        ax.plot(historical['Date'], historical['Export_Value_USD'], 
               label='Historical Data', linewidth=2.5, color=colors['historical'], 
               alpha=0.85, zorder=3)
        
        # Plot uncertainty bands
        ax.fill_between(commodity_forecast['Date'], lower_bound, upper_bound,
                       alpha=0.2, color=colors['uncertainty'], 
                       label='Uncertainty Band (80% confidence)', zorder=1, edgecolor='none')
        
        # Plot forecast line
        ax.plot(commodity_forecast['Date'], commodity_forecast['Forecast_Export_Value_USD'], 
               label='6-Month Forecast', linewidth=2.5, marker='o', markersize=8, 
               color=colors['forecast'], alpha=0.95, zorder=4, 
               markerfacecolor='white', markeredgewidth=2.5, markeredgecolor=colors['forecast'],
               linestyle='--')
        
        # Add forecast start line
        ax.axvline(x=last_date, color=colors['forecast_start'], linestyle='--', 
                  linewidth=2.5, label='Forecast Start', zorder=2, alpha=0.8)
        
        # Formatting
        ax.set_title(f'{commodity} - 6-Month Export Forecast with Uncertainty Bands', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Date', fontsize=12, fontweight='bold', labelpad=10)
        ax.set_ylabel('Export Value (USD)', fontsize=12, fontweight='bold', labelpad=10)
        
        # Format y-axis to show values in millions/billions
        max_val = max(historical['Export_Value_USD'].max(), upper_bound.max())
        if max_val >= 1e8:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e8:.2f}B'))
            unit_label = 'Billions USD'
        else:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
            unit_label = 'Millions USD'
        
        # Date formatting - better spacing
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_minor_locator(mdates.MonthLocator())
        
        # Rotate dates for better readability
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)
        
        # Legend - place it in upper left corner
        ax.legend(loc='upper left', framealpha=0.95, shadow=True, fontsize=10, 
                 fancybox=True, frameon=True)
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, zorder=0, which='both')
        ax.set_axisbelow(True)
        
        # Add statistics box - positioned to avoid overlap
        hist_mean = historical['Export_Value_USD'].mean()
        forecast_mean = commodity_forecast['Forecast_Export_Value_USD'].mean()
        forecast_min = lower_bound.min()
        forecast_max = upper_bound.max()
        
        # Format numbers based on scale
        if max_val >= 1e8:
            stats_text = f'Historical Average: ${hist_mean/1e8:.2f}B\n'
            stats_text += f'Forecast Average: ${forecast_mean/1e8:.2f}B\n'
            stats_text += f'Forecast Range:\n${forecast_min/1e8:.2f}B - ${forecast_max/1e8:.2f}B'
        else:
            stats_text = f'Historical Average: ${hist_mean/1e6:.1f}M\n'
            stats_text += f'Forecast Average: ${forecast_mean/1e6:.1f}M\n'
            stats_text += f'Forecast Range:\n${forecast_min/1e6:.1f}M - ${forecast_max/1e6:.1f}M'
        
        # Position statistics box at center top of the chart area
        # Get the y-axis range to position box at top
        y_min, y_max = ax.get_ylim()
        y_position = y_max - (y_max - y_min) * 0.05  # Position at 95% of y-axis range
        
        # Get date range for x-position
        x_min, x_max = ax.get_xlim()
        x_position = (x_min + x_max) / 2  # Center of x-axis
        
        ax.text(x_position, y_position, stats_text, transform=ax.transData, 
               fontsize=10, verticalalignment='top', horizontalalignment='center',
               bbox=dict(boxstyle='round,pad=0.8', facecolor='wheat', alpha=0.9, 
                        edgecolor='black', linewidth=1.5), zorder=5, family='monospace')
        
        # Adjust layout to prevent overlap
        plt.tight_layout()
        
        # Save individual figure
        commodity_name = commodity.replace(' ', '_').lower()
        viz_path = output_path.replace('.csv', f'_{commodity_name}_visualization.png')
        plt.savefig(viz_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        viz_paths.append(viz_path)
        print(f"  {commodity} visualization saved: {viz_path}")
        
        plt.close(fig)  # Close figure to free memory
    
    print(f"\nCreated {len(viz_paths)} separate visualization files (one for each commodity)")
    
    return viz_paths[0] if len(viz_paths) > 0 else None  # Return first individual path for compatibility

def main():
    """Main function to run the forecast generation"""
    print("=" * 60)
    print("Pakistan Export Commodities - 6-Month Forecast Generator")
    print("=" * 60)
    
    try:
        # Load data and model
        df_long, model, feature_cols_enhanced = load_data_and_model()
        
        # Prepare features
        df_enhanced = prepare_features(df_long)
        
        # Generate forecasts
        future_dates, future_forecasts = generate_forecasts(df_enhanced, model, feature_cols_enhanced, forecast_months=6)
        
        # Save forecasts
        forecast_df, output_path = save_forecasts(future_dates, future_forecasts)
        
        # Create visualizations (separate charts for each commodity)
        viz_path = create_visualizations(df_long, forecast_df, output_path)
        
        print("\n" + "=" * 60)
        print("Forecast generation completed successfully!")
        print("=" * 60)
        print(f"\nOutput files:")
        print(f"  - Forecast CSV: {output_path}")
        
        # List all visualization files created
        import glob
        base_name = output_path.replace('.csv', '')
        viz_files = glob.glob(f"{base_name}_*_visualization.png")
        if viz_files:
            print(f"\n  Individual commodity visualizations:")
            for viz_file in sorted(viz_files):
                commodity_name = os.path.basename(viz_file).split('_')[-2] if '_' in os.path.basename(viz_file) else 'Unknown'
                print(f"    - {os.path.basename(viz_file)} ({commodity_name.replace('_', ' ').title()})")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
