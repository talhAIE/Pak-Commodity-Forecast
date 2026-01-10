"""
Forecast Generation Utilities for Streamlit Dashboard
Adapted from scripts/generate_forecast.py for dashboard use
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
from pathlib import Path
import importlib.util

# Add parent directory to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / 'scripts'

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

def prepare_features_for_dashboard(df):
    """Prepare all features for the model (adapted from generate_forecast.py)"""
    df_lgbm = df.copy()
    
    # Time features
    df_lgbm['Year'] = pd.to_datetime(df_lgbm['Date']).dt.year
    df_lgbm['Month'] = pd.to_datetime(df_lgbm['Date']).dt.month
    df_lgbm['Quarter'] = pd.to_datetime(df_lgbm['Date']).dt.quarter
    
    # Cyclical encoding
    df_lgbm['Month_sin'] = np.sin(2 * np.pi * df_lgbm['Month'] / 12)
    df_lgbm['Month_cos'] = np.cos(2 * np.pi * df_lgbm['Month'] / 12)
    df_lgbm['Quarter_sin'] = np.sin(2 * np.pi * df_lgbm['Quarter'] / 4)
    df_lgbm['Quarter_cos'] = np.cos(2 * np.pi * df_lgbm['Quarter'] / 4)
    
    # Encode commodity
    commodity_mapping = {'Rice': 0, 'Cotton Yarn': 1, 'Copper': 2}
    df_lgbm['Commodity_encoded'] = df_lgbm['Commodity'].map(commodity_mapping)
    
    # Create lag features
    df_lgbm = create_lag_features(df_lgbm, 'Export_Value_USD', [1, 3, 6, 12, 24])
    df_lgbm = create_rolling_features(df_lgbm, 'Export_Value_USD', [3, 6, 12])
    
    # External drivers lags
    external_drivers = df_lgbm.groupby('Date')[['USD_PKR', 'Oil_Price', 'US_Confidence']].first().reset_index()
    for lag in [1, 3, 6]:
        external_drivers[f'USD_PKR_lag_{lag}'] = external_drivers['USD_PKR'].shift(lag)
        external_drivers[f'Oil_Price_lag_{lag}'] = external_drivers['Oil_Price'].shift(lag)
        external_drivers[f'US_Confidence_lag_{lag}'] = external_drivers['US_Confidence'].shift(lag)
    
    df_lgbm = df_lgbm.merge(
        external_drivers[['Date'] + [col for col in external_drivers.columns if 'lag' in col]],
        on='Date', how='left'
    )
    
    # Fill missing values
    lag_cols = [col for col in df_lgbm.columns if 'lag' in col or 'rolling' in col]
    for col in lag_cols:
        df_lgbm[col] = df_lgbm.groupby('Commodity')[col].ffill().fillna(0)
    
    # Enhanced features (simplified - using key features only)
    df_enhanced = df_lgbm.copy()
    
    # Interaction features
    df_enhanced['Commodity_USD_PKR'] = df_enhanced['Commodity_encoded'] * df_enhanced['USD_PKR']
    df_enhanced['Commodity_Oil'] = df_enhanced['Commodity_encoded'] * df_enhanced['Oil_Price']
    df_enhanced['Commodity_Confidence'] = df_enhanced['Commodity_encoded'] * df_enhanced['US_Confidence']
    
    # Ratio features
    df_enhanced['Export_USD_PKR_ratio'] = df_enhanced['Export_Value_USD_lag_1'] / (df_enhanced['USD_PKR'] + 1)
    df_enhanced['Export_Oil_ratio'] = df_enhanced['Export_Value_USD_lag_1'] / (df_enhanced['Oil_Price'] + 1)
    df_enhanced['Export_Confidence_ratio'] = df_enhanced['Export_Value_USD_lag_1'] / (df_enhanced['US_Confidence'] + 1)
    df_enhanced[['Export_USD_PKR_ratio', 'Export_Oil_ratio', 'Export_Confidence_ratio']] = \
        df_enhanced[['Export_USD_PKR_ratio', 'Export_Oil_ratio', 'Export_Confidence_ratio']].fillna(0)
    
    # Momentum features
    df_enhanced['Export_YoY_change'] = df_enhanced.groupby('Commodity')['Export_Value_USD'].pct_change(periods=12) * 100
    df_enhanced['Export_momentum_3'] = df_enhanced.groupby('Commodity')['Export_Value_USD'].pct_change(periods=3) * 100
    df_enhanced['Export_momentum_6'] = df_enhanced.groupby('Commodity')['Export_Value_USD'].pct_change(periods=6) * 100
    df_enhanced[['Export_YoY_change', 'Export_momentum_3', 'Export_momentum_6']] = \
        df_enhanced[['Export_YoY_change', 'Export_momentum_3', 'Export_momentum_6']].fillna(0)
    
    # Additional features (simplified version)
    df_enhanced['Export_Value_USD_lag_24'] = df_enhanced.get('Export_Value_USD_lag_24', 0)
    df_enhanced['Export_trend_3'] = df_enhanced.groupby('Commodity')['Export_Value_USD_lag_1'].diff(periods=3).fillna(0)
    df_enhanced['Export_trend_6'] = df_enhanced.groupby('Commodity')['Export_Value_USD_lag_1'].diff(periods=6).fillna(0)
    df_enhanced['Export_acceleration'] = df_enhanced.groupby('Commodity')['Export_trend_3'].diff(periods=1).fillna(0)
    
    # Volatility
    df_enhanced['Export_volatility_3'] = df_enhanced.groupby('Commodity')['Export_Value_USD'].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).std()
    ).fillna(0)
    df_enhanced['Export_volatility_6'] = df_enhanced.groupby('Commodity')['Export_Value_USD'].transform(
        lambda x: x.shift(1).rolling(window=6, min_periods=1).std()
    ).fillna(0)
    
    # External driver changes
    df_enhanced = df_enhanced.sort_values(['Date', 'Commodity']).reset_index(drop=True)
    df_enhanced['USD_PKR_change_3'] = df_enhanced.groupby('Date')['USD_PKR'].transform('first').pct_change(periods=3).fillna(0) * 100
    df_enhanced['Oil_Price_change_3'] = df_enhanced.groupby('Date')['Oil_Price'].transform('first').pct_change(periods=3).fillna(0) * 100
    df_enhanced['US_Confidence_change_3'] = df_enhanced.groupby('Date')['US_Confidence'].transform('first').pct_change(periods=3).fillna(0) * 100
    df_enhanced = df_enhanced.sort_values(['Commodity', 'Date']).reset_index(drop=True)
    
    # Interaction features
    df_enhanced['Lag1_USD_PKR'] = df_enhanced['Export_Value_USD_lag_1'] * df_enhanced['USD_PKR']
    df_enhanced['Lag1_Oil'] = df_enhanced['Export_Value_USD_lag_1'] * df_enhanced['Oil_Price']
    df_enhanced['Lag1_Confidence'] = df_enhanced['Export_Value_USD_lag_1'] * df_enhanced['US_Confidence']
    df_enhanced['Commodity_Month'] = df_enhanced['Commodity_encoded'] * df_enhanced['Month_sin']
    df_enhanced['Commodity_Quarter'] = df_enhanced['Commodity_encoded'] * df_enhanced['Quarter_sin']
    df_enhanced['USD_PKR_squared'] = df_enhanced['USD_PKR'] ** 2
    df_enhanced['Oil_Price_squared'] = df_enhanced['Oil_Price'] ** 2
    df_enhanced['Export_percentile_12'] = df_enhanced.groupby('Commodity')['Export_Value_USD'].transform(
        lambda x: x.shift(1).rolling(window=12, min_periods=1).rank(pct=True)
    ).fillna(0.5)
    df_enhanced['Lag1_vs_rolling_mean_6'] = (df_enhanced['Export_Value_USD_lag_1'] - df_enhanced['Export_Value_USD_rolling_mean_6']) / (df_enhanced['Export_Value_USD_rolling_mean_6'] + 1)
    df_enhanced['Lag1_vs_rolling_mean_12'] = (df_enhanced['Export_Value_USD_lag_1'] - df_enhanced['Export_Value_USD_rolling_mean_12']) / (df_enhanced['Export_Value_USD_rolling_mean_12'] + 1)
    df_enhanced[['Lag1_vs_rolling_mean_6', 'Lag1_vs_rolling_mean_12']] = \
        df_enhanced[['Lag1_vs_rolling_mean_6', 'Lag1_vs_rolling_mean_12']].fillna(0)
    
    df_enhanced = df_enhanced.fillna(0)
    return df_enhanced

def generate_forecast_for_dashboard(df_long, model, feature_cols_enhanced, forecast_months=6, selected_commodities=None):
    """
    Generate forecasts for dashboard use - uses actual generate_forecast.py script functions
    """
    # Add scripts directory to path
    scripts_path = str(PROJECT_ROOT / 'scripts')
    if scripts_path not in sys.path:
        sys.path.insert(0, scripts_path)
    
    # Try to use the actual generate_forecast.py script
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("generate_forecast", PROJECT_ROOT / 'scripts' / 'generate_forecast.py')
        gf_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gf_module)
        
        # Use the prepare_features function from the script
        df_enhanced = gf_module.prepare_features(df_long)
        
        # Use the generate_forecasts function from the script
        future_dates, future_forecasts = gf_module.generate_forecasts(
            df_enhanced, 
            model, 
            feature_cols_enhanced, 
            forecast_months=forecast_months
        )
        
    except Exception as e:
        # Fallback: use simplified version
        import warnings
        warnings.warn(f"Using simplified forecast generation. Original error: {str(e)}")
        df_enhanced = prepare_features_for_dashboard(df_long)
        future_dates, future_forecasts = _generate_forecasts_simplified(
            df_enhanced, model, feature_cols_enhanced, forecast_months
        )
    
    # Filter by selected commodities if specified
    if selected_commodities and 'All' not in selected_commodities:
        future_forecasts = {k: v for k, v in future_forecasts.items() if k in selected_commodities}
    
    # Create forecast DataFrame
    forecast_results = []
    commodities = selected_commodities if selected_commodities and 'All' not in selected_commodities else ['Rice', 'Cotton Yarn', 'Copper']
    
    for commodity in commodities:
        if commodity in future_forecasts:
            for i, date in enumerate(future_dates):
                forecast_results.append({
                    'Date': date,
                    'Commodity': commodity,
                    'Forecast_Export_Value_USD': future_forecasts[commodity][i]
                })
    
    forecast_df = pd.DataFrame(forecast_results)
    if not forecast_df.empty:
        forecast_df = forecast_df.sort_values(['Date', 'Commodity']).reset_index(drop=True)
    
    return forecast_df, future_dates, future_forecasts

def _generate_forecasts_simplified(df_enhanced, model, feature_cols_enhanced, forecast_months):
    """Simplified forecast generation if main script is not available"""
    commodities = ['Rice', 'Cotton Yarn', 'Copper']
    last_date = df_enhanced['Date'].max()
    
    # Generate future dates
    future_dates = []
    current_date = last_date
    for i in range(1, forecast_months + 1):
        if current_date.month == 12:
            future_date = current_date.replace(year=current_date.year + 1, month=1, day=1)
        else:
            future_date = current_date.replace(month=current_date.month + 1, day=1)
        future_dates.append(future_date)
        current_date = future_date
    
    # Get last external values
    last_external = df_enhanced.groupby('Date')[['USD_PKR', 'Oil_Price', 'US_Confidence']].first().tail(1).iloc[0]
    
    future_forecasts = {}
    
    for commodity in commodities:
        commodity_data = df_enhanced[df_enhanced['Commodity'] == commodity].copy()
        if len(commodity_data) == 0:
            continue
            
        last_row = commodity_data.iloc[-1]
        commodity_forecasts = []
        
        for i, future_date in enumerate(future_dates):
            # Create future row (simplified - use last known values for most features)
            future_row = last_row.copy()
            future_row['Date'] = future_date
            future_row['Year'] = future_date.year
            future_row['Month'] = future_date.month
            future_row['Quarter'] = (future_date.month - 1) // 3 + 1
            future_row['Month_sin'] = np.sin(2 * np.pi * future_date.month / 12)
            future_row['Month_cos'] = np.cos(2 * np.pi * future_date.month / 12)
            future_row['Quarter_sin'] = np.sin(2 * np.pi * future_row['Quarter'] / 4)
            future_row['Quarter_cos'] = np.cos(2 * np.pi * future_row['Quarter'] / 4)
            future_row['USD_PKR'] = last_external['USD_PKR']
            future_row['Oil_Price'] = last_external['Oil_Price']
            future_row['US_Confidence'] = last_external['US_Confidence']
            
            # Update lag features
            if i == 0:
                future_row['Export_Value_USD_lag_1'] = last_row['Export_Value_USD']
            else:
                future_row['Export_Value_USD_lag_1'] = commodity_forecasts[-1]
            
            # Update other lags (simplified)
            for lag in [3, 6, 12, 24]:
                lag_col = f'Export_Value_USD_lag_{lag}'
                if i >= lag and len(commodity_forecasts) >= lag:
                    future_row[lag_col] = commodity_forecasts[-lag]
                elif len(commodity_data) >= (lag - i):
                    future_row[lag_col] = commodity_data['Export_Value_USD'].iloc[-(lag-i)]
                else:
                    future_row[lag_col] = last_row.get('Export_Value_USD', last_row['Export_Value_USD'])
            
            # Update enhanced features (interaction, ratio, momentum, etc.)
            future_row['Commodity_USD_PKR'] = future_row['Commodity_encoded'] * future_row['USD_PKR']
            future_row['Commodity_Oil'] = future_row['Commodity_encoded'] * future_row['Oil_Price']
            future_row['Commodity_Confidence'] = future_row['Commodity_encoded'] * future_row['US_Confidence']
            
            future_row['Export_USD_PKR_ratio'] = future_row['Export_Value_USD_lag_1'] / (future_row['USD_PKR'] + 1)
            future_row['Export_Oil_ratio'] = future_row['Export_Value_USD_lag_1'] / (future_row['Oil_Price'] + 1)
            future_row['Export_Confidence_ratio'] = future_row['Export_Value_USD_lag_1'] / (future_row['US_Confidence'] + 1)
            
            # Momentum features
            if i >= 3 and len(commodity_forecasts) >= 3:
                future_row['Export_momentum_3'] = ((commodity_forecasts[-1] - commodity_forecasts[-3]) / (commodity_forecasts[-3] + 1)) * 100 if commodity_forecasts[-3] > 0 else 0
            else:
                future_row['Export_momentum_3'] = last_row.get('Export_momentum_3', 0)
            
            if i >= 6 and len(commodity_forecasts) >= 6:
                future_row['Export_momentum_6'] = ((commodity_forecasts[-1] - commodity_forecasts[-6]) / (commodity_forecasts[-6] + 1)) * 100 if commodity_forecasts[-6] > 0 else 0
            else:
                future_row['Export_momentum_6'] = last_row.get('Export_momentum_6', 0)
            
            future_row['Export_YoY_change'] = last_row.get('Export_YoY_change', 0)
            
            # Trend and other features (simplified)
            future_row['Export_trend_3'] = last_row.get('Export_trend_3', 0)
            future_row['Export_trend_6'] = last_row.get('Export_trend_6', 0)
            future_row['Export_acceleration'] = last_row.get('Export_acceleration', 0)
            future_row['Export_volatility_3'] = last_row.get('Export_volatility_3', 0)
            future_row['Export_volatility_6'] = last_row.get('Export_volatility_6', 0)
            
            # Interaction features
            future_row['Lag1_USD_PKR'] = future_row['Export_Value_USD_lag_1'] * future_row['USD_PKR']
            future_row['Lag1_Oil'] = future_row['Export_Value_USD_lag_1'] * future_row['Oil_Price']
            future_row['Lag1_Confidence'] = future_row['Export_Value_USD_lag_1'] * future_row['US_Confidence']
            future_row['Commodity_Month'] = future_row['Commodity_encoded'] * future_row['Month_sin']
            future_row['Commodity_Quarter'] = future_row['Commodity_encoded'] * future_row['Quarter_sin']
            future_row['USD_PKR_squared'] = future_row['USD_PKR'] ** 2
            future_row['Oil_Price_squared'] = future_row['Oil_Price'] ** 2
            future_row['Export_percentile_12'] = last_row.get('Export_percentile_12', 0.5)
            
            # Lag vs rolling mean
            rolling_mean_6 = last_row.get('Export_Value_USD_rolling_mean_6', future_row['Export_Value_USD_lag_1'])
            rolling_mean_12 = last_row.get('Export_Value_USD_rolling_mean_12', future_row['Export_Value_USD_lag_1'])
            future_row['Lag1_vs_rolling_mean_6'] = (future_row['Export_Value_USD_lag_1'] - rolling_mean_6) / (rolling_mean_6 + 1)
            future_row['Lag1_vs_rolling_mean_12'] = (future_row['Export_Value_USD_lag_1'] - rolling_mean_12) / (rolling_mean_12 + 1)
            
            # External driver lags
            for lag in [1, 3, 6]:
                future_row[f'USD_PKR_lag_{lag}'] = last_row.get(f'USD_PKR_lag_{lag}', future_row['USD_PKR'])
                future_row[f'Oil_Price_lag_{lag}'] = last_row.get(f'Oil_Price_lag_{lag}', future_row['Oil_Price'])
                future_row[f'US_Confidence_lag_{lag}'] = last_row.get(f'US_Confidence_lag_{lag}', future_row['US_Confidence'])
            
            # Rolling features
            for window in [3, 6, 12]:
                future_row[f'Export_Value_USD_rolling_mean_{window}'] = last_row.get(f'Export_Value_USD_rolling_mean_{window}', future_row['Export_Value_USD_lag_1'])
                future_row[f'Export_Value_USD_rolling_std_{window}'] = last_row.get(f'Export_Value_USD_rolling_std_{window}', 0)
            
            # External driver changes
            future_row['USD_PKR_change_3'] = last_row.get('USD_PKR_change_3', 0)
            future_row['Oil_Price_change_3'] = last_row.get('Oil_Price_change_3', 0)
            future_row['US_Confidence_change_3'] = last_row.get('US_Confidence_change_3', 0)
            
            # Fill any missing features
            for col in feature_cols_enhanced:
                if col not in future_row.index:
                    future_row[col] = last_row.get(col, 0)
            
            # Prepare feature vector
            X_future_row = []
            for col in feature_cols_enhanced:
                if col in future_row.index:
                    X_future_row.append(future_row[col])
                else:
                    X_future_row.append(0)
            
            X_future = np.array(X_future_row).reshape(1, -1)
            
            # Make prediction (model outputs log-transformed value)
            try:
                forecast_value_log = model.predict(X_future, num_iteration=model.best_iteration)[0]
                forecast_value = np.expm1(forecast_value_log)  # Inverse log transformation
                forecast_value = max(0, forecast_value)  # Ensure non-negative
            except:
                # Fallback if model prediction fails
                forecast_value = last_row['Export_Value_USD']
            
            commodity_forecasts.append(forecast_value)
        
        future_forecasts[commodity] = commodity_forecasts
    
    return future_dates, future_forecasts

def calculate_uncertainty_bands(forecast_df, historical_df):
    """Calculate uncertainty bands for forecasts based on historical volatility"""
    commodities = forecast_df['Commodity'].unique()
    result_df = forecast_df.copy()
    
    result_df['Lower_Bound'] = 0.0
    result_df['Upper_Bound'] = 0.0
    result_df['Confidence_Range'] = 0.0
    
    for commodity in commodities:
        # Get historical data for this commodity
        hist_data = historical_df[historical_df['Commodity'] == commodity]['Export_Value_USD']
        recent_hist = hist_data.tail(12)
        
        # Calculate coefficient of variation
        hist_mean = recent_hist.mean()
        hist_std = recent_hist.std()
        hist_cv = hist_std / hist_mean if hist_mean > 0 else 0.15
        
        # Get forecasts for this commodity
        comm_forecast = result_df[result_df['Commodity'] == commodity]
        
        # Calculate uncertainty for each forecast point
        for idx in comm_forecast.index:
            forecast_value = comm_forecast.loc[idx, 'Forecast_Export_Value_USD']
            month_idx = comm_forecast.index.get_loc(idx) % len(comm_forecast[comm_forecast['Commodity'] == commodity])
            
            # Uncertainty increases slightly over forecast horizon
            uncertainty_factor = 1.5 * hist_cv
            uncertainty_multiplier = 1 + (month_idx * 0.05)  # 5% increase per month
            
            upper = forecast_value * (1 + uncertainty_factor * uncertainty_multiplier)
            lower = forecast_value * (1 - uncertainty_factor * uncertainty_multiplier)
            lower = max(lower, 0)  # Ensure non-negative
            
            result_df.loc[idx, 'Upper_Bound'] = upper
            result_df.loc[idx, 'Lower_Bound'] = lower
            result_df.loc[idx, 'Confidence_Range'] = upper - lower
    
    return result_df
