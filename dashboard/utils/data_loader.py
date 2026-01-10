"""
Data Loading Utilities for Streamlit Dashboard
Handles loading of data, models, and metadata
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from pathlib import Path

# Get project root directory (two levels up from utils: dashboard/utils -> dashboard -> project_root)
PROJECT_ROOT = Path(__file__).parent.parent.parent

def load_historical_data():
    """Load historical export data"""
    # Try multiple paths - project root first, then relative to current file
    possible_paths = [
        PROJECT_ROOT / 'merged_export_dataset_2010_2025.csv',
        Path(__file__).parent.parent.parent / 'merged_export_dataset_2010_2025.csv',
        Path('merged_export_dataset_2010_2025.csv')
    ]
    
    data_path = None
    for path in possible_paths:
        if path.exists():
            data_path = path
            break
    
    if data_path is None:
        raise FileNotFoundError(f"Data file not found. Tried: {[str(p) for p in possible_paths]}")
    
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Commodity', 'Date']).reset_index(drop=True)
    return df

def load_wide_format_data():
    """Load wide format data for analysis"""
    # Try multiple paths
    possible_paths = [
        PROJECT_ROOT / 'merged_export_dataset_wide_2010_2025.csv',
        Path(__file__).parent.parent.parent / 'merged_export_dataset_wide_2010_2025.csv',
        Path('merged_export_dataset_wide_2010_2025.csv')
    ]
    
    data_path = None
    for path in possible_paths:
        if path.exists():
            data_path = path
            break
    
    if data_path is None:
        raise FileNotFoundError(f"Wide format data file not found. Tried: {[str(p) for p in possible_paths]}")
    
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    return df

def load_model():
    """Load trained LightGBM model"""
    # Try multiple paths
    possible_paths = [
        PROJECT_ROOT / 'models' / 'best_model_lgbm.pkl',
        Path(__file__).parent.parent.parent / 'models' / 'best_model_lgbm.pkl',
        Path('models/best_model_lgbm.pkl')
    ]
    
    model_path = None
    for path in possible_paths:
        if path.exists():
            model_path = path
            break
    
    if model_path is None:
        raise FileNotFoundError(f"Model file not found. Tried: {[str(p) for p in possible_paths]}")
    
    model = joblib.load(model_path)
    return model

def load_feature_names():
    """Load feature names for the model"""
    # Try multiple paths
    possible_paths = [
        PROJECT_ROOT / 'models' / 'feature_names_lgbm.json',
        Path(__file__).parent.parent.parent / 'models' / 'feature_names_lgbm.json',
        Path('models/feature_names_lgbm.json')
    ]
    
    feature_path = None
    for path in possible_paths:
        if path.exists():
            feature_path = path
            break
    
    if feature_path is None:
        raise FileNotFoundError(f"Feature names file not found. Tried: {[str(p) for p in possible_paths]}")
    
    with open(feature_path, 'r') as f:
        feature_names = json.load(f)
    return feature_names

def load_model_metadata():
    """Load model metadata"""
    # Try multiple paths
    possible_paths = [
        PROJECT_ROOT / 'models' / 'model_metadata.json',
        Path(__file__).parent.parent.parent / 'models' / 'model_metadata.json',
        Path('models/model_metadata.json')
    ]
    
    metadata_path = None
    for path in possible_paths:
        if path.exists():
            metadata_path = path
            break
    
    if metadata_path is None:
        return None
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return metadata

def get_commodity_data(df, commodity):
    """Get data for a specific commodity"""
    if commodity == 'All':
        return df
    return df[df['Commodity'] == commodity].copy()

def get_latest_forecast():
    """Get the most recent forecast file"""
    # Try multiple paths
    possible_dirs = [
        PROJECT_ROOT / 'forecasts',
        Path(__file__).parent.parent.parent / 'forecasts',
        Path('forecasts')
    ]
    
    forecasts_dir = None
    for dir_path in possible_dirs:
        if dir_path.exists():
            forecasts_dir = dir_path
            break
    
    if forecasts_dir is None:
        return None
    
    # Find latest forecast CSV
    forecast_files = list(forecasts_dir.glob('forecast_*.csv'))
    if not forecast_files:
        return None
    
    # Get most recent file
    latest_file = max(forecast_files, key=os.path.getctime)
    forecast_df = pd.read_csv(latest_file)
    forecast_df['Date'] = pd.to_datetime(forecast_df['Date'])
    return forecast_df

def get_statistics(df, commodity=None):
    """Calculate statistics for commodity(ies)"""
    if commodity and commodity != 'All':
        df = df[df['Commodity'] == commodity]
    
    stats = {
        'total_months': df['Date'].nunique(),
        'total_commodities': df['Commodity'].nunique(),
        'start_date': df['Date'].min(),
        'end_date': df['Date'].max(),
        'total_export_value': df['Export_Value_USD'].sum(),
        'average_export_value': df['Export_Value_USD'].mean(),
        'median_export_value': df['Export_Value_USD'].median(),
        'std_export_value': df['Export_Value_USD'].std(),
        'min_export_value': df['Export_Value_USD'].min(),
        'max_export_value': df['Export_Value_USD'].max()
    }
    
    # Per-commodity statistics if 'All' selected
    if commodity == 'All':
        commodity_stats = {}
        for comm in df['Commodity'].unique():
            comm_df = df[df['Commodity'] == comm]
            commodity_stats[comm] = {
                'average': comm_df['Export_Value_USD'].mean(),
                'median': comm_df['Export_Value_USD'].median(),
                'std': comm_df['Export_Value_USD'].std(),
                'min': comm_df['Export_Value_USD'].min(),
                'max': comm_df['Export_Value_USD'].max(),
                'total': comm_df['Export_Value_USD'].sum()
            }
        stats['by_commodity'] = commodity_stats
    
    return stats
