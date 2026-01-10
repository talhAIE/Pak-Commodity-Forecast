# Streamlit Dashboard - Pakistan Export Commodities Forecasting

## Overview

Interactive web dashboard for forecasting Pakistan export commodities (Rice, Cotton Yarn, Copper) using trained LightGBM model.

## Features

1. **Overview Page**: Dashboard summary with key statistics and quick overview charts
2. **Historical Analysis**: Interactive time series analysis with statistics and comparisons
3. **Forecast Generator**: Generate custom forecasts (1-12 months) with uncertainty bands
4. **Model Performance**: View model metrics, feature importance, and performance analysis
5. **Insights & Analytics**: Key insights, trend analysis, growth patterns, and risk assessment
6. **External Drivers**: Analysis of USD/PKR, Oil Price, and US Consumer Confidence impacts

## Installation

1. Ensure all dependencies are installed:
```bash
pip install -r ../requirements.txt
```

2. Make sure you have:
   - Trained model saved in `../models/best_model_lgbm.pkl`
   - Feature names in `../models/feature_names_lgbm.json`
   - Historical data in `../merged_export_dataset_2010_2025.csv`
   - Wide format data in `../merged_export_dataset_wide_2010_2025.csv`

## Running the Dashboard

From the project root directory:

```bash
streamlit run dashboard/app.py
```

Or from the dashboard directory:

```bash
cd dashboard
streamlit run app.py
```

The dashboard will open in your default web browser at `http://localhost:8501`

## Dashboard Structure

```
dashboard/
├── app.py                      # Main Streamlit application
├── utils/
│   ├── data_loader.py         # Data and model loading utilities
│   ├── forecast_generator.py  # Forecast generation utilities
│   └── visualizations.py      # Plotly chart creation utilities
└── README.md                   # This file
```

## Pages

1. **Overview**: Quick stats, latest forecast summary, model info
2. **Historical Analysis**: Time series plots, statistics, distributions, correlations
3. **Forecast Generator**: Generate custom forecasts with interactive controls
4. **Model Performance**: Validation/test metrics, feature importance
5. **Insights & Analytics**: Trend analysis, growth patterns, risk indicators
6. **External Drivers**: External driver trends and correlation analysis

## Usage Tips

- Use the sidebar to navigate between pages
- Forecast Generator allows customization of forecast horizon (1-12 months)
- All charts are interactive - zoom, pan, hover for details
- Download forecasts as CSV from the Forecast Generator page
- View feature importance to understand model behavior
