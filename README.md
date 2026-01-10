# Demand Prediction for Pakistan's Export Commodities

Final Year Project: Forecasting system for Pakistan's export commodities (Rice, Cotton Yarn, and Copper) using advanced machine learning models with interactive web dashboard.

## 📋 Project Overview

This project aims to predict future demand for Pakistan's key export commodities using advanced forecasting models and deploy insights through a web dashboard. The system uses LightGBM (gradient boosting) as the primary model to forecast export values with high accuracy.

### Commodities Tracked
- **Rice** (HS Code: 1006)
- **Cotton Yarn** (HS Code: 520512)
- **Copper** (HS Code: 7403)

### Time Period
- Historical Data: January 2010 to December 2025 (188+ months)
- Forecast Horizon: Configurable (1-12 months)

### External Drivers Analyzed
- USD/PKR Exchange Rate
- Brent Oil Price
- US Consumer Confidence Index

## 🏗️ Project Structure

```
Commodity_Forecating2/
├── Data/                                    # Raw data files
│   ├── Pakistan_Exports_*.xlsx             # Export data for each commodity
│   ├── USD_PKR_Exchange_Rate_*.csv         # Exchange rate data
│   ├── Brent_Oil_Prices_*.csv              # Oil price data
│   └── US_Consumer_Confidence_*.csv        # Consumer confidence data
│
├── scripts/                                  # Utility scripts
│   ├── analyze_data.py                      # Data analysis script
│   ├── check_oil_missing.py                 # Oil data gap analysis
│   ├── check_project_status.py              # Project status checker
│   ├── generate_forecast.py                 # Standalone forecast generator
│   └── merge_and_preprocess_data.py         # Data merging and preprocessing
│
├── Documentation/                            # Project documentation
│   ├── DATA_ANALYSIS_SUMMARY.md             # Data analysis results
│   ├── FORECASTING_FORMAT_MODEL_RECOMMENDATIONS.md
│   └── MERGE_STRATEGY_SUMMARY.md
│
├── models/                                   # Saved ML models
│   ├── best_model_lgbm.pkl                  # Trained LightGBM model
│   ├── feature_names_lgbm.json              # Feature names for model
│   └── model_metadata.json                  # Model metadata and metrics
│
├── forecasts/                                # Generated forecast results
│   ├── forecast_*.csv                       # Forecast CSV files
│   └── forecast_*_visualization.png         # Forecast visualizations
│
├── dashboard/                                # Streamlit web dashboard
│   ├── app.py                               # Main Streamlit application
│   ├── utils/                               # Dashboard utilities
│   │   ├── data_loader.py                   # Data and model loading
│   │   ├── forecast_generator.py            # Forecast generation
│   │   └── visualizations.py                # Interactive Plotly charts
│   └── README.md                            # Dashboard documentation
│
├── Forecasting_Pipeline.ipynb               # Main Jupyter notebook (EDA, training, evaluation)
├── merged_export_dataset_2010_2025.csv      # Long format dataset (for LightGBM)
├── merged_export_dataset_wide_2010_2025.csv # Wide format dataset (for Prophet/SARIMAX)
├── requirements.txt                         # Python dependencies
├── DEPLOYMENT_GUIDE.md                      # Detailed deployment instructions
├── QUICK_DEPLOY.md                          # Quick deployment reference
├── plan.md                                  # Project plan and decisions
└── README.md                                # This file
```

## 🤖 Machine Learning Models

### Primary Model: LightGBM (Gradient Boosting)
- **Type**: Multi-output regression
- **Status**: ✅ Trained and deployed
- **Performance**:
  - Validation MAPE: ~16-21%
  - Test MAPE: ~21.74%
  - R² Score: 0.7173
- **Features**: Advanced feature engineering with lag features, rolling statistics, external drivers, and interaction features
- **Output**: Forecasts for all 3 commodities simultaneously

### Secondary Models (Implemented but not primary)
- **Prophet** (with External Regressors) - Time series forecasting
- **SARIMAX** - Baseline/comparison model

### Model Selection Criteria
- MAPE (Mean Absolute Percentage Error) - Primary metric
- RMSE, MAE, R² Score - Secondary metrics

## 🚀 Getting Started

### Prerequisites
- Python 3.8+ (Python 3.10+ recommended)
- pip package manager

### Installation

1. **Clone the repository** (or download as ZIP):
   ```bash
   git clone <your-repository-url>
   cd Commodity_Forecating2
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Running Scripts

All scripts should be run from the **project root directory**:

```bash
# Run data preprocessing
python scripts/merge_and_preprocess_data.py

# Run data analysis
python scripts/analyze_data.py

# Check oil data gaps
python scripts/check_oil_missing.py

# Generate forecasts (6 months default)
python scripts/generate_forecast.py
```

## 📊 Datasets

### Long Format (`merged_export_dataset_2010_2025.csv`)
- **Rows**: 564 (188 months × 3 commodities)
- **Format**: One row per commodity per month
- **Use for**: LightGBM multi-output model
- **Columns**: 
  - Date, Commodity, HS_Code
  - Export_Value_USD, Weight_kg
  - USD_PKR, Oil_Price, US_Confidence

### Wide Format (`merged_export_dataset_wide_2010_2025.csv`)
- **Rows**: 188 (one per month)
- **Format**: One row per month with separate columns for each commodity
- **Use for**: Prophet and SARIMAX (per-commodity models)
- **Columns**: 
  - Date
  - Rice_Export_Value_USD, Cotton_Yarn_Export_Value_USD, Copper_Export_Value_USD
  - USD_PKR, Oil_Price, US_Confidence

## 🌐 Web Dashboard

### Running Locally

Launch the interactive Streamlit dashboard:

```bash
streamlit run dashboard/app.py
```

The dashboard will open at `http://localhost:8501`

### Dashboard Features

1. **Overview Page**
   - Key performance indicators
   - Latest forecast summary
   - Quick overview charts

2. **Historical Analysis**
   - Interactive time series plots
   - Statistical summaries
   - Distribution analysis
   - Commodity comparisons
   - Correlation heatmaps

3. **Forecast Generator** ⭐
   - Generate custom forecasts (1-12 months)
   - Select specific commodities
   - Uncertainty bands visualization
   - Download forecasts as CSV

4. **Model Performance**
   - Validation and test metrics
   - Feature importance charts
   - Model metadata

5. **Insights & Analytics**
   - Trend analysis
   - Growth patterns
   - Risk assessment
   - Year-over-year comparisons

6. **External Drivers**
   - USD/PKR exchange rate trends
   - Oil price analysis
   - US consumer confidence impact
   - Correlation analysis

### Deploying Online

Deploy your dashboard to Streamlit Community Cloud (free) so friends can access it:

**Quick Steps:**
1. Push code to GitHub (public repository required for free tier)
2. Go to https://share.streamlit.io
3. Sign in with GitHub
4. Deploy app with main file: `dashboard/app.py`

**Detailed Instructions:** See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) or [QUICK_DEPLOY.md](QUICK_DEPLOY.md)

## 📝 Evaluation Metrics

- **MAPE** (Mean Absolute Percentage Error) - Primary metric (target: <25%)
- **RMSE** (Root Mean Squared Error) - Scale-dependent metric
- **MAE** (Mean Absolute Error) - Average error magnitude
- **R² Score** - Coefficient of determination (0-1, higher is better)

### Current Model Performance (LightGBM)

**Test Set Results:**
- MAPE: 21.74%
- RMSE: $66,219,447
- MAE: $36,900,626
- R² Score: 0.7173

**By Commodity (Test Set):**
- Rice: MAPE = 31.92%
- Cotton Yarn: MAPE = 25.12%
- Copper: MAPE = 8.18% ⭐ (Best accuracy)

## 🔄 Development Status

- ✅ Data Collection and Preprocessing - **Complete**
- ✅ EDA and Visualization - **Complete**
- ✅ Feature Engineering - **Complete**
- ✅ Model Training (LightGBM, Prophet, SARIMAX) - **Complete**
- ✅ Model Evaluation and Selection - **Complete**
- ✅ Model Persistence - **Complete**
- ✅ Web Dashboard (Streamlit) - **Complete**
- ✅ Forecast Generation Script - **Complete**
- ✅ Deployment Setup - **Complete**
- ⏭️ RAG Chatbot Integration - **Pending**

## 📖 Documentation

- **Data Analysis**: See `Documentation/DATA_ANALYSIS_SUMMARY.md`
- **Model Recommendations**: See `Documentation/FORECASTING_FORMAT_MODEL_RECOMMENDATIONS.md`
- **Merge Strategy**: See `Documentation/MERGE_STRATEGY_SUMMARY.md`
- **Dashboard Guide**: See `dashboard/README.md`
- **Deployment**: See `DEPLOYMENT_GUIDE.md`

## 🔧 Key Technologies

- **Machine Learning**: LightGBM, Prophet, SARIMAX (pmdarima)
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Matplotlib
- **Web Framework**: Streamlit
- **Development**: Jupyter Notebook, Python 3.10+

## 👥 Authors

Final Year Project - Pakistan Export Commodities Forecasting System

## 📄 License

This project is for academic purposes only.

## 🤝 Contributing

This is a final year project. For questions or suggestions, please open an issue on GitHub.

---

**Note**: The dashboard requires trained models and data files to be present in the repository. Ensure all files from the `models/` and root directory (CSV files) are included when deploying.
