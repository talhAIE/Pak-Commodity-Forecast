# Demand Prediction for Pakistan's Export Commodities

Final Year Project: Forecasting system for Pakistan's export commodities (Rice, Cotton Yarn, and Copper) using machine learning models.

## 📋 Project Overview

This project aims to predict future demand for Pakistan's key export commodities using advanced forecasting models and deploy insights through a web dashboard with an intelligent Trade Chatbot.

### Commodities
- **Rice** (HS Code: 1006)
- **Cotton Yarn** (HS Code: 520512)
- **Copper** (HS Code: 7403)

### Time Period
- Historical Data: January 2010 to August 2025 (188 months)

### External Drivers
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
│   └── merge_and_preprocess_data.py         # Data merging and preprocessing
│
├── Documentation/                            # Project documentation
│   ├── DATA_ANALYSIS_SUMMARY.md             # Data analysis results
│   ├── FORECASTING_FORMAT_MODEL_RECOMMENDATIONS.md
│   └── MERGE_STRATEGY_SUMMARY.md
│
├── models/                                   # Saved ML models
│   └── (Models will be saved here after training)
│
├── forecasts/                                # Generated forecast results
│   └── (Forecast CSV/Excel files will be saved here)
│
├── dashboard/                                # Web dashboard application
│   ├── templates/                           # HTML templates (if using Flask)
│   └── (Dashboard files will be added here)
│
├── merged_export_dataset_2010_2025.csv      # Long format dataset (for LightGBM)
├── merged_export_dataset_wide_2010_2025.csv # Wide format dataset (for Prophet/SARIMAX)
├── plan.md                                   # Project plan and decisions
└── README.md                                 # This file
```

## 🤖 Models

1. **LightGBM/XGBoost (Multi-Output)** - Primary model
2. **Prophet (with External Regressors)** - Secondary model
3. **SARIMAX** - Baseline/comparison model

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Required libraries (see `requirements.txt` - to be created)

### Running Scripts

All scripts in the `scripts/` folder should be run from the **project root directory**:

```bash
# Run data preprocessing
python scripts/merge_and_preprocess_data.py

# Run data analysis
python scripts/analyze_data.py

# Check oil data gaps
python scripts/check_oil_missing.py
```

## 📊 Datasets

### Long Format (`merged_export_dataset_2010_2025.csv`)
- **Rows**: 564 (188 months × 3 commodities)
- **Use for**: LightGBM multi-output model
- **Columns**: Date, Commodity, HS_Code, Export_Value_USD, Weight_kg, USD_PKR, Oil_Price, US_Confidence

### Wide Format (`merged_export_dataset_wide_2010_2025.csv`)
- **Rows**: 188 (one per month)
- **Use for**: Prophet and SARIMAX (per-commodity models)
- **Columns**: Date, Rice_Export_Value_USD, Cotton_Export_Value_USD, Copper_Export_Value_USD, etc.

## 📝 Evaluation Metrics

- **MAPE** (Mean Absolute Percentage Error) - Primary metric
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)

## 🔄 Development Status

- ✅ Data Collection and Preprocessing - Complete
- ⏭️ EDA and Visualization - In Progress
- ⏭️ Model Development - Pending
- ⏭️ Web Dashboard - Pending
- ⏭️ RAG Chatbot Integration - Pending

## 📖 Documentation

See the `Documentation/` folder for detailed analysis summaries and recommendations.

## 👥 Authors

Final Year Project

## 📄 License

This project is for academic purposes.
