# Demand Prediction for Pakistan's Export Commodities - Project Plan

## 📋 Project Overview

**Objective**: Develop a forecasting system to predict future demand for Pakistan's key export commodities (Rice, Cotton Yarn, and Copper) using machine learning models and deploy it via a web dashboard with an intelligent Trade Chatbot.

**Commodities**: Rice (HS Code: 1006), Cotton Yarn (HS Code: 520512), Copper (HS Code: 7403)

**Time Period**: January 2010 to August 2025 (188 months of historical data)

**External Drivers**: USD/PKR Exchange Rate, Brent Oil Price, US Consumer Confidence Index

---

## 🎯 Project Goals

1. Build accurate forecasting models for export commodity demand
2. Compare multiple model approaches and select the best performing one
3. Create an interactive web dashboard for visualization and forecasting
4. Implement a RAG-powered Trade Chatbot for intelligent querying

---

## 🤖 Selected Models

After analysis of the dataset (188 months, 3 commodities), the following models have been selected:

### 1. **LightGBM/XGBoost (Multi-Output)** ⭐ Primary Model
- **Rationale**: 
  - Excellent performance with small-to-medium datasets (188 months × 3 commodities = 564 samples)
  - Handles mixed features (numeric + categorical + time features)
  - Built-in regularization prevents overfitting
  - Fast training and interpretable results
  - Native multi-output support for all 3 commodities simultaneously
- **Format**: Long format dataset (564 rows)
- **Features**: Lag features, rolling statistics, seasonal features, external regressors, commodity embeddings

### 2. **Prophet (with External Regressors)**
- **Rationale**:
  - Designed for business time series with strong seasonality handling
  - Works well with limited data (188 rows per commodity)
  - Built-in external regressors support
  - Highly interpretable (trend, seasonality components)
- **Format**: Wide format (3 separate models, one per commodity)
- **External Regressors**: USD_PKR, Oil_Price, US_Confidence

### 3. **SARIMAX (Seasonal ARIMA with eXogenous regressors)**
- **Rationale**:
  - Statistical rigor and interpretability
  - Handles seasonality and external regressors
  - Good baseline for comparison
  - Traditional time series approach for validation
- **Format**: Wide format (3 separate models, one per commodity)
- **External Regressors**: USD_PKR, Oil_Price, US_Confidence

---

## 📊 Dataset Information

### Current Data Status
- ✅ **Data Collection**: Complete
- ✅ **Data Preprocessing**: Complete
- ✅ **Data Merging**: Complete
- ✅ **Missing Data Handling**: Complete

### Available Datasets
1. `merged_export_dataset_2010_2025.csv` - **Long Format** (564 rows)
   - Best for: LightGBM multi-output model
   - Structure: Date, Commodity, HS_Code, Export_Value_USD, Weight_kg, USD_PKR, Oil_Price, US_Confidence

2. `merged_export_dataset_wide_2010_2025.csv` - **Wide Format** (188 rows)
   - Best for: Prophet and SARIMAX (per-commodity models)
   - Structure: Date, Rice_Export_Value_USD, Cotton_Export_Value_USD, Copper_Export_Value_USD, etc.

---

## 📓 Jupyter Notebook Structure

### Main Notebook: `Forecasting_Pipeline.ipynb`

#### **Section 1: Project Setup & Data Loading**
- Import libraries
- Load configuration
- Set random seeds for reproducibility
- Load merged datasets (both long and wide format)

#### **Section 2: Exploratory Data Analysis (EDA)**
- Dataset overview and statistics
- Time series plots for each commodity
- Distribution analysis
- Correlation analysis between commodities and external drivers
- Seasonality and trend decomposition
- External drivers impact visualization

#### **Section 3: Feature Engineering**
- **For LightGBM/XGBoost:**
  - Lag features (1, 3, 6, 12 months)
  - Rolling statistics (mean, std over 3, 6, 12 months)
  - Seasonal features (Month, Quarter with cyclical encoding)
  - External driver features (current + lags)
  - Commodity-specific interactions (Commodity × Month, Commodity × USD_PKR)
  - Commodity embeddings/categorical encoding
- **For Prophet & SARIMAX:**
  - External regressors preparation
  - Stationarity checks and transformations (if needed)

#### **Section 4: Data Preprocessing**
- Train/Validation/Test split (Time-series split: ~80/10/10 or 85/15)
- Scaling/Normalization (for LightGBM if needed)
- Stationarity transformation (for SARIMAX)
- Sequence preparation (for LSTM if needed later)

#### **Section 5: Model 1 - LightGBM/XGBoost (Multi-Output)**
- Model architecture setup
- Hyperparameter tuning (optional: Optuna/Hyperopt)
- Training with cross-validation
- Validation predictions
- Evaluation metrics (MAPE, RMSE, MAE)
- Feature importance analysis

#### **Section 6: Model 2 - Prophet (Per Commodity)**
- Train 3 separate Prophet models (Rice, Cotton, Copper)
- Configure external regressors
- Model training and validation
- Validation predictions for each commodity
- Evaluation metrics (MAPE, RMSE, MAE)
- Trend and seasonality component visualization

#### **Section 7: Model 3 - SARIMAX (Per Commodity)**
- Auto ARIMA parameter selection (or grid search)
- Train 3 separate SARIMAX models (Rice, Cotton, Copper)
- Include external regressors
- Model training and validation
- Validation predictions for each commodity
- Evaluation metrics (MAPE, RMSE, MAE)
- Residual analysis

#### **Section 8: Model Comparison & Selection**
- Comprehensive comparison table (all metrics for all models)
- Visualization: Forecast comparisons
- Error analysis
- **Best Model Selection** based on:
  - Lowest MAPE
  - Lowest RMSE
  - Consistency across commodities
  - Interpretability and robustness

#### **Section 9: Best Model Evaluation**
- Test set predictions using best model
- Final evaluation metrics
- Forecast visualization with confidence intervals
- Error analysis and diagnostics
- Performance summary by commodity

#### **Section 10: Future Forecasting**
- Generate forecasts for next 6-12 months
- Uncertainty/confidence intervals
- Forecast visualization
- Export forecast results

#### **Section 11: Model Persistence**
- Save best model (joblib/pickle)
- Save preprocessors/scalers (if used)
- Save feature engineering pipeline
- Save external regressor data for future predictions
- Create model metadata (version, date, performance metrics)

---

## 🌐 Web Dashboard Architecture

### Framework: **Streamlit** (Recommended for FYP) or Flask/FastAPI

### Dashboard Components:
1. **Forecast Visualization Dashboard**
   - Interactive time series plots
   - Forecast predictions with confidence intervals
   - Model performance metrics display
   - Historical vs Forecasted comparisons

2. **Forecast Generation Interface**
   - User input: Forecast horizon (months)
   - Real-time forecast generation using saved model
   - Download forecast results (CSV/Excel)

3. **Trade Chatbot with RAG**
   - Retrieval module: FAISS for document search
   - Generative module: LLM (OpenAI/Llama) with RAG
   - Query interface for:
     - Forecast explanations
     - Historical data queries
     - Market insights
     - Trade recommendations
   - Feedback mechanism for continuous improvement

4. **Model Information Panel**
   - Selected model details
   - Performance metrics
   - Feature importance (for LightGBM)
   - Model version and training date

---

## 📁 Project File Structure

```
Commodity_Forecating2/
├── Data/                                    # Raw data files
│   ├── Pakistan_Exports_*.xlsx
│   ├── USD_PKR_Exchange_Rate_*.csv
│   ├── Brent_Oil_Prices_*.csv
│   └── US_Consumer_Confidence_*.csv
│
├── merged_export_dataset_2010_2025.csv      # Long format (for LightGBM)
├── merged_export_dataset_wide_2010_2025.csv # Wide format (for Prophet/SARIMAX)
│
├── Forecasting_Pipeline.ipynb               # Main notebook ⭐
│
├── models/                                   # Saved models
│   ├── best_model_lgbm.pkl
│   ├── best_model_prophet_rice.pkl
│   ├── best_model_prophet_cotton.pkl
│   ├── best_model_prophet_copper.pkl
│   ├── best_model_sarimax_rice.pkl
│   ├── best_model_sarimax_cotton.pkl
│   ├── best_model_sarimax_copper.pkl
│   └── preprocessor.pkl
│
├── forecasts/                                # Generated forecasts
│   └── forecast_results.csv
│
├── dashboard/                                # Web dashboard
│   ├── app.py                               # Streamlit/Flask app
│   ├── chatbot.py                           # RAG chatbot module
│   ├── utils.py                             # Helper functions
│   └── templates/                           # HTML templates (if Flask)
│
├── requirements.txt                         # Python dependencies
├── README.md                                # Project documentation
├── plan.md                                  # This file
│
└── Documentation/                           # Additional docs
    ├── DATA_ANALYSIS_SUMMARY.md
    ├── FORECASTING_FORMAT_MODEL_RECOMMENDATIONS.md
    └── MERGE_STRATEGY_SUMMARY.md
```

---

## 🔄 Development Workflow

### Phase 1: Model Development (Jupyter Notebook) ✅ In Progress
1. ✅ Data analysis and preprocessing (COMPLETE)
2. ⏭️ EDA and visualization
3. ⏭️ Feature engineering
4. ⏭️ Model training (LightGBM, Prophet, SARIMAX)
5. ⏭️ Model evaluation and comparison
6. ⏭️ Best model selection and persistence

### Phase 2: Web Dashboard Development
1. Load saved model
2. Create forecast visualization interface
3. Implement interactive controls
4. Integrate RAG chatbot
5. Testing and refinement

### Phase 3: Documentation & Presentation
1. Complete project documentation
2. Create presentation materials
3. Prepare demonstration scenarios

---

## 📊 Evaluation Metrics

### Primary Metrics:
- **MAPE** (Mean Absolute Percentage Error) - Primary metric
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)

### Additional Metrics:
- **R² Score** (Coefficient of Determination)
- **Directional Accuracy** (Percentage of correct trend predictions)

### Expected Performance:
Based on dataset characteristics:
- **LightGBM**: ~8-12% MAPE (Best expected)
- **Prophet**: ~10-15% MAPE
- **SARIMAX**: ~12-18% MAPE

---

## 🛠️ Technology Stack

### Machine Learning:
- **LightGBM** / **XGBoost** - Gradient boosting
- **Prophet** - Facebook's time series forecasting
- **statsmodels** - SARIMAX implementation
- **scikit-learn** - Preprocessing and evaluation
- **pandas, numpy** - Data manipulation

### Visualization:
- **matplotlib, seaborn** - Static plots
- **plotly** - Interactive visualizations

### Web Dashboard:
- **Streamlit** (Recommended) or **Flask/FastAPI**
- **Plotly/Dash** - Interactive charts
- **FAISS** - Vector similarity search for RAG
- **LangChain/LLM** - RAG chatbot implementation

### Model Persistence:
- **joblib/pickle** - Model serialization
- **JSON** - Metadata storage

---

## ✅ Decision Summary

### Models Selected:
1. ✅ **LightGBM/XGBoost (Multi-Output)** - Primary model
2. ✅ **Prophet (with External Regressors)** - Secondary model
3. ✅ **SARIMAX** - Baseline/comparison model

### Notebook Approach:
- ✅ Single comprehensive notebook (`Forecasting_Pipeline.ipynb`)
- ✅ Complete workflow from EDA to model persistence
- ✅ Well-documented with markdown cells

### Deployment Strategy:
- ✅ Save best model after notebook execution
- ✅ Load saved model in web dashboard
- ✅ Separate development (notebook) and production (dashboard) environments

### Chatbot Implementation:
- ✅ RAG architecture with FAISS for retrieval
- ✅ LLM for generation
- ✅ Integration in web dashboard

---

## 📝 Next Steps

1. ⏭️ Create the Jupyter notebook template with all sections
2. ⏭️ Implement EDA and visualization code
3. ⏭️ Implement feature engineering
4. ⏭️ Implement model training for all 3 models
5. ⏭️ Implement model comparison and selection
6. ⏭️ Implement model persistence
7. ⏭️ Develop web dashboard
8. ⏭️ Integrate RAG chatbot

---

## 📅 Project Timeline (Suggested)

- **Week 1-2**: Notebook development (EDA, Feature Engineering, Model Training)
- **Week 3**: Model evaluation, comparison, and selection
- **Week 4**: Web dashboard development
- **Week 5**: RAG chatbot integration and testing
- **Week 6**: Documentation, presentation preparation, final testing

---

## 🎯 Success Criteria

1. ✅ All three models successfully trained and evaluated
2. ✅ Best model selected based on comprehensive metrics
3. ✅ Model successfully saved and can be loaded
4. ✅ Web dashboard functional with forecast visualizations
5. ✅ RAG chatbot operational for trade-related queries
6. ✅ Complete documentation and reproducible results

---

**Last Updated**: [Current Date]  
**Status**: Planning Phase ✅ - Ready for Implementation