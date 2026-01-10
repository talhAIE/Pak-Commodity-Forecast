"""
Main Streamlit Dashboard Application
Pakistan Export Commodities Forecasting System
"""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add parent directory and utils to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(Path(__file__).parent / 'utils'))

# Import utilities
from utils.data_loader import (
    load_historical_data, 
    load_wide_format_data,
    load_model,
    load_feature_names,
    load_model_metadata,
    get_commodity_data,
    get_statistics,
    get_latest_forecast
)
from utils.visualizations import (
    create_time_series_plot,
    create_comparison_chart,
    create_correlation_heatmap,
    create_statistics_chart,
    create_feature_importance_chart
)
from utils.forecast_generator import (
    generate_forecast_for_dashboard,
    calculate_uncertainty_bands
)

# Set page configuration
st.set_page_config(
    page_title="Pakistan Export Commodities Forecasting",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state for caching
@st.cache_data
def load_all_data():
    """Load all data with caching"""
    try:
        df_long = load_historical_data()
        df_wide = load_wide_format_data()
        model = load_model()
        feature_names = load_feature_names()
        metadata = load_model_metadata()
        return df_long, df_wide, model, feature_names, metadata
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None, None, None

# Sidebar navigation
st.sidebar.title("Navigation")
st.sidebar.markdown("---")

page = st.sidebar.selectbox(
    "Select Page",
    ["Overview", "Historical Analysis", "Forecast Generator", 
     "Model Performance", "Insights & Analytics", "External Drivers"]
)

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    """
    **Pakistan Export Commodities Forecasting System**
    
    Forecast export demand for:
    - Rice
    - Cotton Yarn  
    - Copper
    
    Using Advanced LightGBM Model
    """
)

# Page functions - Define before use
def show_overview_page(df_long, metadata):
    """Overview/Home Page"""
    st.title("Dashboard Overview")
    st.markdown("---")
    
    # Key Statistics Cards
    st.subheader("Key Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    stats = get_statistics(df_long)
    
    with col1:
        st.metric(
            label="Total Historical Data",
            value=f"{stats['total_months']} months",
            delta=f"{stats['start_date'].year} - {stats['end_date'].year}"
        )
    
    with col2:
        st.metric(
            label="Commodities Tracked",
            value=stats['total_commodities'],
            delta="Rice, Cotton Yarn, Copper"
        )
    
    with col3:
        if metadata:
            val_mape = metadata.get('validation_set_metrics', {}).get('MAPE', 0)
            st.metric(
                label="Model Accuracy (Validation MAPE)",
                value=f"{val_mape:.2f}%",
                delta="Lower is better"
            )
        else:
            st.metric(label="Model Status", value="Active", delta="LightGBM")
    
    with col4:
        latest_forecast = get_latest_forecast()
        if latest_forecast is not None:
            latest_date = latest_forecast['Date'].max()
            st.metric(
                label="Latest Forecast Date",
                value=latest_date.strftime('%Y-%m'),
                delta="Available"
            )
        else:
            st.metric(label="Forecast Status", value="Not Generated", delta="Use Forecast Generator")
    
    st.markdown("---")
    
    # Quick Overview Chart
    st.subheader("Quick Overview - All Commodities")
    fig = create_time_series_plot(df_long, commodity='All', title="Export Values - All Commodities (2010-2025)")
    
    # Add latest forecast if available
    latest_forecast = get_latest_forecast()
    if latest_forecast is not None:
        fig = create_time_series_plot(df_long, commodity='All', 
                                     title="Export Values with Latest Forecast",
                                     show_forecast=True, forecast_df=latest_forecast)
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Latest Forecast Summary
    if latest_forecast is not None:
        st.subheader("Latest Forecast Summary")
        forecast_summary = latest_forecast.groupby('Commodity')['Forecast_Export_Value_USD'].agg([
            'mean', 'min', 'max', 'std'
        ]).round(0)
        forecast_summary.columns = ['Average', 'Minimum', 'Maximum', 'Std Dev']
        forecast_summary = forecast_summary.applymap(lambda x: f"${x:,.0f}")
        st.dataframe(forecast_summary, use_container_width=True)
        st.info(f"Forecast Period: {latest_forecast['Date'].min().strftime('%Y-%m-%d')} to {latest_forecast['Date'].max().strftime('%Y-%m-%d')}")

def show_historical_analysis_page(df_long, df_wide):
    """Historical Analysis Page"""
    st.title("Historical Analysis")
    st.markdown("---")
    
    # Commodity selector
    col1, col2 = st.columns([3, 1])
    with col1:
        commodity = st.selectbox("Select Commodity", ["All", "Rice", "Cotton Yarn", "Copper"])
    with col2:
        show_external = st.checkbox("Show External Drivers", False)
    
    # Get data
    if commodity == 'All':
        data = df_long
    else:
        data = get_commodity_data(df_long, commodity)
    
    # Time series plot
    fig = create_time_series_plot(data, commodity=commodity, 
                                  title=f"{commodity} - Historical Export Values")
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistics and Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Statistics Summary")
        stats = get_statistics(df_long, commodity)
        
        if commodity == 'All' and 'by_commodity' in stats:
            for comm, comm_stats in stats['by_commodity'].items():
                st.write(f"**{comm}:**")
                st.write(f"- Average: ${comm_stats['average']:,.0f}")
                st.write(f"- Min: ${comm_stats['min']:,.0f} | Max: ${comm_stats['max']:,.0f}")
        else:
            st.write(f"**Average:** ${stats['average_export_value']:,.0f}")
            st.write(f"**Median:** ${stats['median_export_value']:,.0f}")
            st.write(f"**Std Dev:** ${stats['std_export_value']:,.0f}")
            st.write(f"**Range:** ${stats['min_export_value']:,.0f} - ${stats['max_export_value']:,.0f}")
    
    with col2:
        st.subheader("Distribution Analysis")
        fig_dist = create_statistics_chart(df_long, commodity=commodity)
        st.plotly_chart(fig_dist, use_container_width=True)
    
    # Comparison view
    if commodity == 'All':
        st.markdown("---")
        st.subheader("Commodity Comparison")
        fig_comp = create_comparison_chart(df_long)
        st.plotly_chart(fig_comp, use_container_width=True)
        
        # Correlation heatmap
        if df_wide is not None:
            st.markdown("---")
            st.subheader("Correlation Analysis")
            fig_corr = create_correlation_heatmap(df_wide)
            st.plotly_chart(fig_corr, use_container_width=True)

def show_forecast_generator_page(df_long, model, feature_names):
    """Forecast Generator Page - Main Feature"""
    st.title("Forecast Generator")
    st.markdown("---")
    
    # Input panel
    with st.expander("Forecast Settings", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            forecast_months = st.slider("Forecast Horizon (months)", 1, 12, 6, 1)
            commodities = st.multiselect(
                "Select Commodities",
                ["Rice", "Cotton Yarn", "Copper"],
                default=["Rice", "Cotton Yarn", "Copper"],
                help="Select one or more commodities to forecast"
            )
        
        with col2:
            date_range = st.date_input(
                "Historical Context Date Range",
                value=[df_long['Date'].min(), df_long['Date'].max()],
                min_value=df_long['Date'].min(),
                max_value=df_long['Date'].max()
            )
            
            generate_btn = st.button("Generate Forecast", type="primary", use_container_width=True)
    
    # Generate forecast
    if generate_btn and model and feature_names:
        if not commodities:
            st.warning("Please select at least one commodity to forecast.")
        else:
            with st.spinner(f"Generating {forecast_months}-month forecast for {', '.join(commodities)}..."):
                try:
                    # Filter historical data by date range if specified
                    if isinstance(date_range, tuple) and len(date_range) == 2:
                        start_date, end_date = date_range
                        df_filtered = df_long[(df_long['Date'] >= pd.Timestamp(start_date)) & 
                                            (df_long['Date'] <= pd.Timestamp(end_date))].copy()
                    else:
                        df_filtered = df_long.copy()
                    
                    # Generate forecast
                    forecast_df, future_dates, future_forecasts = generate_forecast_for_dashboard(
                        df_filtered, model, feature_names, 
                        forecast_months=forecast_months,
                        selected_commodities=commodities if 'All' not in commodities else None
                    )
                    
                    if not forecast_df.empty:
                        # Calculate uncertainty bands
                        forecast_df = calculate_uncertainty_bands(forecast_df, df_filtered)
                        
                        st.success(f"Forecast generated successfully for {len(commodities)} commodity(ies)!")
                        
                        # Display forecasts
                        for commodity in commodities:
                            if commodity in forecast_df['Commodity'].values:
                                st.markdown(f"### {commodity} Forecast")
                                
                                # Plot with uncertainty bands
                                fig = create_time_series_plot(
                                    df_filtered[df_filtered['Commodity'] == commodity] if commodity != 'All' else df_filtered,
                                    commodity=commodity,
                                    title=f"{commodity} - {forecast_months}-Month Forecast with Uncertainty Bands",
                                    show_forecast=True,
                                    forecast_df=forecast_df[forecast_df['Commodity'] == commodity]
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Forecast details
                                comm_forecast = forecast_df[forecast_df['Commodity'] == commodity]
                                st.dataframe(comm_forecast[['Date', 'Forecast_Export_Value_USD', 'Lower_Bound', 'Upper_Bound']], 
                                           use_container_width=True, hide_index=True)
                        
                        # Download section
                        st.markdown("---")
                        st.subheader("Download Forecasts")
                        
                        csv = forecast_df.to_csv(index=False)
                        st.download_button(
                            label="Download Forecast as CSV",
                            data=csv,
                            file_name=f"forecast_{forecast_months}months_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                        
                except Exception as e:
                    st.error(f"Error generating forecast: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    
    elif not model or not feature_names:
        st.warning("Model or feature names not loaded. Please check model files.")

def show_model_performance_page(metadata, model):
    """Model Performance Page"""
    st.title("Model Performance")
    st.markdown("---")
    
    if metadata:
        # Model overview
        st.subheader("Model Overview")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info(f"**Model Name:** {metadata.get('model_name', 'N/A')}\n\n**Model Type:** {metadata.get('model_type', 'N/A')}")
        
        with col2:
            st.info(f"**Training Date:** {metadata.get('training_date', 'N/A')}\n\n**Status:** Active")
        
        with col3:
            commodities = metadata.get('commodities', [])
            st.info(f"**Commodities:** {', '.join(commodities)}\n\n**Data Range:** {metadata.get('data_range', {}).get('start', 'N/A')[:10]} to {metadata.get('data_range', {}).get('end', 'N/A')[:10]}")
        
        st.markdown("---")
        
        # Performance Metrics
        st.subheader("Performance Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Validation Set Performance:**")
            val_metrics = metadata.get('validation_set_metrics', {})
            
            st.metric("MAPE (Primary Metric)", f"{val_metrics.get('MAPE', 0):.2f}%", 
                     help="Mean Absolute Percentage Error - Lower is better")
            st.metric("RMSE", f"${val_metrics.get('RMSE', 0):,.0f}", 
                     help="Root Mean Squared Error")
            st.metric("MAE", f"${val_metrics.get('MAE', 0):,.0f}", 
                     help="Mean Absolute Error")
            st.metric("R² Score", f"{val_metrics.get('R2', 0):.4f}", 
                     help="Coefficient of Determination - Higher is better (max 1.0)")
        
        with col2:
            st.write("**Test Set Performance:**")
            test_metrics = metadata.get('test_set_metrics', {})
            
            st.metric("MAPE", f"{test_metrics.get('MAPE', 0):.2f}%")
            st.metric("RMSE", f"${test_metrics.get('RMSE', 0):,.0f}")
            st.metric("MAE", f"${test_metrics.get('MAE', 0):,.0f}")
            st.metric("R² Score", f"{test_metrics.get('R2', 0):.4f}")
        
        # Feature importance (if model is available)
        if model and hasattr(model, 'feature_importance'):
            st.markdown("---")
            st.subheader("Feature Importance")
            
            try:
                importance = model.feature_importance(importance_type='gain')
                feature_names = load_feature_names()
                
                if feature_names and len(importance) == len(feature_names):
                    importance_df = pd.DataFrame({
                        'feature': feature_names,
                        'importance': importance
                    }).sort_values('importance', ascending=False)
                    
                    top_n = st.slider("Top N Features", 10, 30, 20, 5)
                    
                    fig = create_feature_importance_chart(importance_df, top_n=top_n)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Feature importance table
                    with st.expander("View All Feature Importances"):
                        st.dataframe(importance_df, use_container_width=True, hide_index=True)
            except Exception as e:
                st.warning(f"Could not load feature importance: {str(e)}")
    else:
        st.warning("Model metadata not available. Please ensure model_metadata.json exists.")

def show_insights_analytics_page(df_long):
    """Insights & Analytics Page"""
    st.title("Insights & Analytics")
    st.markdown("---")
    
    commodity = st.selectbox("Select Commodity for Analysis", ["All", "Rice", "Cotton Yarn", "Copper"])
    
    data = get_commodity_data(df_long, commodity)
    
    # Key Insights
    st.subheader("Key Insights")
    
    # Calculate insights
    if commodity != 'All':
        latest_value = data['Export_Value_USD'].iloc[-1]
        avg_value = data['Export_Value_USD'].mean()
        trend = "Increasing" if latest_value > avg_value else "Decreasing" if latest_value < avg_value * 0.9 else "Stable"
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info(f"**Current Trend:** {trend}\n\nLatest: ${latest_value:,.0f}\nAverage: ${avg_value:,.0f}")
        
        with col2:
            pct_change = ((latest_value - avg_value) / avg_value * 100) if avg_value > 0 else 0
            st.info(f"**vs Historical Average:** {pct_change:+.1f}%\n\n{'Above' if pct_change > 0 else 'Below'} average")
        
        with col3:
            volatility = data['Export_Value_USD'].std() / data['Export_Value_USD'].mean() * 100 if data['Export_Value_USD'].mean() > 0 else 0
            risk_level = "High" if volatility > 30 else "Medium" if volatility > 15 else "Low"
            st.info(f"**Volatility:** {volatility:.1f}%\n\n**Risk Level:** {risk_level}")
    
    # Growth Analysis
    st.markdown("---")
    st.subheader("Growth Analysis")
    
    # Year-over-year analysis
    data['Year'] = pd.to_datetime(data['Date']).dt.year
    data['Month'] = pd.to_datetime(data['Date']).dt.month
    
    if commodity != 'All':
        yearly_data = data.groupby('Year')['Export_Value_USD'].mean().reset_index()
        yearly_data['YoY_Change'] = yearly_data['Export_Value_USD'].pct_change() * 100
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Year-over-Year Growth:**")
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=yearly_data['Year'],
                y=yearly_data['YoY_Change'],
                marker_color=['green' if x > 0 else 'red' for x in yearly_data['YoY_Change']],
                text=[f"{x:.1f}%" for x in yearly_data['YoY_Change']],
                textposition='auto'
            ))
            fig.update_layout(
                title="Year-over-Year Growth Rate (%)",
                xaxis_title="Year",
                yaxis_title="Growth Rate (%)",
                template='plotly_white',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("**Monthly Patterns:**")
            monthly_avg = data.groupby('Month')['Export_Value_USD'].mean()
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=[pd.Timestamp(2020, m, 1).strftime('%b') for m in monthly_avg.index],
                y=monthly_avg.values,
                marker_color='#2E86AB'
            ))
            fig.update_layout(
                title="Average Export Value by Month",
                xaxis_title="Month",
                yaxis_title="Average Export Value (USD)",
                template='plotly_white',
                height=400,
                yaxis_tickformat="$,.0f"
            )
            st.plotly_chart(fig, use_container_width=True)

def show_external_drivers_page(df_wide):
    """External Drivers Page"""
    st.title("External Drivers Analysis")
    st.markdown("---")
    
    if df_wide is None:
        st.error("Wide format data not available.")
        return
    
    # External drivers overview
    st.subheader("External Drivers Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        latest_usd = df_wide['USD_PKR'].iloc[-1]
        avg_usd = df_wide['USD_PKR'].mean()
        st.metric("USD/PKR Exchange Rate", f"{latest_usd:.2f}", f"{((latest_usd-avg_usd)/avg_usd*100):+.1f}% vs avg")
    
    with col2:
        latest_oil = df_wide['Oil_Price'].iloc[-1]
        avg_oil = df_wide['Oil_Price'].mean()
        st.metric("Brent Oil Price", f"${latest_oil:.2f}", f"{((latest_oil-avg_oil)/avg_oil*100):+.1f}% vs avg")
    
    with col3:
        latest_conf = df_wide['US_Confidence'].iloc[-1]
        avg_conf = df_wide['US_Confidence'].mean()
        st.metric("US Consumer Confidence", f"{latest_conf:.1f}", f"{((latest_conf-avg_conf)/avg_conf*100):+.1f}% vs avg")
    
    st.markdown("---")
    
    # External drivers time series
    st.subheader("External Drivers Trends")
    
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=("USD/PKR Exchange Rate", "Brent Oil Price", "US Consumer Confidence"),
        vertical_spacing=0.1
    )
    
    fig.add_trace(
        go.Scatter(x=df_wide['Date'], y=df_wide['USD_PKR'], name="USD/PKR", line=dict(color='#2E86AB')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df_wide['Date'], y=df_wide['Oil_Price'], name="Oil Price", line=dict(color='#F18F01')),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df_wide['Date'], y=df_wide['US_Confidence'], name="US Confidence", line=dict(color='#06A77D')),
        row=3, col=1
    )
    
    fig.update_layout(height=800, template='plotly_white', showlegend=False)
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_yaxes(title_text="Rate", row=1, col=1)
    fig.update_yaxes(title_text="Price (USD)", row=2, col=1, tickformat="$,.2f")
    fig.update_yaxes(title_text="Index", row=3, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation analysis
    st.markdown("---")
    st.subheader("Correlation with Commodities")
    
    fig_corr = create_correlation_heatmap(df_wide)
    st.plotly_chart(fig_corr, use_container_width=True)

# Main execution - Load data and route to pages
df_long, df_wide, model, feature_names, metadata = load_all_data()

if df_long is None:
    st.error("Error: Could not load data. Please check if all files are in the correct locations.")
    st.stop()

# Page content based on selection
if page == "Overview":
    show_overview_page(df_long, metadata)
elif page == "Historical Analysis":
    show_historical_analysis_page(df_long, df_wide)
elif page == "Forecast Generator":
    show_forecast_generator_page(df_long, model, feature_names)
elif page == "Model Performance":
    show_model_performance_page(metadata, model)
elif page == "Insights & Analytics":
    show_insights_analytics_page(df_long)
elif page == "External Drivers":
    show_external_drivers_page(df_wide)

