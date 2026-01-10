"""
Visualization Utilities for Streamlit Dashboard
Creates interactive Plotly charts
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def create_time_series_plot(df, commodity=None, title="Export Value Over Time", show_forecast=False, forecast_df=None):
    """Create interactive time series plot"""
    fig = go.Figure()
    
    if commodity and commodity != 'All':
        data = df[df['Commodity'] == commodity]
        colors = {'Rice': '#2E86AB', 'Cotton Yarn': '#A23B72', 'Copper': '#F18F01'}
        color = colors.get(commodity, '#2E86AB')
        
        fig.add_trace(go.Scatter(
            x=data['Date'],
            y=data['Export_Value_USD'],
            mode='lines',
            name=f'{commodity} - Historical',
            line=dict(color=color, width=2),
            hovertemplate='<b>%{fullData.name}</b><br>' +
                         'Date: %{x}<br>' +
                         'Value: $%{y:,.0f}<br>' +
                         '<extra></extra>'
        ))
        
        # Add forecast if provided
        if show_forecast and forecast_df is not None:
            forecast_data = forecast_df[forecast_df['Commodity'] == commodity]
            if not forecast_data.empty:
                # Add uncertainty bands
                if 'Lower_Bound' in forecast_data.columns and 'Upper_Bound' in forecast_data.columns:
                    # Convert hex color to RGB for rgba
                    hex_color = color.lstrip('#')
                    try:
                        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                        fillcolor_rgba = f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.2)'
                    except:
                        fillcolor_rgba = 'rgba(255, 107, 107, 0.2)'  # Light red as fallback
                    
                    fig.add_trace(go.Scatter(
                        x=pd.concat([forecast_data['Date'], forecast_data['Date'][::-1]]),
                        y=pd.concat([forecast_data['Upper_Bound'], forecast_data['Lower_Bound'][::-1]]),
                        fill='toself',
                        fillcolor=fillcolor_rgba,
                        line=dict(color='rgba(255,255,255,0)'),
                        hoverinfo="skip",
                        showlegend=True,
                        name='Uncertainty Band (80% confidence)'
                    ))
                
                # Add forecast line
                fig.add_trace(go.Scatter(
                    x=forecast_data['Date'],
                    y=forecast_data['Forecast_Export_Value_USD'],
                    mode='lines+markers',
                    name=f'{commodity} - Forecast',
                    line=dict(color='#DC143C', width=2.5, dash='dash'),
                    marker=dict(size=8, symbol='circle'),
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                 'Date: %{x}<br>' +
                                 'Forecast: $%{y:,.0f}<br>' +
                                 '<extra></extra>'
                ))
    else:
        # Plot all commodities
        colors = {'Rice': '#2E86AB', 'Cotton Yarn': '#A23B72', 'Copper': '#F18F01'}
        for comm in df['Commodity'].unique():
            data = df[df['Commodity'] == comm]
            fig.add_trace(go.Scatter(
                x=data['Date'],
                y=data['Export_Value_USD'],
                mode='lines',
                name=f'{comm} - Historical',
                line=dict(color=colors.get(comm, '#2E86AB'), width=2),
                hovertemplate='<b>%{fullData.name}</b><br>' +
                             'Date: %{x}<br>' +
                             'Value: $%{y:,.0f}<br>' +
                             '<extra></extra>'
            ))
            
            # Add forecast if provided
            if show_forecast and forecast_df is not None:
                forecast_data = forecast_df[forecast_df['Commodity'] == comm]
                if not forecast_data.empty:
                    fig.add_trace(go.Scatter(
                        x=forecast_data['Date'],
                        y=forecast_data['Forecast_Export_Value_USD'],
                        mode='lines+markers',
                        name=f'{comm} - Forecast',
                        line=dict(color='#DC143C', width=2.5, dash='dash'),
                        marker=dict(size=6, symbol='circle'),
                        hovertemplate='<b>%{fullData.name}</b><br>' +
                                     'Date: %{x}<br>' +
                                     'Forecast: $%{y:,.0f}<br>' +
                                     '<extra></extra>'
                    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, family="Arial Black")),
        xaxis_title="Date",
        yaxis_title="Export Value (USD)",
        yaxis=dict(tickformat="$,.0f"),
        hovermode='x unified',
        template='plotly_white',
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def create_comparison_chart(df, commodities=None):
    """Create side-by-side comparison chart for multiple commodities"""
    if commodities is None:
        commodities = df['Commodity'].unique()
    
    fig = make_subplots(
        rows=len(commodities), cols=1,
        subplot_titles=[f'{comm} Export Value' for comm in commodities],
        vertical_spacing=0.08
    )
    
    colors = {'Rice': '#2E86AB', 'Cotton Yarn': '#A23B72', 'Copper': '#F18F01'}
    
    for idx, comm in enumerate(commodities, 1):
        data = df[df['Commodity'] == comm]
        fig.add_trace(
            go.Scatter(
                x=data['Date'],
                y=data['Export_Value_USD'],
                mode='lines',
                name=comm,
                line=dict(color=colors.get(comm, '#2E86AB'), width=2),
                showlegend=False
            ),
            row=idx, col=1
        )
    
    fig.update_layout(
        height=300 * len(commodities),
        template='plotly_white',
        title_text="Commodity Comparison",
        title_x=0.5
    )
    
    for i in range(len(commodities)):
        fig.update_yaxes(title_text="Export Value (USD)", row=i+1, col=1, tickformat="$,.0f")
        if i == len(commodities) - 1:
            fig.update_xaxes(title_text="Date", row=i+1, col=1)
    
    return fig

def create_correlation_heatmap(df_wide):
    """Create correlation heatmap between commodities and external drivers"""
    # Select relevant columns
    cols = ['Rice_Export_Value_USD', 'Cotton_Yarn_Export_Value_USD', 'Copper_Export_Value_USD',
            'USD_PKR', 'Oil_Price', 'US_Confidence']
    
    # Use only columns that exist
    available_cols = [col for col in cols if col in df_wide.columns]
    corr_df = df_wide[available_cols].corr()
    
    # Rename columns for better display
    corr_df.columns = [col.replace('_Export_Value_USD', '').replace('_', ' ') for col in corr_df.columns]
    corr_df.index = [col.replace('_Export_Value_USD', '').replace('_', ' ') for col in corr_df.index]
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_df.values,
        x=corr_df.columns,
        y=corr_df.index,
        colorscale='RdBu',
        zmid=0,
        text=corr_df.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 12},
        hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Correlation Heatmap: Commodities vs External Drivers",
        height=500,
        template='plotly_white'
    )
    
    return fig

def create_statistics_chart(df, commodity=None):
    """Create statistics visualization (box plot, histogram)"""
    if commodity and commodity != 'All':
        data = df[df['Commodity'] == commodity]['Export_Value_USD']
        title = f'{commodity} Export Value Distribution'
    else:
        data = df['Export_Value_USD']
        title = 'All Commodities Export Value Distribution'
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Distribution (Histogram)', 'Box Plot'),
        horizontal_spacing=0.15
    )
    
    # Histogram
    fig.add_trace(
        go.Histogram(
            x=data,
            nbinsx=30,
            name='Distribution',
            marker_color='#2E86AB'
        ),
        row=1, col=1
    )
    
    # Box plot
    fig.add_trace(
        go.Box(
            y=data,
            name='Box Plot',
            marker_color='#A23B72'
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title=title,
        height=400,
        template='plotly_white',
        showlegend=False
    )
    
    fig.update_xaxes(title_text="Export Value (USD)", row=1, col=1, tickformat="$,.0f")
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_yaxes(title_text="Export Value (USD)", row=1, col=2, tickformat="$,.0f")
    
    return fig

def create_feature_importance_chart(importance_data, top_n=20):
    """Create feature importance bar chart"""
    # importance_data should be a dict or DataFrame with 'feature' and 'importance' columns
    if isinstance(importance_data, dict):
        df = pd.DataFrame(list(importance_data.items()), columns=['feature', 'importance'])
    else:
        df = importance_data.copy()
    
    df = df.nlargest(top_n, 'importance').sort_values('importance', ascending=True)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df['importance'],
        y=df['feature'],
        orientation='h',
        marker=dict(color='#2E86AB'),
        hovertemplate='<b>%{y}</b><br>Importance: %{x:,.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f"Top {top_n} Feature Importance",
        xaxis_title="Importance (Gain)",
        yaxis_title="Feature",
        height=600,
        template='plotly_white'
    )
    
    return fig
