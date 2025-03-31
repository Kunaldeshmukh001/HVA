import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

def plot_time_series(df, date_column, features, target):
    """
    Create a time series plot for all features and target variable
    
    Parameters:
    -----------
    df : pd.DataFrame
        The input data
    date_column : str
        Name of the date column
    features : list
        List of feature column names
    target : str
        Name of target variable
    
    Returns:
    --------
    plotly.graph_objects.Figure
        The time series plot
    """
    # Create subplots: one for features, one for target
    fig = make_subplots(rows=2, cols=1, 
                        shared_xaxes=True,
                        subplot_titles=("Conversion Metrics", f"Target: {target}"),
                        row_heights=[0.7, 0.3],
                        vertical_spacing=0.1)
    
    # Add traces for features
    for feature in features:
        fig.add_trace(
            go.Scatter(x=df[date_column], y=df[feature], name=feature),
            row=1, col=1
        )
    
    # Add trace for target variable
    fig.add_trace(
        go.Scatter(x=df[date_column], y=df[target], name=target, line=dict(width=3, color='black')),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=700,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Conversion Count", row=1, col=1)
    fig.update_yaxes(title_text=target, row=2, col=1)
    
    return fig

def plot_collinearity(df, features):
    """
    Create a heatmap of correlations between features
    
    Parameters:
    -----------
    df : pd.DataFrame
        The input data
    features : list
        List of feature column names
    
    Returns:
    --------
    plotly.graph_objects.Figure
        The correlation heatmap
    """
    # Calculate correlation matrix
    corr_matrix = df[features].corr()
    
    # Create heatmap
    fig = px.imshow(
        corr_matrix,
        text_auto='.2f',
        color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1,
        aspect="auto",
        title="Feature Correlation Matrix"
    )
    
    fig.update_layout(
        height=600,
        width=700,
    )
    
    return fig

def calculate_vif(df, features):
    """
    Calculate Variance Inflation Factor for each feature
    
    Parameters:
    -----------
    df : pd.DataFrame
        The input data
    features : list
        List of feature column names
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with VIF values for each feature
    """
    # Create a dataframe with just the features
    X = df[features].copy()
    
    # Calculate VIF for each feature
    vif_data = pd.DataFrame()
    vif_data["Feature"] = features
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    
    # Sort by VIF
    vif_data = vif_data.sort_values("VIF", ascending=False)
    
    return vif_data

def plot_correlations(df, target, features):
    """
    Create a bar chart of correlations between features and target
    
    Parameters:
    -----------
    df : pd.DataFrame
        The input data
    target : str
        Name of target variable
    features : list
        List of feature column names
    
    Returns:
    --------
    plotly.graph_objects.Figure
        The correlation bar chart
    """
    # Calculate correlations
    correlations = []
    for feature in features:
        corr = df[feature].corr(df[target])
        correlations.append({
            'Feature': feature,
            'Correlation': corr,
            'Abs_Correlation': abs(corr)
        })
    
    # Create dataframe and sort
    corr_df = pd.DataFrame(correlations)
    corr_df = corr_df.sort_values('Abs_Correlation', ascending=False)
    
    # Create color map - blue for positive, red for negative
    colors = ['blue' if c >= 0 else 'red' for c in corr_df['Correlation']]
    
    # Create bar chart
    fig = px.bar(
        corr_df,
        x='Feature',
        y='Correlation',
        color='Correlation',
        color_continuous_scale='RdBu',
        title=f"Correlation with {target}"
    )
    
    fig.update_layout(
        xaxis_title="Feature",
        yaxis_title="Correlation Coefficient",
        yaxis_range=[-1, 1],
        height=500,
    )
    
    return fig

def plot_volumes(df, features):
    """
    Create a bar chart showing the volume (sum) of each conversion metric
    
    Parameters:
    -----------
    df : pd.DataFrame
        The input data
    features : list
        List of feature column names
    
    Returns:
    --------
    plotly.graph_objects.Figure
        The volume bar chart
    """
    # Calculate sums
    volumes = []
    for feature in features:
        volumes.append({
            'Feature': feature,
            'Total Volume': df[feature].sum()
        })
    
    # Create dataframe and sort
    vol_df = pd.DataFrame(volumes)
    vol_df = vol_df.sort_values('Total Volume', ascending=False)
    
    # Create bar chart
    fig = px.bar(
        vol_df,
        x='Feature',
        y='Total Volume',
        title="Total Volume by Conversion Metric",
        color='Feature'
    )
    
    fig.update_layout(
        xaxis_title="Conversion Metric",
        yaxis_title="Total Volume",
        height=500,
        showlegend=False
    )
    
    return fig
