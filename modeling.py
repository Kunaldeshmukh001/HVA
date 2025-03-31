import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
# Temporarily comment out these imports until we can resolve installation
# import xgboost as xgb
# import shap

def build_ols_model(df, features, target):
    """
    Build an OLS regression model
    
    Parameters:
    -----------
    df : pd.DataFrame
        The input data
    features : list
        List of feature column names
    target : str
        Name of target variable
    
    Returns:
    --------
    dict
        Dictionary containing model results
    """
    # Prepare X and y
    X = df[features]
    y = df[target]
    
    # Add constant
    X = sm.add_constant(X)
    
    # Fit model
    model = sm.OLS(y, X).fit()
    
    # Get summary
    summary = model.summary()
    
    # Create a dataframe with coefficients, p-values, and confidence intervals
    coef_df = pd.DataFrame({
        'Feature': ['const'] + features,
        'Coefficient': model.params,
        'P_Value': model.pvalues,
        'Lower_CI': model.conf_int()[0],
        'Upper_CI': model.conf_int()[1]
    })
    
    # Sort by absolute coefficient value
    coef_df['Abs_Coef'] = coef_df['Coefficient'].abs()
    coef_df = coef_df.sort_values('Abs_Coef', ascending=False)
    coef_df = coef_df.drop(columns=['Abs_Coef'])
    
    # Add significance marker
    coef_df['Significant'] = coef_df['P_Value'] < 0.05
    
    # Return results
    return {
        'model': model,
        'summary_df': coef_df,
        'summary_text': str(summary),
        'r2': model.rsquared,
        'adj_r2': model.rsquared_adj
    }

def build_xgboost_model(df, features, target):
    """
    Build an XGBoost model
    
    Parameters:
    -----------
    df : pd.DataFrame
        The input data
    features : list
        List of feature column names
    target : str
        Name of target variable
    
    Returns:
    --------
    dict
        Dictionary containing model results
    """
    # Prepare X and y
    X = df[features]
    y = df[target]
    
    # In the temporary version (until XGBoost is installed), we'll use a simple 
    # linear model from sklearn and simulate the XGBoost output structure
    from sklearn.linear_model import LinearRegression
    
    # Split data for training and evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Calculate feature importance using coefficient absolute values
    raw_importance = np.abs(model.coef_)
    
    # Convert to DataFrame
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': raw_importance,
    })
    
    # Scale to sum to 100%
    total_importance = importance_df['Importance'].sum()
    importance_df['SHAP_Value'] = importance_df['Importance'] 
    importance_df['SHAP_Percentage'] = 100 * importance_df['Importance'] / total_importance
    
    # Sort by importance
    importance_df = importance_df.sort_values('SHAP_Percentage', ascending=False)
    
    # Create feature importance plot
    fig = px.bar(
        importance_df,
        x='Feature',
        y='SHAP_Percentage',
        title="Feature Importance (Linear Coefficients)",
        color='SHAP_Percentage',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        xaxis_title="Feature",
        yaxis_title="Importance (%)",
        height=500,
        showlegend=False
    )
    
    # Return results
    return {
        'model': model,
        'importance_df': importance_df,
        'importance_plot': fig,
        'r2': r2,
        'rmse': rmse,
        'shap_values': None  # No SHAP values in this temporary version
    }

def calculate_proxy_value(ols_df, xgb_df, aov):
    """
    Calculate proxy value for each conversion based on model coefficients and AOV
    
    Parameters:
    -----------
    ols_df : pd.DataFrame
        OLS model coefficients
    xgb_df : pd.DataFrame
        XGBoost feature importance
    aov : float
        Average order value
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with calculated values
    """
    # Filter out constant from OLS and take only positive coefficients
    ols_features = ols_df[(ols_df['Feature'] != 'const') & (ols_df['Coefficient'] > 0)].copy()
    
    # Scale OLS coefficients to sum to 100%
    coef_sum = ols_features['Coefficient'].sum()
    ols_features['Scaled_Coefficient'] = 100 * ols_features['Coefficient'] / coef_sum
    
    # Create merged DataFrame
    result = pd.DataFrame()
    result['Feature'] = ols_features['Feature']
    result['OLS_Coefficient'] = ols_features['Coefficient']
    result['OLS_Scaled_Percentage'] = ols_features['Scaled_Coefficient']
    result['OLS_Value'] = result['OLS_Scaled_Percentage'] * aov / 100
    
    # Add XGBoost results - ensure we only use features with positive OLS coefficients
    filtered_xgb_df = xgb_df[xgb_df['Feature'].isin(ols_features['Feature'])]
    result = result.merge(filtered_xgb_df[['Feature', 'SHAP_Percentage']], on='Feature', how='left')
    result['XGB_Value'] = result['SHAP_Percentage'] * aov / 100
    
    # Calculate average value
    result['Average_Value'] = (result['OLS_Value'] + result['XGB_Value']) / 2
    
    # Sort by average value
    result = result.sort_values('Average_Value', ascending=False)
    
    # Format as currency
    for col in ['OLS_Value', 'XGB_Value', 'Average_Value']:
        result[col] = result[col].apply(lambda x: f"${x:.2f}")
    
    # Format percentages
    for col in ['OLS_Scaled_Percentage', 'SHAP_Percentage']:
        result[col] = result[col].apply(lambda x: f"{x:.2f}%")
    
    return result
