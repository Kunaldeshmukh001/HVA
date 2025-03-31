import pandas as pd
import numpy as np
from datetime import datetime
import os

def load_demo_data(filename):
    """
    Load demo dataset from the data directory
    
    Parameters:
    -----------
    filename : str
        Name of the file to load
    
    Returns:
    --------
    pd.DataFrame
        The loaded demo data
    """
    filepath = os.path.join("data", filename)
    try:
        # If running in production with proper file structure
        data = pd.read_csv(filepath)
    except FileNotFoundError:
        # For demonstration, generate demo data if file doesn't exist
        # This would be replaced by actual demo files in production
        if 'auto' in filename.lower():
            data = generate_auto_industry_demo()
        else:
            data = generate_cpg_industry_demo()
    
    return data

def generate_auto_industry_demo():
    """Generate sample auto industry data for demonstration purposes"""
    # Date range - 18 months weekly data
    date_range = pd.date_range(start='2022-01-01', periods=78, freq='W')
    
    # Base data
    data = pd.DataFrame({'Date': date_range})
    
    # Seasonal pattern (higher in summer, lower in winter)
    seasonal = np.sin(np.linspace(0, 4*np.pi, len(date_range)))
    
    # Different patterns for variability
    trend1 = np.linspace(0, 20, len(date_range))  # Increasing trend
    cycle1 = np.cos(np.linspace(0, 6*np.pi, len(date_range)))  # Different cycle
    
    # Conversion metrics with some realistic names and varied patterns
    np.random.seed(42)  # For reproducibility
    
    # More variability in each metric to create different correlation ranges
    data['TestDriveBookings'] = 100 + 30*seasonal + 5*trend1 + np.random.normal(0, 15, len(date_range))
    data['VehicleConfigurations'] = 300 + 80*seasonal + 15*cycle1 + np.random.normal(0, 30, len(date_range))
    data['BrochureDownloads'] = 500 + 100*seasonal - 10*trend1 + np.random.normal(0, 50, len(date_range))
    data['DealerLocatorUses'] = 250 + 60*cycle1 + 8*trend1 + np.random.normal(0, 25, len(date_range))
    data['PriceRequestsSubmitted'] = 150 + 45*seasonal*cycle1 + np.random.normal(0, 20, len(date_range))
    data['FinanceCalculatorUses'] = 200 + 50*seasonal + 12*trend1*seasonal + np.random.normal(0, 22, len(date_range))
    data['SpecialOffersViews'] = 400 + 90*cycle1 - 5*trend1 + np.random.normal(0, 40, len(date_range))
    data['VideoViews'] = 600 + 120*seasonal + 10*np.sin(trend1) + np.random.normal(0, 60, len(date_range))
    
    # Target variable - vehicle sales with lag effect
    base_sales = 50 + 15*seasonal + 3*trend1
    
    # Add influence from conversions with different lags and strengths
    test_drive_effect = 0.2 * data['TestDriveBookings'].shift(2).fillna(0)
    config_effect = 0.05 * data['VehicleConfigurations'].shift(1).fillna(0)
    brochure_effect = 0.01 * data['BrochureDownloads'].shift(3).fillna(0)
    price_effect = 0.1 * data['PriceRequestsSubmitted'].shift(1).fillna(0)
    
    data['VehicleSales'] = base_sales + test_drive_effect + config_effect + brochure_effect + price_effect + np.random.normal(0, 5, len(date_range))
    
    # Ensure no negative values and convert to float (not int) to prevent dtype warnings
    for col in data.columns:
        if col != 'Date':
            data[col] = data[col].apply(lambda x: max(0, x))
            data[col] = data[col].round().astype(float)
    
    return data

def generate_cpg_industry_demo():
    """Generate sample CPG industry data for demonstration purposes"""
    # Date range - 18 months weekly data
    date_range = pd.date_range(start='2022-01-01', periods=78, freq='W')
    
    # Base data
    data = pd.DataFrame({'Date': date_range})
    
    # Seasonal pattern 
    seasonal = np.sin(np.linspace(0, 4*np.pi, len(date_range)))
    
    # Additional patterns for variability
    trend = np.linspace(0, 30, len(date_range))
    cycle2 = np.sin(np.linspace(0, 8*np.pi, len(date_range)))  # Different frequency
    shock = np.zeros(len(date_range))
    shock[30:40] = 5  # Temporary shock period
    
    # Conversion metrics with some realistic names and varied patterns
    np.random.seed(42)  # For reproducibility
    
    data['ProductPageViews'] = 5000 + 1000*seasonal + 20*trend + 500*cycle2 + np.random.normal(0, 200, len(date_range))
    data['AddToCart'] = 800 + 200*seasonal + 10*trend - 100*cycle2 + 50*shock + np.random.normal(0, 60, len(date_range)) 
    data['CheckoutStarts'] = 400 + 100*seasonal + 5*trend + 30*shock + np.random.normal(0, 35, len(date_range))
    data['PromotionClicks'] = 600 + 150*seasonal*cycle2 + 8*trend + np.random.normal(0, 45, len(date_range))
    data['RecipePageViews'] = 1200 + 300*cycle2 + 15*trend + np.random.normal(0, 90, len(date_range))
    data['StoreLocatorUses'] = 300 + 80*seasonal + 4*trend - 20*cycle2 + np.random.normal(0, 30, len(date_range))
    data['EmailSignups'] = 200 + 50*seasonal - 3*trend*seasonal + np.random.normal(0, 25, len(date_range))
    data['ProductReviewReads'] = 900 + 220*cycle2 + 12*trend + 30*shock + np.random.normal(0, 70, len(date_range))
    
    # Target variable - product sales with lag effect
    base_sales = 200 + 50*seasonal + 5*trend + 20*cycle2
    
    # Add influence from conversions with different lags and strengths
    product_view_effect = 0.01 * data['ProductPageViews'].shift(1).fillna(0)
    cart_effect = 0.2 * data['AddToCart'].shift(1).fillna(0)
    checkout_effect = 0.3 * data['CheckoutStarts'].shift(1).fillna(0)
    promo_effect = 0.05 * data['PromotionClicks'].shift(2).fillna(0)
    
    data['ProductSales'] = base_sales + product_view_effect + cart_effect + checkout_effect + promo_effect + np.random.normal(0, 15, len(date_range))
    
    # Ensure no negative values and convert to float (not int) to prevent dtype warnings
    for col in data.columns:
        if col != 'Date':
            data[col] = data[col].apply(lambda x: max(0, x))
            data[col] = data[col].round().astype(float)
    
    return data

def preprocess_data(data, date_column, target_variable, features):
    """
    Preprocess the data for analysis
    
    Parameters:
    -----------
    data : pd.DataFrame
        The input data
    date_column : str
        Name of the date column
    target_variable : str
        Name of the target variable
    features : list
        List of feature column names
    
    Returns:
    --------
    pd.DataFrame
        Preprocessed data
    """
    # Create a copy to avoid modifying original data
    df = data.copy()
    
    # Convert date column to datetime
    if df[date_column].dtype != 'datetime64[ns]':
        df[date_column] = pd.to_datetime(df[date_column])
    
    # Sort by date
    df = df.sort_values(by=date_column)
    
    # Select only the columns we need
    cols_to_keep = [date_column, target_variable] + [f for f in features]
    df = df[cols_to_keep]
    
    # Fill missing values (if any)
    for col in df.columns:
        if col != date_column:
            # Forward fill then backward fill to handle gaps
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
            
            # If still missing values, replace with zeros
            df[col] = df[col].fillna(0)
    
    return df

def apply_adstock(series, carryover):
    """
    Apply adstock transformation to a series with a given carryover rate
    
    Parameters:
    -----------
    series : pd.Series
        The input series
    carryover : float
        The carryover rate (between 0 and 1)
    
    Returns:
    --------
    pd.Series
        Transformed series
    """
    result = series.copy()
    for i in range(1, len(series)):
        result.iloc[i] = series.iloc[i] + carryover * result.iloc[i-1]
    
    # Ensure that the result has the correct data type (float)
    # This prevents FutureWarning about incompatible dtype
    result = result.astype(float)
    return result

def apply_adstock_transformations(df, features, target, max_lag=2, carryover_range=np.arange(0.1, 1.0, 0.1)):
    """
    Apply adstock transformations with different carryover rates,
    and select the best transformation for each feature based on correlation with target
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input data
    features : list
        List of features to transform
    target : str
        Target variable name
    max_lag : int
        Maximum lag to consider
    carryover_range : array-like
        Range of carryover values to test
    
    Returns:
    --------
    tuple
        (transformed_df, best_params_dict)
    """
    # Create a copy of the DataFrame
    result_df = df.copy()
    
    # Dictionary to store best parameters for each feature
    best_params = {}
    
    for feature in features:
        best_correlation = 0
        best_carryover = 0
        best_column = None
        
        for carryover in carryover_range:
            # Create transformed feature name
            transformed_name = f"{feature}_adstock_{carryover:.2f}"
            
            # Apply adstock transformation
            result_df[transformed_name] = apply_adstock(df[feature], carryover)
            
            # Calculate correlation with target
            correlation = abs(result_df[transformed_name].corr(result_df[target]))
            
            # Update best parameters if this is better
            if correlation > best_correlation:
                best_correlation = correlation
                best_carryover = carryover
                best_column = transformed_name
        
        # Keep only the best transformation
        if best_column:
            # Rename to simpler name for clarity
            result_df[feature] = result_df[best_column]
            
            # Store best parameters
            best_params[feature] = {
                'carryover': best_carryover,
                'correlation': best_correlation
            }
            
            # Drop intermediate columns
            for carryover in carryover_range:
                col_name = f"{feature}_adstock_{carryover:.2f}"
                if col_name in result_df.columns and col_name != best_column:
                    result_df = result_df.drop(columns=[col_name])
        
        # Drop the best column name too (since we renamed it)
        if best_column in result_df.columns:
            result_df = result_df.drop(columns=[best_column])
    
    return result_df, best_params
