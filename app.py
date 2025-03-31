import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from io import StringIO

from data_processing import load_demo_data, preprocess_data, apply_adstock_transformations
from eda import plot_time_series, plot_collinearity, calculate_vif, plot_correlations, plot_volumes
from modeling import build_ols_model, build_xgboost_model, calculate_proxy_value

st.set_page_config(
    page_title="High Value Action Modeling",
    page_icon="📊",
    layout="wide"
)

st.title("High Value Action Modeling")
st.markdown("""
This application analyzes marketing conversion data to identify high-value actions through adstock modeling, 
statistical checks, and machine learning techniques. Upload your conversion data to get started.
""")

# Session state management
if 'data' not in st.session_state:
    st.session_state.data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'adstock_data' not in st.session_state:
    st.session_state.adstock_data = None
if 'removed_features' not in st.session_state:
    st.session_state.removed_features = []
if 'target_variable' not in st.session_state:
    st.session_state.target_variable = None
if 'date_column' not in st.session_state:
    st.session_state.date_column = None
if 'features' not in st.session_state:
    st.session_state.features = []
if 'ols_results' not in st.session_state:
    st.session_state.ols_results = None
if 'xgb_results' not in st.session_state:
    st.session_state.xgb_results = None
if 'best_adstock_params' not in st.session_state:
    st.session_state.best_adstock_params = {}

# Sidebar for data input and configuration
with st.sidebar:
    st.header("Data Input")
    
    # AOV input
    aov = st.number_input("Average Order Value ($)", min_value=1, value=100)
    
    # Demo data selection
    demo_option = st.selectbox(
        "Select a demo dataset",
        ["None", "Auto Industry", "CPG Industry"]
    )
    
    if demo_option != "None":
        if st.button("Load Demo Data"):
            demo_file = "auto_industry.csv" if demo_option == "Auto Industry" else "cpg_industry.csv"
            st.session_state.data = load_demo_data(demo_file)
            st.success(f"Demo data loaded: {demo_option}")
            st.session_state.removed_features = []
            
    # Upload own data
    uploaded_file = st.file_uploader("Or upload your own CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            st.session_state.data = pd.read_csv(uploaded_file)
            st.success("Data successfully uploaded")
            st.session_state.removed_features = []
        except Exception as e:
            st.error(f"Error uploading file: {e}")

    # Display data info if data is loaded
    if st.session_state.data is not None:
        st.write(f"Data shape: {st.session_state.data.shape}")
        
        # Configuration options appear only after data is loaded
        st.header("Configuration")
        
        # Select date column
        date_cols = [col for col in st.session_state.data.columns if 'date' in col.lower() or 'time' in col.lower()]
        if not date_cols:
            date_cols = list(st.session_state.data.columns)
        
        st.session_state.date_column = st.selectbox(
            "Select date column",
            date_cols,
            index=0 if date_cols else None
        )
        
        # Select target variable (exclude date column)
        target_options = [col for col in st.session_state.data.columns if col != st.session_state.date_column]
        st.session_state.target_variable = st.selectbox(
            "Select target variable",
            target_options,
            index=0 if target_options else None
        )
        
        # Get conversion features (exclude date and target)
        st.session_state.features = [col for col in st.session_state.data.columns 
                                    if col != st.session_state.date_column 
                                    and col != st.session_state.target_variable]
        
        # Adstock configuration
        st.subheader("Adstock Configuration")
        
        data_granularity = st.radio(
            "Data granularity",
            ["Daily", "Weekly"]
        )
        
        max_lag_weeks = st.number_input(
            "Max lag (weeks)" if data_granularity == "Weekly" else "Max lag (days)",
            min_value=1,
            max_value=12,
            value=2 if data_granularity == "Weekly" else 7
        )
        
        carryover_min = 0.0  # Start from 0
        carryover_max = 0.9  # End at 0.9
        carryover_step = 0.1  # Step by 0.1

# Main content area
if st.session_state.data is not None:
    # Data preprocessing tab
    tab1, tab2, tab3, tab4 = st.tabs(["Data Overview", "Exploratory Analysis", "Adstock Modeling", "Value Analysis"])
    
    with tab1:
        st.header("Data Overview")
        
        st.dataframe(st.session_state.data.head(10))
        
        if st.button("Preprocess Data"):
            with st.spinner("Preprocessing data..."):
                st.session_state.processed_data = preprocess_data(
                    st.session_state.data, 
                    st.session_state.date_column, 
                    st.session_state.target_variable,
                    st.session_state.features
                )
                st.success("Data preprocessed successfully")
        
        if st.session_state.processed_data is not None:
            st.subheader("Preprocessed Data")
            st.dataframe(st.session_state.processed_data.head(10))
    
    with tab2:
        st.header("Exploratory Data Analysis")
        
        if st.session_state.processed_data is not None:
            # Time series visualization
            st.subheader("Time Series Trends")
            fig_time_series = plot_time_series(
                st.session_state.processed_data, 
                st.session_state.date_column, 
                st.session_state.features, 
                st.session_state.target_variable
            )
            st.plotly_chart(fig_time_series, use_container_width=True)
            
            # Volume chart
            st.subheader("Conversion Volumes")
            fig_volumes = plot_volumes(
                st.session_state.processed_data, 
                [f for f in st.session_state.features if f not in st.session_state.removed_features]
            )
            st.plotly_chart(fig_volumes, use_container_width=True)
            
            # Collinearity analysis
            st.subheader("Collinearity Analysis")
            fig_corr = plot_collinearity(
                st.session_state.processed_data, 
                [f for f in st.session_state.features if f not in st.session_state.removed_features]
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Correlation with target
            st.subheader("Correlation with Target Variable")
            fig_target_corr = plot_correlations(
                st.session_state.processed_data, 
                st.session_state.target_variable, 
                [f for f in st.session_state.features if f not in st.session_state.removed_features]
            )
            st.plotly_chart(fig_target_corr, use_container_width=True)
            
            # VIF analysis
            st.subheader("Multicollinearity Check (VIF)")
            features_to_check = [f for f in st.session_state.features if f not in st.session_state.removed_features]
            vif_df = calculate_vif(st.session_state.processed_data, features_to_check)
            
            st.dataframe(vif_df)
            
            # VIF warning
            if vif_df['VIF'].max() > 10:
                st.warning("""
                **High Multicollinearity Detected!**
                
                Some features have VIF values greater than 10, indicating strong multicollinearity.
                
                **Business Implication:** High multicollinearity makes it difficult to isolate the individual effect 
                of each conversion on the target variable. This can lead to unstable model coefficients 
                and potentially misleading interpretations about which conversions are most valuable.
                
                Consider removing some highly correlated features.
                """)
            
            # Feature selection and target variable selection
            st.subheader("Feature Selection and Target Variable")
            
            # Allow user to reselect target variable
            new_target = st.selectbox(
                "Confirm target variable",
                options=[st.session_state.target_variable] + [f for f in st.session_state.features],
                index=0
            )
            
            # If target variable changed, update features list
            if new_target != st.session_state.target_variable:
                if st.session_state.target_variable in st.session_state.features:
                    st.session_state.features.remove(st.session_state.target_variable)
                if new_target in st.session_state.features:
                    st.session_state.features.remove(new_target)
                    st.session_state.features.append(st.session_state.target_variable)
                st.session_state.target_variable = new_target
            
            # Feature selection
            features_to_keep = st.multiselect(
                "Select features to include in analysis",
                options=st.session_state.features,
                default=[f for f in st.session_state.features if f not in st.session_state.removed_features]
            )
            
            if st.button("Update Feature Selection"):
                st.session_state.removed_features = [f for f in st.session_state.features if f not in features_to_keep]
                st.success(f"Selected {len(features_to_keep)} features to include in the analysis")
                st.rerun()
        else:
            st.info("Please preprocess the data first")
    
    with tab3:
        st.header("Adstock Modeling")
        
        if st.session_state.processed_data is not None:
            active_features = [f for f in st.session_state.features if f not in st.session_state.removed_features]
            
            if st.button("Run Adstock Transformation"):
                with st.spinner("Applying adstock transformations... This may take a moment"):
                    try:
                        st.session_state.adstock_data, st.session_state.best_adstock_params = apply_adstock_transformations(
                            st.session_state.processed_data,
                            active_features,
                            st.session_state.target_variable,
                            max_lag=max_lag_weeks,
                            carryover_range=np.arange(carryover_min, carryover_max + 0.001, carryover_step)
                        )
                        st.success("Adstock transformations applied successfully")
                    except Exception as e:
                        st.error(f"Error applying adstock transformations: {e}")
            
            if st.session_state.adstock_data is not None and st.session_state.best_adstock_params:
                st.subheader("Best Adstock Parameters")
                
                # Convert dictionary to DataFrame for better display
                best_params_df = pd.DataFrame(
                    [(feature, params['carryover'], params['correlation']) 
                     for feature, params in st.session_state.best_adstock_params.items()],
                    columns=['Feature', 'Best Carryover Rate', 'Correlation with Target']
                ).sort_values('Correlation with Target', ascending=False)
                
                st.dataframe(best_params_df)
                
                st.subheader("Transformed Data Preview")
                st.dataframe(st.session_state.adstock_data.head())
                
                # Modeling section
                st.subheader("Build Models")
                
                if st.button("Build OLS Model"):
                    with st.spinner("Building OLS model..."):
                        try:
                            st.session_state.ols_results = build_ols_model(
                                st.session_state.adstock_data,
                                active_features,
                                st.session_state.target_variable
                            )
                            st.success("OLS model built successfully")
                        except Exception as e:
                            st.error(f"Error building OLS model: {e}")
                
                if st.button("Build XGBoost Model"):
                    with st.spinner("Building XGBoost model..."):
                        try:
                            st.session_state.xgb_results = build_xgboost_model(
                                st.session_state.adstock_data,
                                active_features,
                                st.session_state.target_variable
                            )
                            st.success("XGBoost model built successfully")
                        except Exception as e:
                            st.error(f"Error building XGBoost model: {e}")
            else:
                st.info("Run adstock transformation first")
        else:
            st.info("Please preprocess the data first")
    
    with tab4:
        st.header("Value Analysis")
        
        if st.session_state.ols_results is not None or st.session_state.xgb_results is not None:
            st.subheader("Model Results Comparison")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**OLS Model Results**")
                if st.session_state.ols_results is not None:
                    st.dataframe(st.session_state.ols_results['summary_df'])
                    
                    # Display OLS model statistics
                    st.metric("Model R-squared", f"{st.session_state.ols_results['r2']:.3f}")
                    st.metric("Adjusted R-squared", f"{st.session_state.ols_results['adj_r2']:.3f}")
                    
                    # Model summary
                    with st.expander("Detailed OLS Model Summary"):
                        st.text(st.session_state.ols_results['summary_text'])
                else:
                    st.info("OLS model not yet built")
            
            with col2:
                st.markdown("**XGBoost Model Results**")
                if st.session_state.xgb_results is not None:
                    st.dataframe(st.session_state.xgb_results['importance_df'])
                    
                    # Display XGBoost model statistics
                    st.metric("Model R-squared", f"{st.session_state.xgb_results['r2']:.3f}")
                    st.metric("RMSE", f"{st.session_state.xgb_results['rmse']:.3f}")
                    
                    # Plot feature importance
                    st.plotly_chart(st.session_state.xgb_results['importance_plot'], use_container_width=True)
                else:
                    st.info("XGBoost model not yet built")
            
            # Value calculation
            st.subheader("High Value Action Analysis")
            
            if st.session_state.ols_results is not None and st.session_state.xgb_results is not None:
                value_df = calculate_proxy_value(
                    st.session_state.ols_results['summary_df'],
                    st.session_state.xgb_results['importance_df'],
                    aov
                )
                
                st.dataframe(value_df)
                
                # Visualization of value contribution
                st.subheader("Value Contribution Comparison")
                
                # Create bar chart for value comparison
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots
                
                fig = make_subplots(rows=1, cols=2, 
                                   subplot_titles=("OLS Model Value", "XGBoost Model Value"),
                                   specs=[[{"type": "bar"}, {"type": "bar"}]])
                
                # Sort by OLS value for consistent order
                sorted_df = value_df.sort_values('OLS_Value', ascending=False)
                
                fig.add_trace(
                    go.Bar(x=sorted_df['Feature'], y=sorted_df['OLS_Value'], name="OLS Value"),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Bar(x=sorted_df['Feature'], y=sorted_df['XGB_Value'], name="XGB Value"),
                    row=1, col=2
                )
                
                fig.update_layout(height=500, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                # Disclaimer
                st.markdown("""
                ---
                ### **Important Note**
                
                **The results presented are directional in nature and should be validated with domain knowledge.** 
                
                These models provide insights into potential high-value actions based on historical correlation patterns, 
                but they should be interpreted alongside business context and expertise.
                
                Factors to consider:
                - The analyzed time period may not represent all business cycles
                - Correlation does not necessarily imply causation
                - External market factors may not be fully accounted for
                
                Always validate these findings against your business expertise before making significant strategic decisions.
                """)
            else:
                st.info("Build both models to see the value comparison")
        else:
            st.info("Please build models first")
else:
    st.info("Please upload data or select a demo dataset to begin analysis")
