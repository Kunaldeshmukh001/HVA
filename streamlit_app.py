
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from datetime import datetime

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_theme()
sns.set_palette("husl")

# Streamlit UI - CSV Upload
st.title("High Value Actions")
st.header("Upload your CSV data")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read CSV
    df = pd.read_csv(uploaded_file)
    st.write("First 5 rows of your data")
    st.write(df.head())

    # User input parameters
    Target = 'Sales'
    Date = 'Dt'
    aggregation_needed = 'Daily'
    train_test_split_method = 'random'
    vif_threshold = 20
    p_value_threshold = 0.2
    clip_outliers = 'yes'
    max_lag = 3

    # Handle missing values
    df.fillna(0, inplace=True)

    # Drop columns based on zero percentages
    zero_counts = df.eq(0).sum(axis=0)
    zero_percentages = zero_counts / len(df)
    columns_to_drop = zero_percentages[zero_percentages > 0.8].index
    df.drop(columns_to_drop, axis=1, inplace=True)
    st.write(f"Columns dropped due to high zero values: {', '.join(columns_to_drop)}")

    # Convert Date and Target columns to proper types
    df[Date] = pd.to_datetime(df[Date])
    df = df.set_index(Date)
    df[Target] = pd.to_numeric(df[Target])

    # Split target and predictor variables
    y = df[[Target]]
    x = df.drop(columns=[Target])

    # Resample the data for weekly/daily aggregation
    if aggregation_needed == 'Weekly':
        x = x.resample('7D').sum()
        y = y.resample('7D').sum()
    st.write(f"Aggregated Data ({aggregation_needed})", x.head())

    # Convert to DataFrame if Series
    if isinstance(x, pd.Series):
        x = x.to_frame()
    if isinstance(y, pd.Series):
        y = y.to_frame()

    # Train-test split
    if train_test_split_method == 'random':
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    else:
        split_index = int(len(x) * 0.8)
        X_train = x[:split_index].copy()
        X_test = x[split_index:].copy()
        y_train = y[:split_index].copy()
        y_test = y[split_index:].copy()

    st.write(f"Train/Test Split ({train_test_split_method})", X_train.shape, X_test.shape)

    # Data Distribution and Visualization Section
    st.header("Data Analysis and Visualizations")

    # 1. Visualize time series of input features
    st.subheader("Input Features Time Series")
    num_features = len(x.columns)
    fig_height = max(6, num_features * 2)
    fig, axes = plt.subplots(nrows=num_features, figsize=(15, fig_height))
    if num_features == 1:
        axes = [axes]

    for ax, col in zip(axes, x.columns):
        x[col].plot(ax=ax)
        ax.set_title(f'Time Series: {col}')
        ax.grid(True)
    plt.tight_layout()
    st.pyplot(fig)

    # 2. Visualize target variable time series
    st.subheader("Target Variable Time Series")
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.fill_between(y.index, y[Target], alpha=0.3, color="#4285F4")
    ax.plot(y.index, y[Target], color="#4285F4", linewidth=2)
    ax.set_title(f'{Target} Over Time')
    ax.grid(True)
    plt.tight_layout()
    st.pyplot(fig)

    # 3. Box Plot for Outliers
    st.subheader("Outliers Analysis - Box Plots")
    fig, ax = plt.subplots(figsize=(15, 6))
    sns.boxplot(data=x, ax=ax)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)

    # 4. Zero-inflated columns visualization
    st.subheader("Zero-Inflated Columns Analysis")
    zero_percentages = x.eq(0).sum() / len(x) * 100
    zero_df = pd.DataFrame({'Feature': x.columns, 'Zero Percentage': zero_percentages})

    fig, ax = plt.subplots(figsize=(15, 6))
    sns.barplot(data=zero_df, x='Feature', y='Zero Percentage', ax=ax)
    plt.xticks(rotation=45, ha='right')
    ax.set_title('Percentage of Zero Values by Feature')
    plt.tight_layout()
    st.pyplot(fig)

    # 5. Correlation Analysis
    st.subheader("Correlation Analysis")
    correlations = []
    for col in x.columns:
        correlation, p_value = stats.pearsonr(x[col], y[Target])
        correlations.append({
            'Feature': col,
            'Correlation': correlation,
            'P-value': p_value
        })

    corr_df = pd.DataFrame(correlations)
    corr_df = corr_df.sort_values('Correlation', ascending=True)

    fig, ax = plt.subplots(figsize=(15, 6))
    sns.barplot(data=corr_df, x='Correlation', y='Feature', ax=ax)
    ax.set_title('Feature Correlations with Target')
    plt.tight_layout()
    st.pyplot(fig)
    st.write("Correlation Details:")
    st.write(corr_df)

    # 6. Scatter plots for each feature vs target
    st.subheader("Feature vs Target Scatter Plots")
    num_cols = 3
    num_rows = (len(x.columns) + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5*num_rows))
    axes = axes.flatten()

    for idx, col in enumerate(x.columns):
        sns.scatterplot(data=df, x=col, y=Target, ax=axes[idx], alpha=0.5)
        axes[idx].set_title(f'{col} vs {Target}')

    for idx in range(len(x.columns), len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    st.pyplot(fig)

    # Drop negatively correlated columns
    negative_features = corr_df[corr_df['Correlation'] < 0]['Feature'].tolist()
    if negative_features:
        st.write(f"Dropping negatively correlated features: {negative_features}")
        x = x.drop(columns=negative_features)
        X_train = X_train.drop(columns=negative_features)
        X_test = X_test.drop(columns=negative_features)

    # Define function to clip outliers
    def clip_outliers(data, clip_outliers):
        if clip_outliers == 'yes':
            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            for col in data.columns:
                data[col] = np.clip(data[col], lower_bound[col], upper_bound[col])
        return data

    X_train = clip_outliers(X_train, clip_outliers)
    X_test = clip_outliers(X_test, clip_outliers)
    x = clip_outliers(x, clip_outliers)
    y_train = clip_outliers(y_train, clip_outliers)
    y_test = clip_outliers(y_test, clip_outliers)

    # Apply Lag Variables
    def transformation(dataframe, column):
        lag = []
        for i in range(1, max_lag + 1):
            data = dataframe[column].shift(i).to_frame()
            data.columns = [f'{column}__Lag{i}']
            lag.append(data)
        return pd.concat(lag, axis=1).fillna(0)

    X_train_with_lags = X_train.copy()
    X_test_with_lags = X_test.copy()
    x_with_lags = x.copy()

    for col in X_train.columns:
        transformed_features = transformation(X_train, col)
        X_train_with_lags = pd.concat([X_train_with_lags, transformed_features], axis=1)

    for col in X_test.columns:
        transformed_features = transformation(X_test, col)
        X_test_with_lags = pd.concat([X_test_with_lags, transformed_features], axis=1)

    for col in x.columns:
        transformed_features = transformation(x, col)
        x_with_lags = pd.concat([x_with_lags, transformed_features], axis=1)

    st.subheader("Feature Selection - Best Lags")
    original_features = [col for col in X_train_with_lags.columns if '__Lag' not in col]
    best_features = []

    for feature in original_features:
        related_columns = [col for col in X_train_with_lags.columns if col == feature or col.startswith(f"{feature}__Lag")]
        corr_df = pd.DataFrame([(col, np.abs(stats.pearsonr(X_train_with_lags[col], y_train[Target])[0])) for col in related_columns], columns=['Feature', 'Correlation'])
        corr_df = corr_df.sort_values(by='Correlation', ascending=False)
        best_feature = corr_df.iloc[0]["Feature"]
        best_features.append(best_feature)

    selected_features = X_train_with_lags[best_features]
    X_test = X_test_with_lags[best_features]
    x = x_with_lags[best_features]
    st.write("Best features with lags selected:", best_features)

    # VIF Calculation with Constant
    def calculate_vif(X):
        X = add_constant(X)  # Add a constant to the dataset
        vif_data = pd.DataFrame()
        vif_data["Feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        vif_data = vif_data[vif_data["Feature"] != "const"]  # Exclude constant from the list
        return vif_data.sort_values('VIF', ascending=False)

    # Iteratively remove features with high VIF
    cols_to_drop = []
    selected_features_with_const = add_constant(selected_features)  # Include constant in the starting dataset

    while True:
        vif_df = calculate_vif(selected_features_with_const)
        if vif_df.empty or vif_df["VIF"].max() < vif_threshold:
            break

        # Identify the feature with the highest VIF (excluding the constant)
        feature_to_drop = vif_df.iloc[0]["Feature"]
        cols_to_drop.append(feature_to_drop)
        selected_features_with_const = selected_features_with_const.drop(columns=[feature_to_drop])

        # Also remove from test and full dataset if it exists
        if feature_to_drop in X_test.columns:
            X_test = X_test.drop(columns=[feature_to_drop])
        if feature_to_drop in x.columns:
            x = x.drop(columns=[feature_to_drop])

    # Display results
    if cols_to_drop:
        st.write("Columns dropped due to high VIF:", cols_to_drop)
    else:
        st.write("No columns needed to be dropped based on VIF analysis")

    # Display final VIF values without the constant
    final_vif = calculate_vif(selected_features_with_const.drop(columns="const", errors="ignore"))
    st.write("Final VIF values:")
    st.write(final_vif)
    import streamlit as st
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import statsmodels.api as sm
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score
    import xgboost as xgb
    import shap

    # Display header
    st.title("Feature Importance Analysis with XGBoost and SHAP")

    # Assuming your dataset and preprocessed DataFrames like 'df_pivot' and 'selected_features_scaled' already exist



    # Prepare the final DataFrame
    final_df = selected_features_with_const
    final_df['Date'] = final_df.index
    final_df['Target'] = y

    # Plot setup
    plt.rcParams['figure.figsize'] = [20, 10]

    # Further DataFrame setup
    df = final_df
    df = df.set_index(df['Date']).drop(columns=['Date'])
    kpi = 'Target'
    y_train = df[df.columns[df.columns == kpi]]
    X_train = df[df.columns[df.columns != kpi]]

    # Add a constant to X_train and X_test
    X_test['const'] = 1
    X_train = sm.add_constant(X_train)

    # OLS model creation and backward elimination
    model = sm.OLS(y_train, X_train)
    results = model.fit()

    # Backward elimination based on p-values
    while True:
        # Identify variables to drop based on p-values
        remaining_vars = X_train.columns[1:]
        p_values = results.pvalues.loc[remaining_vars]
        max_p = p_values.max()

        # Remove high p-value variables
        if max_p > p_value_threshold:
            drop_var = p_values.idxmax()
            X_train = X_train.drop(drop_var, axis=1)
            st.write(f"Dropped variable '{drop_var}' with p-value {max_p:.4f}")
            X_test = X_test.drop(drop_var, axis=1)
            x = x.drop(drop_var, axis=1)

            # Refit the model
            model = sm.OLS(y_train, X_train)
            results = model.fit()
        else:
            break

    st.write("OLS Regression Model Summary:")
    st.text(results.summary())

    # Align test data columns with training data
    X_test = X_test.reindex(columns=X_train.columns)
    y_pred = results.predict(X_test)

    # Calculate R-squared
    r2 = r2_score(y_test, y_pred)
    st.write("Final test R-squared value is", r2)

    # XGBoost Model Training
    X_train1, X_test1, y_train1, y_test1 = train_test_split(x, y, test_size=0.2, random_state=42)
    dtrain = xgb.DMatrix(X_train1, label=y_train1)
    dtest = xgb.DMatrix(X_test1, label=y_test1)
    params = {'objective': 'reg:squarederror'}
    model = xgb.train(params, dtrain, evals=[(dtrain, 'train'), (dtest, 'test')], feval=lambda preds, dmatrix: ('r2', r2_score(dmatrix.get_label(), preds)))

    # Feature Importance Analysis
    importance_scores = model.get_score(importance_type='gain')
    sorted_scores = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
    feature_names, feature_scores = zip(*sorted_scores)
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance Score': feature_scores})
    scaling_factor = 100 / feature_importance_df['Importance Score'].sum()
    feature_importance_df['Scaled Importance'] = feature_importance_df['Importance Score'] * scaling_factor
    st.write("Feature Importance DataFrame:")
    st.dataframe(feature_importance_df)

    # SHAP Analysis
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x)
    average_abs_shap_values = np.mean(abs(shap_values), axis=0)
    scaling_factor = 100 / np.sum(np.abs(average_abs_shap_values))
    scaled_average_shap_values = (average_abs_shap_values * scaling_factor)
    sorted_feature_names, sorted_shap_values = zip(*sorted(zip(x.columns, scaled_average_shap_values), key=lambda x: -abs(x[1])))

    # Plotting Scaled SHAP Values
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_shap_values)), sorted_shap_values)
    plt.yticks(range(len(sorted_shap_values)), sorted_feature_names)
    plt.xlabel('Scaled Average SHAP Value (Sum = 100)')
    plt.ylabel('Feature')
    plt.title('Scaled Average SHAP Values for Features (Descending Order)')
    plt.xlim(0, 100)
    plt.gca().invert_yaxis()
    plt.tight_layout()

    # Add value labels
    for i, value in enumerate(sorted_shap_values):
        plt.text(value + 2, i, f'{value:.2f}', va='center')

    # Display the plot in Streamlit
    st.pyplot(plt)




