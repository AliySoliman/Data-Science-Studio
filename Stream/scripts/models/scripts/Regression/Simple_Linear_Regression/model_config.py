import streamlit as st
import pandas as pd

def model_config(model_data, edit):
    # Check if selected_data exists and has columns
    if 'selected_data' not in st.session_state or st.session_state.selected_data.empty:
        st.error("No data available. Please load data first.")
        return {
            "features": [],
            "target": None,
            "model_type": "linear",
            "alpha": 1.0,
            "df": pd.DataFrame()
        }
    
    # Get all columns for feature selection
    all_cols = st.session_state.selected_data.columns.tolist()
    
    # Feature and target selection
    col1, col2 = st.columns(2)
    
    with col1:
        # Handle default features for edit mode
        default_features = []
        if edit and model_data.get('model param'):
            for param in model_data['model param']:
                if param['name'] == 'features':
                    suggested_features = param['value'] if isinstance(param['value'], list) else []
                    # Safety filter: ensure recommended features exist in current available options
                    default_features = [f for f in suggested_features if f in all_cols]
                    break
        
        features = st.multiselect(
            "Select feature columns:",
            options=all_cols,
            default=default_features
        )
    
    with col2:
        # Handle default target for edit mode
        default_target = None
        if edit and model_data.get('model param'):
            for param in model_data['model param']:
                if param['name'] == 'target':
                    default_target = param['value']
                    break
        
        # Find default index
        default_index = 0
        if default_target and default_target in all_cols:
            default_index = all_cols.index(default_target)
        
        target = st.selectbox(
            "Select target column:",
            options=all_cols,
            index=default_index,
            help="Select the continuous variable you want to predict"
        )
    
    # Model type selection
    st.subheader("Model Configuration")
    
    # Default values
    default_model_type = "linear"
    default_alpha = 1.0
    
    # Extract parameters from model_data if in edit mode
    if edit and model_data.get('model param'):
        param_dict = {}
        for param in model_data['model param']:
            param_dict[param['name']] = param['value']
        
        default_model_type = param_dict.get('model_type', 'linear')
        try:
            default_alpha = float(param_dict.get('alpha', 1.0))
        except (ValueError, TypeError):
            default_alpha = 1.0
    
    model_type = st.radio(
        "Select Regression Model:",
        options=["linear", "ridge", "lasso"],
        format_func=lambda x: {
            "linear": "Linear Regression",
            "ridge": "Ridge Regression (L2 Regularization)",
            "lasso": "Lasso Regression (L1 Regularization)"
        }[x],
        index=["linear", "ridge", "lasso"].index(default_model_type) 
        if default_model_type in ["linear", "ridge", "lasso"] else 0
    )
    
    # Alpha parameter for Ridge/Lasso
    if model_type in ["ridge", "lasso"]:
        alpha = st.number_input(
            "Regularization strength (alpha):",
            min_value=0.01,
            max_value=100.0,
            value=default_alpha,
            step=0.1,
            help="Higher alpha values increase regularization strength"
        )
        
        if model_type == "ridge":
            st.info("Ridge Regression: Uses L2 regularization to prevent overfitting")
        else:
            st.info("Lasso Regression: Uses L1 regularization for feature selection")
    else:
        alpha = 1.0
        st.info("Linear Regression: Ordinary least squares without regularization")
    
    # Data preview and statistics
    if features and target:
        st.subheader("Data Preview")
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.write("Selected Features Preview:")
            st.dataframe(st.session_state.selected_data[features].head(), use_container_width=True)
        
        with col4:
            st.write("Target Variable Statistics:")
            target_stats = st.session_state.selected_data[target].describe()
            st.dataframe(pd.DataFrame(target_stats).T, use_container_width=True)
        
        # Show correlation with target
        if len(features) > 0:
            st.write("Feature-Target Correlations:")
            correlations = st.session_state.selected_data[features + [target]].corr()[target].drop(target)
            corr_df = pd.DataFrame({
                'Feature': correlations.index,
                'Correlation with Target': correlations.values
            }).sort_values('Correlation with Target', key=abs, ascending=False)
            
            st.dataframe(corr_df, use_container_width=True)
    
    return {
        "features": features,
        "target": target,
        "model_type": model_type,
        "alpha": alpha,
        "df": st.session_state.selected_data,
        "edit": edit
    }
model_description = """
<div class="code-container">
    <div class="code-header">
        <span>MODEL DESCRIPTION</span>
        <span>Linear Regression Models</span>
    </div>
    <div>
        Linear Regression models predict continuous target variables using linear relationships 
        between features and target. Three variants are available:
        
        <strong>1. Linear Regression:</strong>
        • Ordinary Least Squares regression
        • Minimizes sum of squared residuals
        • Best for datasets without multicollinearity
        
        <strong>2. Lasso Regression (L1):</strong>
        • Adds L1 regularization penalty
        • Performs feature selection by shrinking some coefficients to zero
        • Good for datasets with many irrelevant features
        
        <strong>3. Ridge Regression (L2):</strong>
        • Adds L2 regularization penalty
        • Shrinks coefficients but doesn't eliminate them
        • Handles multicollinearity well
        
        <strong>Key Evaluation Metrics:</strong>
        • <strong>R² Score:</strong> Proportion of variance explained
        • <strong>MSE:</strong> Mean Squared Error
        • <strong>MAE:</strong> Mean Absolute Error
        • <strong>RMSE:</strong> Root Mean Squared Error
        
        <strong>Hyperparameters:</strong>
        • <strong>alpha:</strong> Regularization strength (Lasso/Ridge)
        • <strong>fit_intercept:</strong> Whether to calculate intercept
        • <strong>normalize:</strong> Whether to normalize features
    </div>
</div>
"""
model_reference_code = """
<div class="code-container">
    <div class="code-header">
        <span>REFERENCE CODE</span>
        <span>Linear Regression Implementation</span>
    </div>
    <div>
        <pre style="background: #f4f4f4; padding: 10px; border-radius: 5px; overflow-x: auto;">
<code>
# Linear Regression Implementation
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Load and prepare data
data = pd.read_csv('your_data.csv')
X = data[['feature1', 'feature2', 'feature3']]
y = data['target']

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

# Model Selection and Grid Search
model_type = 'ridge'  # 'linear', 'lasso', or 'ridge'

if model_type == 'linear':
    model = LinearRegression()
    param_grid = {
        'fit_intercept': [True, False],
        'copy_X': [True, False]
    }
elif model_type == 'lasso':
    model = Lasso()
    param_grid = {
        'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
        'fit_intercept': [True, False],
        'max_iter': [1000, 2000]
    }
else:  # ridge
    model = Ridge()
    param_grid = {
        'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
        'fit_intercept': [True, False],
        'solver': ['auto', 'svd', 'cholesky']
    }

# With Grid Search
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Calculate metrics
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"R² Score: {r2:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print("Best parameters:", grid_search.best_params_)
</code>
        </pre>
    </div>
</div>
"""