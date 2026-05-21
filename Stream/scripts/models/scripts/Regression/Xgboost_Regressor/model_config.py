import streamlit as st
import pandas as pd
import numpy as np

def model_config(model_data, edit):
    if 'selected_data' not in st.session_state or st.session_state.selected_data.empty:
        st.error("No data available. Please load data first.")
        return {
            "features": [],
            "target": None,
            "df": pd.DataFrame(),
            "edit": edit,
            "use_grid_search": True,
            "param_grid": {},
            "manual_params": {},
            "cv_folds": 5
        }
    
    all_cols = st.session_state.selected_data.columns.tolist()
    numeric_cols = [col for col in all_cols 
                   if st.session_state.selected_data[col].dtype in ['int64', 'float64', 'int32', 'float32']]
    
    col1, col2 = st.columns(2)
    with col1:
        default_features = []
        if edit and model_data.get('model param'):
            for param in model_data['model param']:
                if param['name'] == 'features':
                    # Extract features and ensure they exist in current data
                    suggested_features = param['value'] if isinstance(param['value'], list) else []
                    default_features = [f for f in suggested_features if f in all_cols]
                    break
        
        features = st.multiselect(
            "Select feature columns:",
            options=all_cols,
            default=default_features
        )
    
    with col2:
        if len(numeric_cols) == 0:
            st.warning("No numeric columns available for target selection")
            target = None
        else:
            numeric_targets = [col for col in numeric_cols if col not in features]
            default_index = 0
            if edit and model_data.get('model param'):
                target_value = None
                for param in model_data['model param']:
                    if param['name'] == 'target':
                        target_value = param['value']
                        break
                
                if target_value and target_value in numeric_targets:
                    default_index = numeric_targets.index(target_value)
            
            target = st.selectbox(
                "Select target column:",
                options=numeric_targets,
                index=default_index,
                key="target_column_xgb_reg"
            )
    
    st.subheader("Model Configuration")
    
    default_use_grid_search = True
    default_cv_folds = 5
    
    if edit and model_data.get('model param'):
        param_dict = {}
        for param in model_data['model param']:
            param_dict[param['name']] = param['value']
        
        default_use_grid_search = param_dict.get('use_grid_search', True)
        default_cv_folds = int(param_dict.get('cv_folds', 5))
    
    use_grid_search = st.checkbox(
        "Use Grid Search for hyperparameter tuning",
        value=default_use_grid_search
    )
    
    if use_grid_search:
        st.info("Grid Search will automatically find the best hyperparameters for XGBoost Regressor")
        
        col3, col4 = st.columns(2)
        with col3:
            cv_folds = st.number_input(
                "Cross-validation folds:",
                min_value=2,
                max_value=10,
                value=default_cv_folds
            )
        
        with col4:
            n_estimators_range = st.text_input(
                "Number of Estimators (Trees) (comma-separated):",
                value="100,200,300"
            )
            try:
                n_estimators_values = [int(x.strip()) for x in n_estimators_range.split(',')]
            except (ValueError, TypeError):
                n_estimators_values = [100, 200, 300]
                st.warning("Invalid format. Using default estimator values")
        
        col5, col6 = st.columns(2)
        with col5:
            learning_rate_range = st.text_input(
                "Learning Rate (comma-separated):",
                value="0.01,0.1,0.2"
            )
            try:
                learning_rate_values = [float(x.strip()) for x in learning_rate_range.split(',')]
            except (ValueError, TypeError):
                learning_rate_values = [0.01, 0.1, 0.2]
                st.warning("Invalid format. Using default learning rate values")
        
        with col6:
            max_depth_range = st.text_input(
                "Max Depth (comma-separated):",
                value="3,5,7"
            )
            try:
                max_depth_values = [int(x.strip()) for x in max_depth_range.split(',')]
            except (ValueError, TypeError):
                max_depth_values = [3, 5, 7]
                st.warning("Invalid format. Using default max depth values")
                
        col7, col8 = st.columns(2)
        with col7:
            min_child_weight_range = st.text_input(
                "Min Child Weight (comma-separated):",
                value="1,3,5"
            )
            try:
                min_child_weight_values = [int(x.strip()) for x in min_child_weight_range.split(',')]
            except (ValueError, TypeError):
                min_child_weight_values = [1, 3, 5]
                st.warning("Invalid format. Using default min child weight values")
        
        with col8:
            gamma_range = st.text_input(
                "Gamma (comma-separated):",
                value="0,0.1,0.2"
            )
            try:
                gamma_values = [float(x.strip()) for x in gamma_range.split(',')]
            except (ValueError, TypeError):
                gamma_values = [0, 0.1, 0.2]
                st.warning("Invalid format. Using default gamma values")
        
        param_grid = {
            'n_estimators': n_estimators_values,
            'learning_rate': learning_rate_values,
            'max_depth': max_depth_values,
            'min_child_weight': min_child_weight_values,
            'gamma': gamma_values
        }
        
        manual_params = {}
        
    else:
        st.info("Configure hyperparameters manually for XGBoost Regressor")
        
        cv_folds = 5
        
        col3, col4, col5 = st.columns(3)
        with col3:
            n_estimators = st.number_input(
                "Number of Estimators:",
                min_value=10,
                max_value=1000,
                value=100,
                step=10,
                help="The number of boosting stages to perform"
            )
        with col4:
            learning_rate = st.number_input(
                "Learning Rate:",
                min_value=0.001,
                max_value=1.0,
                value=0.1,
                step=0.01,
                format="%.3f",
                help="Learning rate shrinks the contribution of each tree"
            )
        with col5:
            max_depth = st.number_input(
                "Max Depth:",
                min_value=1,
                max_value=20,
                value=3,
                step=1,
                help="Maximum depth of the individual trees"
            )
            
        col6, col7 = st.columns(2)
        with col6:
            min_child_weight = st.number_input(
                "Min Child Weight:",
                min_value=1,
                max_value=20,
                value=1,
                step=1,
                help="Minimum sum of instance weight (hessian) needed in a child"
            )
        with col7:
            gamma = st.number_input(
                "Gamma:",
                min_value=0.0,
                max_value=10.0,
                value=0.0,
                step=0.1,
                format="%.1f",
                help="Minimum loss reduction required to make a further partition on a leaf node of the tree"
            )
        
        manual_params = {
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'min_child_weight': min_child_weight,
            'gamma': gamma
        }
        param_grid = {}
    
    return {
        "features": features,
        "target": target,
        "df": st.session_state.selected_data,
        "edit": edit,
        "use_grid_search": use_grid_search,
        "param_grid": param_grid,
        "manual_params": manual_params,
        "cv_folds": cv_folds
    }

model_description = """
<div class="code-container">
    <div class="code-header">
        <span>MODEL DESCRIPTION</span>
        <span>XGBoost Regressor</span>
    </div>
    <div>
        <strong>XGBoost (eXtreme Gradient Boosting)</strong> is an optimized distributed gradient boosting library 
        designed to be highly efficient, flexible and portable. It implements machine learning algorithms under the 
        Gradient Boosting framework.
        
        <br><br><strong>Key Features:</strong><br>
        • <strong>Regularization:</strong> Prevents overfitting using L1/L2 penalties on tree weights<br>
        • <strong>Sparsity Aware:</strong> Naturally handles sparse data and missing values<br>
        • <strong>Tree Pruning:</strong> Employs a depth-first approach and prunes backwards<br>
        • <strong>Parallelized:</strong> Very fast implementation utilizing multiple CPU cores<br>
        
        <br><strong>Key Evaluation Metrics:</strong><br>
        • <strong>R² Score:</strong> Proportion of variance explained (closer to 1 is better)<br>
        • <strong>MSE/RMSE:</strong> Mean Squared/Root Mean Squared Error (lower is better)<br>
        • <strong>MAE:</strong> Mean Absolute Error (lower is better)<br>
        
        <br><strong>Key Parameters:</strong><br>
        • <strong>n_estimators:</strong> Number of gradient boosted trees<br>
        • <strong>learning_rate:</strong> Step size shrinkage<br>
        • <strong>max_depth:</strong> Maximum tree depth<br>
        • <strong>min_child_weight:</strong> Minimum child weight (hessian) for tree partitioning<br>
        • <strong>gamma:</strong> Minimum loss reduction required to make a further partition<br>
    </div>
</div>
"""

model_reference_code = """
<div class="code-container">
    <div class="code-header">
        <span>REFERENCE CODE</span>
        <span>XGBoost Regressor Implementation</span>
    </div>
    <div>
        <pre><code class="language-python">
# XGBoost Regressor Example
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 1. Prepare Data
X = df[['feature1', 'feature2', 'feature3']]
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Define Param Grid
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.1, 0.01],
    'max_depth': [3, 5],
    'min_child_weight': [1, 3],
    'gamma': [0, 0.1]
}

# 3. Initialize Model and GridSearchCV
xgb = XGBRegressor(random_state=42)
grid_search = GridSearchCV(xgb, param_grid, cv=5, scoring='neg_mean_squared_error')

# 4. Fit the Model
grid_search.fit(X_train, y_train)

# 5. Evaluate
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print(f"R2 Score: {r2_score(y_test, y_pred)}")
print(f"MSE: {mean_squared_error(y_test, y_pred)}")
print(f"MAE: {mean_absolute_error(y_test, y_pred)}")
print(f"Best Parameters: {grid_search.best_params_}")
        </code></pre>
    </div>
</div>
"""
