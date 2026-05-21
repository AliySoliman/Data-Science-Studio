import streamlit as st
import numpy as np

def model_config(model_data, edit):
    # Check if selected_data exists and has columns
    if 'selected_data' not in st.session_state or st.session_state.selected_data.empty:
        st.error("No data available. Please load data first.")
        return {
            "features": [],
            "target": None,
            "df": None,
            "edit": edit,
            "use_grid_search": True,
            "param_grid": {},
            "manual_params": {},
            "cv_folds": 5,
            "solver": 'lbfgs',
            "max_iter": 100
        }
    
    numeric_cols = [col for col in st.session_state.selected_data.columns 
                   if st.session_state.selected_data[col].dtype in ['int64', 'float64']]
    
    # Feature and target selection
    all_cols = st.session_state.selected_data.columns.tolist()
    col1, col2 = st.columns(2)
    with col1:
        # Features can be numeric or categorical (script handles encoding)
        feature_options = all_cols
        
        # Handle default features for edit mode
        default_features = []
        if edit and model_data.get('model param'):
            for param in model_data['model param']:
                if param['name'] == 'features':
                    suggested_features = param['value'] if isinstance(param['value'], list) else []
                    # Safety filter: ensure recommended features exist in current numeric options
                    default_features = [f for f in suggested_features if f in feature_options]
                    break
        
        features = st.multiselect(
            "Select feature columns (Numerical or Categorical):",
            options=feature_options,
            default=default_features
        )
    
    with col2:
        if len(numeric_cols) == 0:
            st.warning("No numeric columns available for target selection")
            target = None
        else:
            # For classification, target can be categorical or numeric
            all_cols = st.session_state.selected_data.columns.tolist()
            default_index = 0
            if edit and model_data.get('model param'):
                target_value = None
                for param in model_data['model param']:
                    if param['name'] == 'target':
                        target_value = param['value']
                        break
                
                if target_value and target_value in all_cols:
                    default_index = all_cols.index(target_value)
            
            target = st.selectbox(
                "Select target column:",
                options=all_cols,
                index=default_index,
                key="target_column_logistic"
            )
    
    # Grid Search Configuration
    st.subheader("Hyperparameter Configuration")
    
    # Default values
    default_use_grid_search = True
    default_solver = 'lbfgs'
    default_max_iter = 100
    default_cv_folds = 5
    default_c_value = "0.1,1,10"
    
    # Extract parameters from model_data if in edit mode
    if edit and model_data.get('model param'):
        param_dict = {}
        for param in model_data['model param']:
            param_dict[param['name']] = param['value']
        
        default_use_grid_search = param_dict.get('use_grid_search', True)
        default_solver = param_dict.get('solver', 'lbfgs')
        
        # Fix: Handle max_iter conversion safely
        max_iter_value = param_dict.get('max_iter', '100')
        try:
            # If it's a string representation of a list, take the first value
            if isinstance(max_iter_value, str) and max_iter_value.startswith('['):
                # Extract first number from list string like '[100, 200]'
                import re
                numbers = re.findall(r'\d+', max_iter_value)
                default_max_iter = int(numbers[0]) if numbers else 100
            else:
                default_max_iter = int(max_iter_value)
        except (ValueError, TypeError):
            default_max_iter = 100
        
        default_cv_folds = int(param_dict.get('cv_folds', 5))
        default_c_value = param_dict.get('c_values', "0.1,1,10")

    # Add checkbox for Grid Search selection
    use_grid_search = st.checkbox(
        "Use Grid Search for hyperparameter tuning",
        value=default_use_grid_search
    )

    if use_grid_search:
        st.info("Grid Search will automatically find the best hyperparameters")
        
        # Grid Search parameters
        col3, col4 = st.columns(2)
        with col3:
            c_values_input = st.text_input(
                "C values (comma-separated):",
                value=default_c_value,
                help="Regularization strength - smaller values specify stronger regularization"
            )
        
        with col4:
            cv_folds = st.number_input(
                "Cross-validation folds:",
                min_value=2,
                max_value=10,
                value=default_cv_folds
            )
        
        # Parse C values
        try:
            c_values = [float(x.strip()) for x in c_values_input.split(',')]
        except (ValueError, TypeError):
            c_values = [0.1, 1, 10]
            st.warning("Invalid format. Using default values: 0.1, 1, 10")
        
        solvers = ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga']
        
        param_grid = {
            'C': c_values,
            'solver': solvers,
            'max_iter': [100, 200]
        }
        
        manual_params = {}
        
    else:
        # Manual parameter configuration
        st.info("Configure hyperparameters manually")
        
        col3, col4, col5 = st.columns(3)
        
        with col3:
            c_value = st.number_input(
                "Regularization Strength (C):",
                min_value=0.01,
                max_value=100.0,
                value=1.0,
                step=0.1,
                help="Inverse of regularization strength"
            )
        
        with col4:
            solver = st.selectbox(
                "Solver:",
                options=['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga'],
                index=0 if default_solver == 'lbfgs' else ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga'].index(default_solver) 
                if default_solver in ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga'] else 0
            )
        
        with col5:
            max_iter = st.number_input(
                "Maximum iterations:",
                min_value=50,
                max_value=1000,
                value=default_max_iter
            )
        
        manual_params = {
            'C': c_value,
            'solver': solver,
            'max_iter': int(max_iter)
        }
        param_grid = {}
        cv_folds = 5
        c_values = []
    
    return {
        "features": features,
        "target": target,
        "df": st.session_state.selected_data,
        "edit": edit,
        "use_grid_search": use_grid_search,
        "param_grid": param_grid,
        "manual_params": manual_params,
        "cv_folds": cv_folds,
        "solver": solver if not use_grid_search else '',
        "max_iter": max_iter if not use_grid_search else 100
    }
model_reference_code = """
<div class="code-container">
    <div class="code-header">
        <span>REFERENCE CODE</span>
        <span>Logistic Regression Implementation</span>
    </div>
    <div>
        <pre><code>
# Logistic Regression Classifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import pandas as pd
import numpy as np

def train_logistic_regression(df, features, target, use_grid_search=True, cv_folds=5):
    # Prepare data
    X = df[features].values
    y = df[target].values
    
    # Encode target if categorical
    if df[target].dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    if use_grid_search:
        # Grid Search configuration
        param_grid = {
            'C': [0.1, 1, 10],
            'solver': ['lbfgs', 'liblinear', 'newton-cg'],
            'max_iter': [100, 200]
        }
        
        lr = LogisticRegression(random_state=42)
        grid_search = GridSearchCV(lr, param_grid, cv=cv_folds, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        print(f"Best parameters: {best_params}")
        
    else:
        # Manual parameters
        best_model = LogisticRegression(
            C=1.0,
            solver='lbfgs',
            max_iter=100,
            random_state=42
        )
        best_model.fit(X_train, y_train)
        best_params = {'C': 1.0, 'solver': 'lbfgs', 'max_iter': 100}
    
    # Make predictions
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # AUC-ROC for binary classification
    if len(best_model.classes_) == 2:
        auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])
        print(f"AUC-ROC: {auc_score:.4f}")
    
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(class_report)
    
    return {
        'model': best_model,
        'scaler': scaler,
        'accuracy': accuracy,
        'classification_report': class_report,
        'confusion_matrix': conf_matrix,
        'best_params': best_params
    }

# Usage example:
# results = train_logistic_regression(df, ['age', 'income'], 'purchased')
        </code></pre>
    </div>
</div>
"""
model_description = """
<div class="code-container">
    <div class="code-header">
        <span>MODEL DESCRIPTION</span>
        <span>Logistic Regression Classifier</span>
    </div>
    <div>
        Logistic Regression is a statistical model that uses a logistic function to model 
        a binary dependent variable. It's widely used for classification tasks, especially 
        when the target variable is categorical.
        
        Key Concepts:
        • Sigmoid Function: Maps predicted values to probabilities between 0 and 1
        • Decision Boundary: Typically 0.5 threshold for binary classification
        • Regularization: Prevents overfitting through L1/L2 regularization

        Hyperparameters:
        • C: Inverse of regularization strength (smaller values = stronger regularization)
        • solver: Algorithm to use in optimization problem
        • max_iter: Maximum number of iterations for convergence
        
        Key Evaluation Metrics:
        • Accuracy: Overall classification correctness
        • Precision: Ratio of true positives to predicted positives
        • Recall: Ratio of true positives to actual positives  
        • F1-Score: Harmonic mean of precision and recall
        • AUC-ROC: Area under the ROC curve (for binary classification)
        
        Best For: Binary classification problems, linearly separable data, 
        interpretable models where coefficient importance matters.
    </div>
</div>
"""