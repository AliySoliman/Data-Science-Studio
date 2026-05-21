import streamlit as st

def model_config(model_data, edit):        
    numeric_cols = [col for col in st.session_state.selected_data.columns 
                   if st.session_state.selected_data[col].dtype in ['int64', 'float64']] 
    
    all_cols = st.session_state.selected_data.columns.tolist()
    # Feature and target selection
    col1, col2 = st.columns(2)
    with col1:
        # Filter numeric columns for features
        feature_options = [col for col in all_cols 
                          if st.session_state.selected_data[col].dtype in ['int64', 'float64']]
        
        default_features = []
        if edit and model_data.get('model param') and len(model_data['model param']) > 0:
            suggested_features = model_data['model param'][0]['value'] if isinstance(model_data['model param'][0]['value'], list) else []
            # Safety filter: ensure recommended features exist in current numeric options
            default_features = [f for f in suggested_features if f in feature_options]
            
        features = st.multiselect(
            "Select feature columns:",
            options=feature_options,
            default=default_features
        )
    
    with col2:
        if len(numeric_cols) == 0:
            st.warning("No numeric columns available for target selection")
            target = None
        else:
            default_index = numeric_cols.index(
                model_data['model param'][1]['value']
            ) if edit else 0
            target = st.selectbox(
                "Select target column:",
                options=numeric_cols,
                index=default_index if edit else len(numeric_cols)-1,
                key="target_column"
            )
    
    # Grid Search Configuration
    st.subheader("Hyperparameter Configuration")
    
    use_grid_search = st.checkbox(
        "Enable Grid Search for automatic parameter tuning",
        value=model_data['model param'][2]['value'] if edit and len(model_data['model param']) > 2 else False,
        help="Automatically find the best hyperparameters using cross-validation"
    )
    
    # Default parameter grid for grid search
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt'],
        'bootstrap': [True, False]
    }
    
    # Default manual parameters
    manual_params = {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'max_features': 'auto',
        'bootstrap': True
    }
    
    cv_folds = 5
    
    if use_grid_search:
        st.info("Grid Search will automatically find the best parameters from predefined ranges")
        
        # Grid search options
        col3, col4 = st.columns(2)
        with col3:
            cv_folds = st.slider(
                "Cross-Validation Folds:",
                min_value=3,
                max_value=10,
                value=model_data['model param'][3]['value'] if edit and len(model_data['model param']) > 3 else 5,
                help="Number of cross-validation folds"
            )
            
            # Customize parameter grid
            st.subheader("Grid Search Ranges")
            
            n_estimators_range = st.multiselect(
                "Number of Trees:",
                options=[10, 50, 100, 200, 300, 500],
                default=model_data['model param'][4]['value'] if edit and len(model_data['model param']) > 4 else [50, 100, 200],
            )
            
            max_depth_range = st.multiselect(
                "Max Depth:",
                options=[3, 5, 10, 15, 20, None],
                default=model_data['model param'][5]['value'] if edit and len(model_data['model param']) > 5 else [5, 10, 15, None],
            )
        
        with col4:
            min_samples_split_range = st.multiselect(
                "Min Samples Split:",
                options=[2, 5, 10, 15, 20],
                default=model_data['model param'][6]['value'] if edit and len(model_data['model param']) > 6 else [2, 5, 10],
            )
            
            min_samples_leaf_range = st.multiselect(
                "Min Samples Leaf:",
                options=[1, 2, 4, 6, 8],
                default=model_data['model param'][7]['value'] if edit and len(model_data['model param']) > 7 else [1, 2, 4],
            )
        
        # Update parameter grid with user selections
        if n_estimators_range:
            param_grid['n_estimators'] = n_estimators_range
        if max_depth_range:
            param_grid['max_depth'] = max_depth_range
        if min_samples_split_range:
            param_grid['min_samples_split'] = min_samples_split_range
        if min_samples_leaf_range:
            param_grid['min_samples_leaf'] = min_samples_leaf_range
    
    else:
        # Manual parameter configuration
        st.subheader("Manual Parameter Configuration")
        col3, col4, col5 = st.columns(3)
        
        with col3:
            manual_params['n_estimators'] = st.slider(
                "Number of Trees:",
                min_value=10,
                max_value=500,
                value=model_data['model param'][8]['value'] if edit and len(model_data['model param']) > 8 else 100,
                help="Number of trees in the forest"
            )
            
            manual_params['max_depth'] = st.slider(
                "Max Depth:",
                min_value=1,
                max_value=50,
                value=model_data['model param'][9]['value'] if edit and len(model_data['model param']) > 9 else 10,
                help="Maximum depth of the trees"
            )
        
        with col4:
            manual_params['min_samples_split'] = st.slider(
                "Min Samples Split:",
                min_value=2,
                max_value=20,
                value=model_data['model param'][10]['value'] if edit and len(model_data['model param']) > 10 else 2,
                help="Minimum number of samples required to split a node"
            )
            
            manual_params['min_samples_leaf'] = st.slider(
                "Min Samples Leaf:",
                min_value=1,
                max_value=20,
                value=model_data['model param'][11]['value'] if edit and len(model_data['model param']) > 11 else 1,
                help="Minimum number of samples required at a leaf node"
            )
        
        with col5:
            max_features_options = ['sqrt', 'log2', None]
            default_max_features_index = max_features_options.index(
                model_data['model param'][12]['value']
            ) if edit and len(model_data['model param']) > 12 else 0
            manual_params['max_features'] = st.selectbox(
                "Max Features:",
                options=max_features_options,
                index=default_max_features_index,
                help="Number of features to consider for the best split"    
            )
            
            bootstrap_default = model_data['model param'][13]['value'] if edit and len(model_data['model param']) > 13 else True
            manual_params['bootstrap'] = st.checkbox(
                "Bootstrap Sampling",
                value=bootstrap_default,
                help="Whether to use bootstrap samples when building trees"
            )
    
    return {
        "features": features,
        "target": target,
        "edit": edit,
        "use_grid_search": use_grid_search,
        "param_grid": param_grid,
        "manual_params": manual_params,
        "cv_folds": cv_folds,
        "df": st.session_state.selected_data
    }

model_description = """
            <div class="code-container">
                <div class="code-header">
                    <span>MODEL DESCRIPTION</span>
                    <span>Random Forest Regressor</span>
                </div>
                <div>
                    Random Forest is an ensemble learning method that constructs multiple decision trees 
                    during training and outputs the average prediction for regression tasks. This implementation 
                    supports both manual parameter configuration and automated hyperparameter tuning using Grid Search.
                    
                    <strong>Key Features:</strong>
                    • Ensemble of decision trees with reduced overfitting
                    • Bootstrap sampling for diversity in trees
                    • Feature importance rankings for interpretability
                    • Robust to outliers and missing values
                    • Parallel processing for faster training
                    
                    <strong>Working Principle:</strong>
                    1. Creates multiple decision trees on random subsets of data (bootstrap samples)
                    2. At each split, considers only a random subset of features
                    3. Combines predictions from all trees by averaging
                    4. Uses out-of-bag samples for internal validation
                    
                    <strong>Hyperparameters:</strong>
                    • <strong>n_estimators:</strong> Number of trees in the forest
                    • <strong>max_depth:</strong> Maximum depth of each tree
                    • <strong>min_samples_split:</strong> Minimum samples required to split a node
                    • <strong>min_samples_leaf:</strong> Minimum samples required at a leaf node
                    • <strong>max_features:</strong> Number of features considered for splits
                    • <strong>bootstrap:</strong> Whether to use bootstrap sampling
                    
                    <strong>Grid Search Capability:</strong>
                    • Automatically tests multiple parameter combinations
                    • Uses cross-validation for robust performance estimation
                    • Returns the best performing parameter set
                    • Reduces manual tuning effort significantly
                    
                    <strong>Advantages:</strong>
                    • Handles non-linear relationships well
                    • Provides feature importance scores
                    • Less prone to overfitting than single decision trees
                    • Works well with both numerical and categorical features
                    • Good performance on medium to large datasets
                </div>
            </div>
            """

model_reference_code = """
            <div class="code-container">
                <div class="code-header">
                    <span>MODEL REFERENCE CODE</span>
                    <span>Random Forest Regressor</span>
                </div>
                <div>
                    # Import required libraries
                    from sklearn.ensemble import RandomForestRegressor
                    from sklearn.model_selection import train_test_split, GridSearchCV
                    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
                    import pandas as pd
                    
                    # Prepare data
                    X = df[features].values
                    y = df[target].values
                    
                    # Split data into training and testing sets
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )
                    
                    # Option 1: Manual Parameter Configuration
                    def train_manual_rf():
                        model = RandomForestRegressor(
                            n_estimators=100,           # Number of trees
                            max_depth=10,               # Maximum tree depth
                            min_samples_split=2,        # Minimum samples to split node
                            min_samples_leaf=1,         # Minimum samples at leaf node
                            max_features='auto',        # Features for best split
                            bootstrap=True,             # Bootstrap sampling
                            random_state=42,            # Reproducibility
                            n_jobs=-1                   # Use all CPU cores
                        )
                        
                        # Train the model
                        model.fit(X_train, y_train)
                        
                        # Make predictions
                        y_pred = model.predict(X_test)
                        
                        # Calculate metrics
                        mse = mean_squared_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        mae = mean_absolute_error(y_test, y_pred)
                        
                        return model, mse, r2, mae, y_pred
                    
                    # Option 2: Grid Search for Automatic Tuning
                    def train_grid_search_rf():
                        # Define parameter grid
                        param_grid = {
                            'n_estimators': [50, 100, 200],
                            'max_depth': [5, 10, 15, None],
                            'min_samples_split': [2, 5, 10],
                            'min_samples_leaf': [1, 2, 4],
                            'max_features': ['auto', 'sqrt'],
                            'bootstrap': [True, False]
                        }
                        
                        # Initialize base model
                        rf = RandomForestRegressor(random_state=42)
                        
                        # Configure Grid Search
                        grid_search = GridSearchCV(
                            estimator=rf,
                            param_grid=param_grid,
                            cv=5,                    # 5-fold cross-validation
                            scoring='r2',           # Optimize for R² score
                            n_jobs=-1,              # Use all available cores
                            verbose=1,              # Show progress
                            return_train_score=True # Return training scores
                        )
                        
                        # Perform Grid Search
                        grid_search.fit(X_train, y_train)
                        
                        # Get best model
                        best_model = grid_search.best_estimator_
                        best_params = grid_search.best_params_
                        best_score = grid_search.best_score_
                        
                        # Make predictions with optimized model
                        y_pred = best_model.predict(X_test)
                        
                        # Calculate metrics
                        mse = mean_squared_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        mae = mean_absolute_error(y_test, y_pred)
                        
                        return best_model, mse, r2, mae, y_pred, best_params, best_score
                    
                    # Train model based on configuration
                    use_grid_search = True  # Set to False for manual mode
                    
                    if use_grid_search:
                        model, mse, r2, mae, y_pred, best_params, best_score = train_grid_search_rf()
                        print(f"Best parameters from Grid Search: {best_params}")
                        print(f"Best cross-validation score: {best_score:.4f}")
                    else:
                        model, mse, r2, mae, y_pred = train_manual_rf()
                    
                    # Display results
                    print(f"R² Score: {r2:.4f}")
                    print(f"Mean Squared Error: {mse:.4f}")
                    print(f"Mean Absolute Error: {mae:.4f}")
                    
                    # Feature importance
                    feature_importance = dict(zip(features, model.feature_importances_))
                    print("\\nFeature Importance:")
                    for feature, importance in sorted(feature_importance.items(), 
                                                    key=lambda x: x[1], reverse=True):
                        print(f"  {feature}: {importance:.4f}")
                    
                    # Store results for further analysis
                    results = {
                        'model': model,
                        'metrics': {
                            'MSE': mse,
                            'R2 Score': r2,
                            'MAE': mae,
                            'Feature Importance': feature_importance
                        },
                        'predictions': y_pred,
                        'best_params': best_params if use_grid_search else None
                    }
                </div>
            </div>
            """