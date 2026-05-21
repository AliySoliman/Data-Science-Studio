import streamlit as st
import pandas as pd

def model_config(model_data, edit):        
    # Get all columns for both features and targets
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
                    # Safety filter: ensure recommended features exist in current data
                    default_features = [f for f in suggested_features if f in feature_options]
                    break
        
        features = st.multiselect(
            "Select feature columns (Numerical or Categorical):",
            options=feature_options,
            default=default_features
        )
    
    with col2:
        # For classification, target can be categorical or numeric
        target_options = [col for col in all_cols if col not in features]
        if len(target_options) == 0:
            st.warning("No columns available for target selection")
            target = None
        else:
            # Safe index calculation for selectbox
            default_index = 0
            if edit and model_data.get('model param'):
                target_value = None
                for param in model_data['model param']:
                    if param['name'] == 'target':
                        target_value = param['value']
                        break
                
                if target_value and target_value in target_options:
                    default_index = target_options.index(target_value)
            
            target = st.selectbox(
                "Select target column (Categorical/Numeric):",
                options=target_options,
                index=default_index,
                key="target_column_rf"
            )    
    
    # Random Forest Hyperparameter Configuration
    st.subheader("🎯 Random Forest Hyperparameters")
    
    col3, col4 = st.columns(2)
    
    with col3:
        n_estimators = st.slider(
            "Number of Trees (n_estimators):",
            min_value=10,
            max_value=500,
            value=100,
            help="Number of trees in the forest"
        )
        
        max_depth = st.selectbox(
            "Max Depth:",
            options=[3, 5, 10, 15, 20, None],
            index=2,
            help="Maximum depth of the trees (None for unlimited)"
        )
    
    with col4:
        min_samples_split = st.slider(
            "Min Samples Split:",
            min_value=2,
            max_value=20,
            value=2,
            help="Minimum number of samples required to split a node"
        )
        
        min_samples_leaf = st.slider(
            "Min Samples Leaf:",
            min_value=1,
            max_value=10,
            value=1,
            help="Minimum number of samples required at a leaf node"
        )
    
    bootstrap = st.checkbox(
        "Bootstrap Sampling",
        value=True,
        help="Whether bootstrap samples are used when building trees"
    )
    
    # Prepare parameters for model script
    manual_params = {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'bootstrap': bootstrap
    }
    
    return {
        "features": features, 
        "target": target, 
        "df": st.session_state.selected_data, 
        "edit": edit,
        "use_grid_search": False,
        "param_grid": {},
        "manual_params": manual_params,
        "cv_folds": 5
    }

model_reference_code = """
<div class="code-container">
    <div class="code-header">
        <span>REFERENCE CODE</span>
        <span>Random Forest Classifier</span>
    </div>
    <div>
        # Random Forest Classifier Implementation
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        
        # Prepare data
        X = df[features].values
        y = df[target].values
        
        # Encode target if categorical
        if df[target].dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Train Random Forest model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            bootstrap=True,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Store results
        st.session_state.model_results = {
            'model': model,
            'metrics': {
                'Accuracy': accuracy,
                'Classification Report': class_report,
                'Confusion Matrix': conf_matrix,
                'Feature Importances': model.feature_importances_
            },
            'features': features,
            'target': target
        }
    </div>
</div>
"""

model_description = """
<div class="code-container">
    <div class="code-header">
        <span>MODEL DESCRIPTION</span>
        <span>Random Forest Classifier</span>
    </div>
    <div>
        Random Forest is an ensemble learning method that constructs multiple decision trees 
        during training and outputs the mode of the classes (classification) of the individual trees.
        
        <strong>Key Advantages:</strong>
        • Handles high dimensionality well
        • Provides feature importance rankings
        • Resistant to overfitting
        • Works with both numerical and categorical data
        
        <strong>Key Evaluation Metrics:</strong>
        • <strong>Accuracy:</strong> Overall classification correctness
        • <strong>Precision:</strong> Ratio of true positives to predicted positives
        • <strong>Recall:</strong> Ratio of true positives to actual positives  
        • <strong>F1-Score:</strong> Harmonic mean of precision and recall
        
        <strong>Key Parameters:</strong>
        • <strong>n_estimators:</strong> Number of trees in the forest
        • <strong>max_depth:</strong> Maximum depth of each tree
        • <strong>min_samples_split:</strong> Minimum samples required to split a node
        • <strong>min_samples_leaf:</strong> Minimum samples required at a leaf node
        • <strong>bootstrap:</strong> Whether bootstrap samples are used
    </div>
</div>
"""