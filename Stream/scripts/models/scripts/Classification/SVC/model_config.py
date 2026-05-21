import streamlit as st
import pandas as pd

def model_config(model_data, edit):        
    numeric_cols = [col for col in st.session_state.selected_data.columns if st.session_state.selected_data[col].dtype in ['int64', 'float64']] 
    
    # Feature and target selection
    all_cols = st.session_state.selected_data.columns.tolist()
    col1, col2 = st.columns(2)
    with col1:
        # Filter numeric columns for features
        feature_options = [col for col in all_cols if st.session_state.selected_data[col].dtype in ['int64', 'float64']]
        
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
            "Select feature columns:",
            options=feature_options,
            default=default_features
        )
    
    with col2:
        if len(numeric_cols) == 0:
            st.warning("No numeric columns available for target selection")
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
                
                if target_value and target_value in numeric_cols:
                    default_index = numeric_cols.index(target_value)
            
            target = st.selectbox(
                "Select target column:",
                options=numeric_cols,
                index=default_index,
                key="target_column_svc"
            )    
    
    # Support Vector Machine Hyperparameter Configuration
    st.subheader("🎯 Support Vector Machine Hyperparameters")
    
    col3, col4 = st.columns(2)
    
    with col3:
        C = st.slider(
            "Regularization Parameter (C):",
            min_value=0.1,
            max_value=10.0,
            value=1.0,
            step=0.1,
            help="Regularization parameter. The strength of the regularization is inversely proportional to C."
        )
        
        kernel = st.selectbox(
            "Kernel Type:",
            options=['linear', 'poly', 'rbf', 'sigmoid'],
            index=2,
            help="Specifies the kernel type to be used in the algorithm"
        )
    
    with col4:
        gamma = st.selectbox(
            "Kernel Coefficient (Gamma):",
            options=['scale', 'auto', 'custom'],
            index=0,
            help="Kernel coefficient for 'rbf', 'poly' and 'sigmoid'"
        )
        
        degree = st.slider(
            "Degree (for polynomial kernel):",
            min_value=2,
            max_value=6,
            value=3,
            help="Degree of the polynomial kernel function ('poly')"
        )
    
    # Custom gamma input if selected
    custom_gamma = None
    if gamma == 'custom':
        custom_gamma = st.slider(
            "Custom Gamma Value:",
            min_value=0.001,
            max_value=10.0,
            value=0.1,
            step=0.001,
            format="%.3f",
            help="Custom value for gamma parameter"
        )
    
    # Prepare parameters for model script
    manual_params = {
        'C': C,
        'kernel': kernel,
        'gamma': custom_gamma if gamma == 'custom' else gamma,
        'degree': degree
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
        <span>Support Vector Machine Classifier</span>
    </div>
    <div>
        # Support Vector Machine Classifier Implementation
        from sklearn.svm import SVC
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder, StandardScaler
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        
        # Prepare data
        X = df[features].values
        y = df[target].values
        
        # Encode target if categorical
        if df[target].dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        # Scale features for SVM
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Train SVM model
        model = SVC(
            C=1.0,
            kernel='rbf',
            gamma='scale',
            degree=3,
            probability=True,
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
            'scaler': scaler,
            'metrics': {
                'Accuracy': accuracy,
                'Classification Report': class_report,
                'Confusion Matrix': conf_matrix
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
        <span>Support Vector Machine Classifier</span>
    </div>
    <div>
        Support Vector Machine (SVM) is a powerful supervised learning algorithm used for 
        classification tasks. It works by finding the optimal hyperplane that separates 
        classes in the feature space with the maximum margin.
        
        <strong>Key Advantages:</strong>
        • Effective in high-dimensional spaces
        • Memory efficient (uses subset of training points)
        • Versatile through different kernel functions
        • Robust against overfitting in high-dimensional space
        
        <strong>Key Evaluation Metrics:</strong>
        • <strong>Accuracy:</strong> Overall classification correctness
        • <strong>Precision:</strong> Ratio of true positives to predicted positives
        • <strong>Recall:</strong> Ratio of true positives to actual positives  
        • <strong>F1-Score:</strong> Harmonic mean of precision and recall
        
        <strong>Key Parameters:</strong>
        • <strong>C:</strong> Regularization parameter - controls trade-off between decision boundary smoothness and training points classification
        • <strong>kernel:</strong> Specifies the kernel type (linear, poly, rbf, sigmoid)
        • <strong>gamma:</strong> Kernel coefficient for 'rbf', 'poly' and 'sigmoid'
        • <strong>degree:</strong> Degree of the polynomial kernel function
    </div>
</div>
"""