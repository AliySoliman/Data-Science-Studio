import streamlit as st
import numpy as np
import pandas as pd

def model_config(model_data, edit):
    # Check if selected_data exists and has columns
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
    
    numeric_cols = [col for col in st.session_state.selected_data.columns 
                   if st.session_state.selected_data[col].dtype in ['int64', 'float64']]
    
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
                key="target_column"
            )
    
    # Grid Search Configuration
    st.subheader("Hyperparameter Configuration")
    
    # Default values
    default_use_grid_search = True
    default_n_neighbors = "1,3,5,7,9,11"
    default_cv_folds = 5
    default_manual_n = 5
    default_weights = 'uniform'
    default_algorithm = 'auto'
    
    # Extract parameters from model_data if in edit mode
    if edit and model_data.get('model param'):
        param_dict = {}
        for param in model_data['model param']:
            param_dict[param['name']] = param['value']
        
        default_use_grid_search = param_dict.get('use_grid_search', True)
        default_n_neighbors = param_dict.get('n_neighbors_range', "1,3,5,7,9,11")
        default_cv_folds = int(param_dict.get('cv_folds', 5))
        
        # Handle manual parameters - ensure they are proper types
        manual_n_str = param_dict.get('n_neighbors_range', '5')
        try:
            default_manual_n = int(manual_n_str)
        except (ValueError, TypeError):
            default_manual_n = 5
        
        default_weights = param_dict.get('weights', 'uniform')
        default_algorithm = param_dict.get('algorithm', 'auto')
    
    use_grid_search = st.checkbox(
        "Use Grid Search for hyperparameter tuning",
        value=default_use_grid_search
    )
    
    if use_grid_search:
        st.info("Grid Search will automatically find the best hyperparameters")
        
        # Grid Search parameters
        col3, col4 = st.columns(2)
        with col3:
            n_neighbors_options = st.text_input(
                "Neighbors range (comma-separated or range like 1-10):",
                value=default_n_neighbors
            )
        
        with col4:
            cv_folds = st.number_input(
                "Cross-validation folds:",
                min_value=2,
                max_value=10,
                value=default_cv_folds
            )
        
        # Parse n_neighbors options
        try:
            if '-' in n_neighbors_options:
                start, end = map(int, n_neighbors_options.split('-'))
                n_neighbors_range = list(range(start, end + 1))
            else:
                n_neighbors_range = [int(x.strip()) for x in n_neighbors_options.split(',')]
        except (ValueError, TypeError):
            n_neighbors_range = [1, 3, 5, 7, 9, 11]
            st.warning("Invalid format. Using default range: 1,3,5,7,9,11")
        
        weights_options = ['uniform', 'distance']
        algorithm_options = ['auto', 'ball_tree', 'kd_tree', 'brute']
        
        param_grid = {
            'n_neighbors': n_neighbors_range,
            'weights': weights_options,
            'algorithm': algorithm_options
        }
        
        manual_params = {}
        
    else:
        # Manual parameter configuration
        st.info("Configure hyperparameters manually")
        
        col3, col4, col5 = st.columns(3)
        
        with col3:
            n_neighbors = st.number_input(
                "Number of neighbors (n_neighbors):",
                min_value=1,
                max_value=50,
                value=int(default_manual_n)  # Ensure it's integer
            )
        
        with col4:
            weights = st.selectbox(
                "Weight function:",
                options=['uniform', 'distance'],
                index=0 if default_weights == 'uniform' else 1
            )
        
        with col5:
            algorithm = st.selectbox(
                "Algorithm:",
                options=['auto', 'ball_tree', 'kd_tree', 'brute'],
                index=['auto', 'ball_tree', 'kd_tree', 'brute'].index(default_algorithm) 
                if default_algorithm in ['auto', 'ball_tree', 'kd_tree', 'brute'] else 0
            )
        
        manual_params = {
            'n_neighbors': int(n_neighbors),  # Ensure integer type
            'weights': weights,
            'algorithm': algorithm
        }
        param_grid = {}
        cv_folds = 5
    
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
model_reference_code = """
<div class="code-container">
    <div class="code-header">
        <span>REFERENCE CODE</span>
        <span>KNN Classifier Implementation</span>
    </div>
    <div>
        <pre style="background: #f4f4f4; padding: 10px; border-radius: 5px; overflow-x: auto;">
<code>
# KNN Classifier Implementation
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
# Load and prepare data
data = pd.read_csv('your_data.csv')
X = data[['feature1', 'feature2', 'feature3']]
y = data['target']
# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
le = LabelEncoder()
y_encoded = le.fit_transform(y)
# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2)
# With Grid Search
param_grid = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree']
}
knn = KNeighborsClassifier()
grid_search = GridSearchCV(knn, param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_knn = grid_search.best_estimator_
y_pred = best_knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("Best parameters:", grid_search.best_params_)
    </div>
</div>
"""
model_description = """
<div class="code-container">
    <div class="code-header">
        <span>MODEL DESCRIPTION</span>
        <span>K-Nearest Neighbors Classifier</span>
    </div>
    <div>
        K-Nearest Neighbors (KNN) is a simple, instance-based learning algorithm used for 
        classification tasks. It classifies new data points based on the majority class 
        among their k-nearest neighbors in the feature space.
        
        <strong>Key Evaluation Metrics:</strong>
        • <strong>Accuracy:</strong> Overall classification correctness
        • <strong>Precision:</strong> Ratio of true positives to predicted positives
        • <strong>Recall:</strong> Ratio of true positives to actual positives  
        • <strong>F1-Score:</strong> Harmonic mean of precision and recall
        
        <strong>Hyperparameters:</strong>
        • <strong>n_neighbors:</strong> Number of neighbors (k value)
        • <strong>weights:</strong> 'uniform' or 'distance' based weighting
        • <strong>algorithm:</strong> Algorithm used to compute nearest neighbors
        
        Grid Search automatically finds the optimal combination of these parameters 
        through cross-validation.
    </div>
</div>
"""