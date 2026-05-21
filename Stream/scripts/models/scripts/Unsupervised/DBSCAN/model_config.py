import streamlit as st
import pandas as pd
import numpy as np

def model_config(model_data, edit):
    # Always define a default cv_folds (even if DBSCAN doesn't truly use it)
    default_cv_folds = 5

    # If edit mode has saved cv_folds, reuse it
    if edit and model_data.get('model param'):
        for param in model_data['model param']:
            if param.get('name') == 'cv_folds':
                try:
                    default_cv_folds = int(param.get('value', 5))
                except Exception:
                    default_cv_folds = 5
                break

    cv_folds = default_cv_folds

    # Check if selected_data exists and has columns
    if 'selected_data' not in st.session_state or st.session_state.selected_data.empty:
        st.error("No data available. Please load data first.")
        return {
            "features": [],
            "df": pd.DataFrame(),
            "edit": edit,
            "use_grid_search": True,
            "param_grid": {},
            "manual_params": {},
            "cv_folds": cv_folds,   # ✅ added
        }

    numeric_cols = [
        col for col in st.session_state.selected_data.columns
        if st.session_state.selected_data[col].dtype in ['int64', 'float64']
    ]

    # Feature selection (no target for unsupervised)
    st.markdown("#### Feature Selection")

    # Handle default features for edit mode
    all_cols = st.session_state.selected_data.columns.tolist()
    default_features = []
    if edit and model_data.get('model param'):
        for param in model_data['model param']:
            if param['name'] == 'features':
                suggested_features = param['value'] if isinstance(param['value'], list) else []
                # Safety filter: ensure recommended features exist in current numeric options
                default_features = [f for f in suggested_features if f in numeric_cols]
                break

    features = st.multiselect(
        "Select feature columns:",
        options=numeric_cols if numeric_cols else [],
        default=default_features,
        help="Select numerical features for clustering"
    )

    # Model Configuration
    st.subheader("Model Configuration")

    # Default values
    default_use_grid_search = True

    # Extract parameters from model_data if in edit mode
    if edit and model_data.get('model param'):
        param_dict = {}
        for param in model_data['model param']:
            param_dict[param['name']] = param['value']

        default_use_grid_search = param_dict.get('use_grid_search', True)
        # keep cv_folds consistent if present
        try:
            cv_folds = int(param_dict.get('cv_folds', cv_folds))
        except Exception:
            cv_folds = cv_folds

    # Grid Search Configuration
    use_grid_search = st.checkbox(
        "Use Grid Search for hyperparameter tuning",
        value=default_use_grid_search
    )

    if use_grid_search:
        st.info("Grid Search will automatically find the best hyperparameters for DBSCAN")

        # (Optional UI) Keep cv_folds visible and always defined
        cv_folds = st.number_input(
            "Cross-validation folds (kept for pipeline compatibility):",
            min_value=2,
            max_value=10,
            value=int(cv_folds),
            help="DBSCAN grid search uses internal evaluation, but this is kept for pipeline consistency."
        )

        col3, col4 = st.columns(2)
        with col3:
            eps_range = st.text_input(
                "Eps values (comma-separated):",
                value="0.3,0.5,0.7,1.0",
                help="Maximum distance between two samples to be considered neighbors"
            )
            try:
                eps_values = [float(x.strip()) for x in eps_range.split(',')]
            except (ValueError, TypeError):
                eps_values = [0.3, 0.5, 0.7, 1.0]
                st.warning("Invalid format. Using default eps values")

        with col4:
            min_samples_range = st.text_input(
                "Min Samples (comma-separated):",
                value="3,5,7,10",
                help="Minimum number of samples in a neighborhood to form a core point"
            )
            try:
                min_samples_values = [int(x.strip()) for x in min_samples_range.split(',')]
            except (ValueError, TypeError):
                min_samples_values = [3, 5, 7, 10]
                st.warning("Invalid format. Using default min_samples values")

        col5, col6 = st.columns(2)
        with col5:
            metric_options = st.multiselect(
                "Distance Metrics:",
                options=['euclidean', 'manhattan', 'chebyshev', 'minkowski'],
                default=['euclidean'],
                help="Distance metric to use for calculating neighbors"
            )
            if not metric_options:
                metric_options = ['euclidean']

        param_grid = {
            'eps': eps_values,
            'min_samples': min_samples_values,
            'metric': metric_options
        }

        manual_params = {}

    else:
        # Manual parameter configuration
        st.info("Configure hyperparameters manually for DBSCAN")

        # Keep cv_folds defined (even if unused)
        cv_folds = int(cv_folds)

        col3, col4, col5 = st.columns(3)
        with col3:
            eps = st.number_input(
                "Eps:",
                min_value=0.01,
                max_value=10.0,
                value=0.5,
                step=0.1,
                format="%.2f",
                help="Maximum distance between two samples"
            )
        with col4:
            min_samples = st.number_input(
                "Min Samples:",
                min_value=1,
                max_value=100,
                value=5,
                step=1,
                help="Minimum samples in a neighborhood"
            )
        with col5:
            metric = st.selectbox(
                "Metric:",
                options=['euclidean', 'manhattan', 'chebyshev', 'minkowski'],
                index=0,
                help="Distance metric to use"
            )

        col6, col7 = st.columns(2)
        with col6:
            algorithm = st.selectbox(
                "Algorithm:",
                options=['auto', 'ball_tree', 'kd_tree', 'brute'],
                index=0,
                help="Algorithm to compute nearest neighbors"
            )
        with col7:
            leaf_size = st.number_input(
                "Leaf Size:",
                min_value=10,
                max_value=100,
                value=30,
                step=5,
                help="Leaf size for BallTree or KDTree"
            )

        manual_params = {
            'eps': eps,
            'min_samples': min_samples,
            'metric': metric,
            'algorithm': algorithm,
            'leaf_size': leaf_size
        }
        param_grid = {}

    return {
        "features": features,
        "df": st.session_state.selected_data,
        "edit": edit,
        "use_grid_search": use_grid_search,
        "param_grid": param_grid,
        "manual_params": manual_params,
        "cv_folds": int(cv_folds),   # ✅ added
    }
    
model_description = """
<div class="code-container">
    <div class="code-header">
        <span>MODEL DESCRIPTION</span>
        <span>DBSCAN</span>
    </div>
    <div>
        DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a 
        density-based clustering algorithm that groups together points that are closely 
        packed together, marking points in low-density regions as outliers.
        
        <strong>Key Features:</strong>
        • Density-based clustering method
        • Automatically determines number of clusters
        • Can identify outliers as noise points
        • Does not require specifying number of clusters beforehand
        • Works well with arbitrary shaped clusters
        
        <strong>Key Evaluation Metrics:</strong>
        • <strong>Silhouette Score:</strong> Measures cluster cohesion and separation (-1 to 1, higher is better)
        • <strong>Davies-Bouldin Index:</strong> Average similarity between clusters (lower is better)
        • <strong>Calinski-Harabasz Score:</strong> Ratio of between-cluster to within-cluster dispersion (higher is better)
        • <strong>N Clusters:</strong> Number of clusters found
        • <strong>N Noise Points:</strong> Number of outliers detected
        
        <strong>Hyperparameters:</strong>
        • <strong>eps:</strong> Maximum distance between two samples to be neighbors (0.1-2.0 typical)
        • <strong>min_samples:</strong> Minimum samples in neighborhood to form core point (3-10 typical)
        • <strong>metric:</strong> Distance metric (euclidean, manhattan, etc.)
        • <strong>algorithm:</strong> Algorithm for neighbor search (auto, ball_tree, kd_tree, brute)
        • <strong>leaf_size:</strong> Leaf size for tree algorithms (10-100)
    </div>
</div>
"""

model_reference_code = """
<div class="code-container">
    <div class="code-header">
        <span>REFERENCE CODE</span>
        <span>DBSCAN Implementation</span>
    </div>
    <div>
        <pre style="background: #f4f4f4; padding: 10px; border-radius: 5px; overflow-x: auto;">
<code>
# DBSCAN Implementation
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Load and prepare data
data = pd.read_csv('your_data.csv')
X = data[['feature1', 'feature2', 'feature3']]

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Model and Grid Search
best_score = -1
best_params = None

for eps in [0.3, 0.5, 0.7, 1.0]:
    for min_samples in [3, 5, 7, 10]:
        model = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
        labels = model.fit_predict(X_scaled)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        if n_clusters >= 2 and n_noise < len(labels):
            mask = labels != -1
            if sum(mask) > 1:
                score = silhouette_score(X_scaled[mask], labels[mask])
                if score > best_score:
                    best_score = score
                    best_params = {'eps': eps, 'min_samples': min_samples}
                    best_labels = labels

# Best model
best_model = DBSCAN(**best_params)
labels = best_model.fit_predict(X_scaled)

# Calculate metrics
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)
mask = labels != -1

silhouette = silhouette_score(X_scaled[mask], labels[mask])
davies_bouldin = davies_bouldin_score(X_scaled[mask], labels[mask])
calinski_harabasz = calinski_harabasz_score(X_scaled[mask], labels[mask])

print(f"N Clusters: {n_clusters}")
print(f"N Noise Points: {n_noise}")
print(f"Silhouette Score: {silhouette:.4f}")
print(f"Davies-Bouldin Index: {davies_bouldin:.4f}")
print(f"Calinski-Harabasz Score: {calinski_harabasz:.4f}")
print("Best parameters:", best_params)
</code>
        </pre>
    </div>
</div>
"""