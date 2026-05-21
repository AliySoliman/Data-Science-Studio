import streamlit as st
import pandas as pd

def model_config(model_data, edit):        
    numeric_cols = [col for col in st.session_state.selected_data.columns if st.session_state.selected_data[col].dtype in ['int64', 'float64']] 
    
    # Feature selection only (unsupervised - no target)
    st.subheader("Feature Selection for Clustering")
    
    # Handle default features for edit mode
    default_features = []
    if edit and model_data.get('model param'):
        for param in model_data['model param']:
            if param['name'] == 'features':
                default_features = param['value'] if isinstance(param['value'], list) else []
                break
    
    features = st.multiselect(
        "Select feature columns for clustering:",
        options=[col for col in st.session_state.selected_data.columns if st.session_state.selected_data[col].dtype in ['int64', 'float64']] if len(numeric_cols) != 0 else [],
        default=default_features
    )
    
    # Clustering Configuration
    st.subheader("🎯 Clustering Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Number of clusters selection
        n_samples = len(st.session_state.selected_data) if features else 0
        suggested_clusters = min(10, max(2, n_samples // 10)) if n_samples > 0 else 3
        
        # Get default n_clusters from model_data if in edit mode
        default_n_clusters = suggested_clusters
        if edit and model_data.get('model param'):
            for param in model_data['model param']:
                if param['name'] == 'n_clusters':
                    try:
                        default_n_clusters = int(param['value'])
                    except (ValueError, TypeError):
                        default_n_clusters = suggested_clusters
                    break
        
        n_clusters = st.slider(
            "Number of clusters (k):",
            min_value=2,
            max_value=min(20, n_samples) if n_samples > 0 else 10,
            value=default_n_clusters,
            help="Select the number of clusters to identify in the data"
        )
    
    with col2:
        # KNN parameters
        n_neighbors = st.slider(
            "Number of Neighbors:",
            min_value=3,
            max_value=20,
            value=5,
            help="Number of neighbors to use for KNN graph construction"
        )
        
        metric = st.selectbox(
            "Distance Metric:",
            options=['euclidean', 'manhattan', 'cosine'],
            index=0,
            help="Distance metric used for KNN calculations"
        )
    
    # Algorithm selection
    algorithm = st.selectbox(
        "KNN Algorithm:",
        options=['auto', 'ball_tree', 'kd_tree', 'brute'],
        index=0,
        help="Algorithm used to compute nearest neighbors"
    )
    
    # Clustering method selection
    clustering_method = st.selectbox(
        "Clustering Method:",
        options=['Spectral Clustering', 'DBSCAN'],
        index=0,
        help="Choose the clustering algorithm to use with KNN"
    )
    
    # Prepare parameters for model script
    manual_params = {
        'n_neighbors': n_neighbors,
        'metric': metric,
        'algorithm': algorithm,  # ADD THIS LINE
        'clustering_method': clustering_method
    }
    
    return {
        "features": features, 
        "n_clusters": n_clusters, 
        "df": st.session_state.selected_data, 
        "edit": edit,
        "use_grid_search": False,
        "param_grid": {},
        "manual_params": manual_params,
        "cv_folds": 5  # ADD THIS FOR CONSISTENCY
    }

model_reference_code = """
<div class="code-container">
    <div class="code-header">
        <span>REFERENCE CODE</span>
        <span>Unsupervised KNN Clustering</span>
    </div>
    <div>
        # Unsupervised KNN Clustering Implementation
        from sklearn.neighbors import NearestNeighbors
        from sklearn.cluster import SpectralClustering
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import silhouette_score
        
        # Prepare data
        X = df[features].values
        
        # Scale features for clustering
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Create KNN graph
        nn = NearestNeighbors(n_neighbors=5, algorithm='auto', metric='euclidean')
        nn.fit(X_scaled)
        adjacency_matrix = nn.kneighbors_graph(X_scaled, mode='connectivity')
        
        # Apply spectral clustering with KNN graph
        spectral = SpectralClustering(
            n_clusters=n_clusters, 
            affinity='precomputed',
            random_state=42
        )
        clusters = spectral.fit_predict(adjacency_matrix)
        
        # Calculate clustering metrics
        silhouette = silhouette_score(X_scaled, clusters)
        
        # Store results
        st.session_state.model_results = {
            'clusters': clusters,
            'scaler': scaler,
            'metrics': {
                'Silhouette Score': silhouette,
                'Number of Clusters': n_clusters
            },
            'features': features,
            'n_clusters': n_clusters
        }
    </div>
</div>
"""

model_description = """
<div class="code-container">
    <div class="code-header">
        <span>MODEL DESCRIPTION</span>
        <span>Unsupervised KNN Clustering</span>
    </div>
    <div>
        Unsupervised K-Nearest Neighbors is used for clustering tasks where we want to 
        group similar data points together without predefined labels. This approach 
        uses KNN concepts to identify natural clusters in the data.
        
        <strong>Clustering Approach:</strong>
        • Uses KNN to compute density and connectivity
        • Identifies clusters based on local data density
        • Can handle clusters of arbitrary shapes
        • Works well with noise and outliers
        
        <strong>Key Evaluation Metrics:</strong>
        • <strong>Silhouette Score:</strong> Measures how similar an object is to its own cluster vs other clusters
        • <strong>Calinski-Harabasz Index:</strong> Ratio of between-cluster dispersion to within-cluster dispersion
        • <strong>Davies-Bouldin Index:</strong> Average similarity measure of each cluster with its most similar cluster
        
        <strong>Key Parameters:</strong>
        • <strong>n_neighbors:</strong> Number of neighbors to consider for density estimation
        • <strong>n_clusters:</strong> Number of clusters to identify
        • <strong>metric:</strong> Distance metric used for neighbor calculations
        • <strong>algorithm:</strong> Algorithm used to compute nearest neighbors
        • <strong>clustering_method:</strong> Algorithm used for final clustering (Spectral, DBSCAN)
        
        This implementation uses KNN concepts combined with clustering algorithms 
        to identify natural groupings in your data.
    </div>
</div>
"""