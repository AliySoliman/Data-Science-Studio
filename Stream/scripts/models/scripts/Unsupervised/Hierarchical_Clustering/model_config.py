import streamlit as st

def model_config(model_data, edit):
    """
    Configuration function for hierarchical clustering algorithm
    """
    numeric_cols = [col for col in st.session_state.selected_data.columns 
                   if st.session_state.selected_data[col].dtype in ['int64', 'float64']]
    
    # Feature selection for clustering
    col1, col2 = st.columns(2)
    
    with col1:
        features = st.multiselect(
            "Select features for clustering:",
            options=numeric_cols if len(numeric_cols) != 0 else [],
            default=model_data['model param'][0]['value'] if edit else numeric_cols,
            help="Select numerical columns to use for clustering analysis"
        )
    
    with col2:
        # Number of clusters
        n_clusters = st.number_input(
            "Number of clusters:",
            min_value=2,
            max_value=20,
            value=model_data['model param'][1]['value'] if edit else 3,
            help="Specify the number of clusters to form"
        )
        
        # Linkage method selection
        linkage_methods = ['ward', 'complete', 'average', 'single']
        default_linkage_index = linkage_methods.index(
            model_data['model param'][2]['value']
        ) if edit and 'model param' in model_data and len(model_data['model param']) > 2 else 0
        
        linkage = st.selectbox(
            "Linkage method:",
            options=linkage_methods,
            index=default_linkage_index,
            help="The linkage algorithm to use"
        )
        
        # Distance metric selection
        distance_metrics = ['euclidean', 'manhattan', 'cosine', 'correlation']
        default_metric_index = distance_metrics.index(
            model_data['model param'][3]['value']
        ) if edit and 'model param' in model_data and len(model_data['model param']) > 3 else 0
        
        metric = st.selectbox(
            "Distance metric:",
            options=distance_metrics,
            index=default_metric_index,
            help="The distance metric to use"
        )
    
    # Additional hierarchical clustering parameters
    with st.expander("Advanced Parameters"):
        col3, col4 = st.columns(2)
        
        with col3:
            # Compute full tree
            compute_full_tree = st.checkbox(
                "Compute full tree",
                value=model_data['model param'][4]['value'] if edit and len(model_data['model param']) > 4 else False,
                help="Compute the full tree for faster prediction at the cost of memory"
            )
        
        with col4:
            # Distance threshold (alternative to n_clusters)
            distance_threshold = st.number_input(
                "Distance threshold (optional):",
                min_value=0.0,
                value=model_data['model param'][5]['value'] if edit and len(model_data['model param']) > 5 else None,
                help="The linkage distance threshold above which clusters will not be merged"
            )
    
    return {
        "features": features,
        "n_clusters": n_clusters,
        "linkage": linkage,
        "metric": metric,
        "compute_full_tree": compute_full_tree,
        "distance_threshold": distance_threshold if distance_threshold else None,
        "df": st.session_state.selected_data,
        "edit": edit
    }
model_reference_code = """
            <div class="code-container">
                <div class="code-header">
                    <span>MODEL REFERENCE CODE</span>
                    <span>Hierarchical Clustering</span>
                </div>
                <div>
                    # Prepare data
                    X = df[features].values
                    
                    # Standardize the data
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    # Create and fit the model
                    from sklearn.cluster import AgglomerativeClustering
                    model = AgglomerativeClustering(
                        n_clusters=n_clusters,
                        metric=metric,
                        linkage=linkage,
                        compute_full_tree=compute_full_tree,
                        distance_threshold=distance_threshold
                    )
                    
                    # Make predictions (cluster labels)
                    cluster_labels = model.fit_predict(X_scaled)
                    
                    # Calculate clustering metrics
                    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
                    
                    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
                    calinski_score = calinski_harabasz_score(X_scaled, cluster_labels)
                    davies_score = davies_bouldin_score(X_scaled, cluster_labels)
                    
                    # Store results
                    st.session_state.model_results = {
                        'model': model,
                        'scaler': scaler,
                        'metrics': {
                            'Silhouette Score': silhouette_avg,
                            'Calinski-Harabasz Score': calinski_score,
                            'Davies-Bouldin Score': davies_score,
                            'Number of Clusters': n_clusters,
                            'Cluster Sizes': [sum(cluster_labels == i) for i in range(n_clusters)]
                        },
                        'cluster_labels': cluster_labels,
                        'features': features,
                        'X_scaled': X_scaled
                    }
                </div>
            </div>
            """
model_description = """
            <div class="code-container">
                <div class="code-header">
                    <span>MODEL DESCRIPTION</span>
                    <span>Hierarchical Clustering</span>
                </div>
                <div>
                    Hierarchical clustering is an unsupervised machine learning algorithm that builds a 
                    hierarchy of clusters by either a bottom-up (agglomerative) or top-down (divisive) 
                    approach. The algorithm creates a tree-like structure (dendrogram) that shows the 
                    relationships and similarities between data points at different levels of granularity.
                    
                    <strong>Agglomerative Approach (Bottom-Up):</strong>
                    1. Start with each data point as its own cluster
                    2. Compute distance matrix between all clusters
                    3. Iteratively merge the closest pairs of clusters based on linkage criterion
                    4. Update distance matrix and repeat until all points are in a single cluster
                    
                    <strong>Key Parameters:</strong>
                    • <strong>Linkage Method:</strong> Defines how to measure distance between clusters
                    • <strong>Distance Metric:</strong> How to measure distance between points
                    • <strong>Number of Clusters:</strong> Determines where to cut the dendrogram
                    
                    <strong>Linkage Methods:</strong>
                    • <strong>Ward:</strong> Minimizes variance within clusters (Δ(A,B) = Σ||x - centroid_AB||² - [Σ||x - centroid_A||² + Σ||x - centroid_B||²])
                    • <strong>Complete:</strong> Uses maximum distance between clusters (d(A,B) = max{d(a,b) : a∈A, b∈B})
                    • <strong>Average:</strong> Uses average distance between all pairs (d(A,B) = (1/|A||B|) ΣΣ d(a,b))
                    • <strong>Single:</strong> Uses minimum distance between clusters (d(A,B) = min{d(a,b) : a∈A, b∈B})
                    
                    <strong>Distance Metrics:</strong>
                    • <strong>Euclidean:</strong> √Σ(xᵢ - yᵢ)² (straight-line distance, required for Ward)
                    • <strong>Manhattan:</strong> Σ|xᵢ - yᵢ| (city-block distance)
                    • <strong>Cosine:</strong> 1 - (x·y)/(||x||·||y||) (angle-based similarity)
                    
                    The dendrogram helps determine the optimal number of clusters by identifying 
                    the largest vertical distance that doesn't intersect any horizontal line.
                </div>
            </div>
            """