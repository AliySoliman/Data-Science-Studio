import streamlit as st
import pandas as pd

def model_config(model_data, edit):
    # Check if selected_data exists and has columns
    if 'selected_data' not in st.session_state or st.session_state.selected_data.empty:
        st.error("No data available. Please load data first.")
        return {
            "features": [],
            "k_method": "auto",
            "auto_k": True,
            "manual_k": 3,
            "max_k": 10,
            "df": pd.DataFrame()
        }
    
    # Get numeric columns only for clustering
    numeric_cols = [col for col in st.session_state.selected_data.columns 
                   if st.session_state.selected_data[col].dtype in ['int64', 'float64']]
    
    # Feature selection
    st.subheader("Feature Selection")
    
    # Handle default features for edit mode
    default_features = []
    if edit and model_data.get('model param'):
        for param in model_data['model param']:
            if param['name'] == 'features':
                default_features = param['value'] if isinstance(param['value'], list) else []
                break
    
    features = st.multiselect(
        "Select numeric features for clustering (choose at least 2):",
        options=numeric_cols if numeric_cols else [],
        default=default_features,
        help="K-Means requires numeric features. Select at least 2 features for clustering."
    )
    
    # K Selection Method
    st.subheader("Cluster Configuration")
    
    # Default values
    default_k_method = "auto"
    default_manual_k = 3
    default_max_k = 10
    
    # Extract parameters from model_data if in edit mode
    if edit and model_data.get('model param'):
        param_dict = {}
        for param in model_data['model param']:
            param_dict[param['name']] = param['value']
        
        default_k_method = param_dict.get('k_method', 'auto')
        default_manual_k = int(param_dict.get('manual_k', 3))
        default_max_k = int(param_dict.get('max_k', 10))
    
    k_method = st.radio(
        "K Selection Method:",
        options=["auto", "manual"],
        format_func=lambda x: "Auto-detect (Elbow Method)" if x == "auto" else "Manual K selection",
        index=0 if default_k_method == "auto" else 1
    )
    
    if k_method == "auto":
        st.info("Elbow Method will automatically determine the optimal number of clusters")
        max_k = st.slider(
            "Maximum K to test:",
            min_value=3,
            max_value=min(20, len(st.session_state.selected_data) // 2),
            value=default_max_k,
            help="Maximum number of clusters to test in Elbow Method"
        )
        manual_k = 3  # Default value, not used in auto mode
    else:
        st.info("Manually specify the number of clusters")
        manual_k = st.slider(
            "Number of clusters (K):",
            min_value=2,
            max_value=min(15, len(st.session_state.selected_data) // 2),
            value=default_manual_k,
            help="Number of clusters to create"
        )
        max_k = 10  # Default value, not used in manual mode
    
    # Show data preview
    if features:
        st.subheader("Data Preview")
        st.write(f"Selected {len(features)} features")
        st.dataframe(st.session_state.selected_data[features].head(), use_container_width=True)
        
        # Show basic statistics
        st.write("Feature Statistics:")
        st.dataframe(st.session_state.selected_data[features].describe(), use_container_width=True)
    
    return {
        "features": features,
        "k_method": k_method,
        "auto_k": k_method == "auto",
        "manual_k": manual_k,
        "max_k": max_k,
        "df": st.session_state.selected_data,
        "edit": edit
    }   

model_description = """
<div class="code-container">
    <div class="code-header">
        <span>MODEL DESCRIPTION</span>
        <span>K-Means Clustering Algorithm</span>
    </div>
    <div>
        K-Means is an unsupervised clustering algorithm that partitions data into K distinct clusters
        based on feature similarity. It aims to minimize the within-cluster variance (inertia).
        
        <strong>Key Evaluation Metrics:</strong>
        • <strong>Inertia:</strong> Sum of squared distances of samples to their closest cluster center
        • <strong>Silhouette Score:</strong> Measures how similar an object is to its own cluster vs other clusters (-1 to 1)
        • <strong>Calinski-Harabasz Index:</strong> Ratio of between-clusters dispersion to within-cluster dispersion
        • <strong>Davies-Bouldin Index:</strong> Average similarity measure of each cluster with its most similar cluster (lower is better)
        
        <strong>K Selection Methods:</strong>
        • <strong>Auto (Elbow Method):</strong> Automatically detects optimal K by finding the "elbow" point
        • <strong>Manual:</strong> User specifies the number of clusters
        
        <strong>Interpretation Guidelines:</strong>
        • <strong>Silhouette Score:</strong> >0.7 (Strong), 0.5-0.7 (Reasonable), 0.25-0.5 (Weak), <0.25 (No structure)
        • <strong>Davies-Bouldin:</strong> Lower values indicate better clustering
        • <strong>Calinski-Harabasz:</strong> Higher values indicate better clustering
    </div>
</div>
"""
model_reference_code = """
<div class="code-container">
    <div class="code-header">
        <span>REFERENCE CODE</span>
        <span>K-Means Clustering Implementation</span>
    </div>
    <div>
        <pre style="background: #f4f4f4; padding: 10px; border-radius: 5px; overflow-x: auto;">
<code>
# K-Means Clustering Implementation
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt

def detect_elbow(inertia_values):
    \"\"\"Automatically detect elbow point for optimal K\"\"\"
    x = np.arange(len(inertia_values))
    y = np.array(inertia_values)
    p1 = np.array([x[0], y[0]])
    p2 = np.array([x[-1], y[-1]])
    distances = []
    for i in range(len(x)):
        p = np.array([x[i], y[i]])
        distance = np.abs(np.cross(p2 - p1, p1 - p)) / np.linalg.norm(p2 - p1)
        distances.append(distance)
    return x[np.argmax(distances)] + 2

# Load and prepare data
data = pd.read_csv('your_data.csv')
numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
X = data[numeric_features].values

# Handle missing values
X = np.nan_to_num(X, nan=np.nanmean(X, axis=0))

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Auto K detection using Elbow Method
inertia_values = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia_values.append(kmeans.inertia_)

optimal_k = detect_elbow(inertia_values)
print(f"Optimal K: {optimal_k}")

# Fit final model
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# Evaluation metrics
silhouette_avg = silhouette_score(X_scaled, clusters)
calinski_harabasz = calinski_harabasz_score(X_scaled, clusters)
davies_bouldin = davies_bouldin_score(X_scaled, clusters)

print(f"Silhouette Score: {silhouette_avg:.4f}")
print(f"Calinski-Harabasz Index: {calinski_harabasz:.4f}")
print(f"Davies-Bouldin Index: {davies_bouldin:.4f}")

# Save results
data['Cluster'] = clusters
data.to_csv('clustered_data.csv', index=False)
</code>
        </pre>
    </div>
</div>
"""