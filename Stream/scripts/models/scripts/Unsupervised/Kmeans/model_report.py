import streamlit as st
import pandas as pd
import numpy as np
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any

def get_silhouette_interpretation(score):
    """Provide interpretation of silhouette score"""
    if score >= 0.7:
        return "Strong structure"
    elif score >= 0.5:
        return "Reasonable structure"
    elif score >= 0.25:
        return "Weak structure"
    else:
        return "No substantial structure"

def _get_common_metrics(metrics_snapshot: Dict[str, Any]):
    """Extracts common metrics from the snapshot with safe defaults."""
    
    # Safe access with defaults
    optimal_k = metrics_snapshot.get('Optimal K', 0)
    inertia = metrics_snapshot.get('Inertia', 0.0)
    silhouette_score = metrics_snapshot.get('Silhouette Score', 0.0)
    calinski_harabasz = metrics_snapshot.get('Calinski-Harabasz Index', 0.0)
    davies_bouldin = metrics_snapshot.get('Davies-Bouldin Index', 0.0)
    cluster_distribution = metrics_snapshot.get('Cluster Distribution', {})
    k_method = metrics_snapshot.get('k_method', 'unknown')
    features = metrics_snapshot.get('features', [])
    
    return (optimal_k, inertia, silhouette_score, calinski_harabasz, 
            davies_bouldin, cluster_distribution, k_method, features)

def create_clustering_summary_text(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Generates the main summary text block."""
    (optimal_k, inertia, silhouette_score, calinski_harabasz, 
     davies_bouldin, _, k_method, features) = _get_common_metrics(metrics_snapshot)

    # Check if we have valid data
    if optimal_k == 0:
        return {"type": "text", "content": "❌ No K-Means model data available. Please train a model first."}

    text = """
    ### K-Means Clustering Performance Summary
    - **Features:** {features}
    - **K Selection Method:** {k_method}
    - **Optimal Clusters (K):** **{optimal_k}**
    - **Silhouette Score:** {silhouette:.4f} ({silhouette_text})
    - **Inertia:** {inertia:.4f}
    - **Calinski-Harabasz Index:** {calinski:.4f}
    - **Davies-Bouldin Index:** {davies:.4f}
    """.format(
        features=", ".join(features) if features else "No features selected",
        k_method=k_method,
        optimal_k=optimal_k,
        silhouette=silhouette_score,
        silhouette_text=get_silhouette_interpretation(silhouette_score),
        inertia=inertia,
        calinski=calinski_harabasz,
        davies=davies_bouldin
    )
    return {"type": "text", "content": text}

def create_cluster_distribution_plot(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Generates the cluster distribution bar chart."""
    _, _, _, _, _, cluster_distribution, _, _ = _get_common_metrics(metrics_snapshot)

    # Check if we have valid data
    if not cluster_distribution:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No cluster data available', 
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title('Cluster Distribution - No Data')
        plt.close(fig)
        return {"type": "plot", "content": fig}

    fig, ax = plt.subplots(figsize=(10, 6))
    clusters = list(cluster_distribution.keys())
    sizes = [int(info.split(' ')[0]) for info in cluster_distribution.values()]
    
    bars = ax.bar(clusters, sizes, color=plt.cm.Set3(np.linspace(0, 1, len(clusters))))
    ax.set_xlabel('Clusters')
    ax.set_ylabel('Number of Samples')
    ax.set_title('Cluster Distribution')
    
    # Add value labels on bars
    for bar, size in zip(bars, sizes):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{size}', ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.close(fig)
    
    return {"type": "plot", "content": fig}

# ... (similar safe implementations for the other functions) ...

def create_metrics_radar_chart(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Generates the metrics radar chart (Report Asset Type: plot)."""
    (_, _, silhouette_score, calinski_harabasz, 
     davies_bouldin, _, _, _) = _get_common_metrics(metrics_snapshot)

    metrics_names = ['Silhouette', 'Calinski-Harabasz', 'Davies-Bouldin']
    
    # Normalize metrics for radar chart
    silhouette_norm = silhouette_score
    calinski_norm = min(calinski_harabasz / 1000, 1)
    davies_norm = 1 - min(davies_bouldin / 10, 1)
    
    metrics_values = [silhouette_norm, calinski_norm, davies_norm]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    angles = np.linspace(0, 2 * np.pi, len(metrics_names), endpoint=False).tolist()
    metrics_values += metrics_values[:1]
    angles += angles[:1]
    
    ax.plot(angles, metrics_values, 'o-', linewidth=2, label='Metrics')
    ax.fill(angles, metrics_values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_names)
    ax.set_ylim(0, 1)
    ax.set_title('Clustering Metrics Radar Chart\n(Higher is Better)')
    ax.grid(True)
    plt.close(fig)
    
    return {"type": "plot", "content": fig}

def create_performance_metrics_table(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Generates the performance metrics table (Report Asset Type: dataframe)."""
    (optimal_k, inertia, silhouette_score, calinski_harabasz, 
     davies_bouldin, _, _, _) = _get_common_metrics(metrics_snapshot)

    metrics_data = {
        'Metric': [
            'Optimal K',
            'Silhouette Score', 
            'Inertia',
            'Calinski-Harabasz Index',
            'Davies-Bouldin Index'
        ],
        'Value': [
            f"{optimal_k}",
            f"{silhouette_score:.4f}",
            f"{inertia:.4f}",
            f"{calinski_harabasz:.4f}",
            f"{davies_bouldin:.4f}"
        ],
        'Interpretation': [
            'Number of clusters',
            get_silhouette_interpretation(silhouette_score),
            'Lower is better',
            'Higher is better', 
            'Lower is better'
        ]
    }
    
    df = pd.DataFrame(metrics_data)
    
    return {
        "type": "dataframe",
        "content": df
    }

def create_cluster_analysis_text(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Generates cluster performance interpretation text (Report Asset Type: text)."""
    (_, _, silhouette_score, _, _, _, _, _) = _get_common_metrics(metrics_snapshot)

    if silhouette_score >= 0.7:
        interpretation = "**Excellent Clustering** - Strong cluster structure detected!"
        color_style = "color: green;"
    elif silhouette_score >= 0.5:
        interpretation = "**Good Clustering** - Reasonable cluster structure."
        color_style = "color: blue;"
    elif silhouette_score >= 0.25:
        interpretation = "**Fair Clustering** - Weak cluster structure. Consider feature engineering."
        color_style = "color: orange;"
    else:
        interpretation = "**Poor Clustering** - No substantial cluster structure. Try different features or preprocessing."
        color_style = "color: red;"

    text = f"""
    ### Performance Interpretation
    <div style="{color_style}">
    {interpretation}
    </div>
    
    **Silhouette Score Analysis:**
    - **{silhouette_score:.4f}** indicates {get_silhouette_interpretation(silhouette_score).lower()}
    - Scores closer to 1.0 indicate better defined clusters
    - Scores near 0 indicate overlapping clusters
    - Negative scores indicate possible incorrect clustering
    """
    
    return {"type": "text", "content": text}

def create_configuration_details_text(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Generates model configuration details (Report Asset Type: text)."""
    (optimal_k, _, _, _, _, _, k_method, features) = _get_common_metrics(metrics_snapshot)

    text = f"""
    ### Model Configuration Details
    
    **Clustering Parameters:**
    - **K Selection:** {k_method}
    - **Optimal K:** {optimal_k}
    - **Features Used:** {len(features)}
    
    **Dataset Information:**
    - **Features:** {', '.join(features)}
    - **Total Features:** {len(features)}
    """
    
    return {"type": "text", "content": text}

# --- ML Page Display Function (Original model_report logic, now decoupled) ---

def display_kmeans_report(metrics_snapshot: Dict[str, Any]):
    """
    Function to be called on the main ML page. It uses the granular functions
    to generate and display all components using Streamlit commands.
    """
    if not metrics_snapshot:
        st.error("No model data provided for display.")
        return

    st.markdown("---")
    st.markdown("## K-Means Clustering Analysis (Live View)")
    
    # Get all the report assets
    summary_asset = create_clustering_summary_text(metrics_snapshot)
    distribution_asset = create_cluster_distribution_plot(metrics_snapshot)
    radar_asset = create_metrics_radar_chart(metrics_snapshot)
    table_asset = create_performance_metrics_table(metrics_snapshot)
    analysis_asset = create_cluster_analysis_text(metrics_snapshot)
    config_asset = create_configuration_details_text(metrics_snapshot)
    
    # Display Summary
    st.markdown(summary_asset['content'], unsafe_allow_html=True)
    
    # Display Cluster Distribution
    st.subheader("📊 Cluster Distribution")
    col1, col2 = st.columns(2)
    
    with col1:
        cluster_dist = metrics_snapshot['Cluster Distribution']
        st.write("**Cluster Sizes:**")
        for cluster, info in cluster_dist.items():
            st.write(f"- **{cluster}:** {info}")
    
    with col2:
        st.pyplot(distribution_asset['content'])
    
    # Display Metrics Visualization
    st.subheader("📈 Clustering Metrics Visualization")
    st.pyplot(radar_asset['content'])
    
    # Display Performance Metrics Table
    st.subheader("🔑 Key Metrics Summary")
    st.dataframe(table_asset['content'], use_container_width=True)
    
    # Display Performance Interpretation
    st.markdown(analysis_asset['content'], unsafe_allow_html=True)
    
    # Display Configuration Details
    st.markdown(config_asset['content'], unsafe_allow_html=True)

# For compatibility with your old system, we keep the original function name, 
# but it now just calls the decoupled display function.
def model_report():
    found = False 
    model_results = None
    for model_info in st.session_state.pipeline.get("ML", []):
        if model_info.get("model type") == 'KMeans Clustering':
            if 'metrics_snapshot' in model_info:
                found = True
                model_results = model_info.get("metrics_snapshot", {})
                break
    
    if not found:
        st.error("No K-Means model results found. Please create a model first.")
        return
    
    # Pass the JSON-safe metrics snapshot
    display_kmeans_report(model_results)

def display_compact_metrics():
    """Alternative compact metrics display"""
    found = False 
    model_results = None
    for model_info in st.session_state.pipeline.get("ML", []):
        if model_info.get("model type") == 'KMeans Clustering':
            if 'metrics_snapshot' in model_info:
                found = True
                model_results = model_info.get("metrics_snapshot", {})
                break
    
    if not found:
        return
    
    metrics = model_results
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Optimal K", f"{metrics['Optimal K']}")
    with col2:
        st.metric("Silhouette", f"{metrics['Silhouette Score']:.4f}")
    with col3:
        st.metric("Inertia", f"{metrics['Inertia']:.0f}")
    with col4:
        st.metric("Calinski", f"{metrics['Calinski-Harabasz Index']:.0f}")
    with col5:
        st.metric("Davies", f"{metrics['Davies-Bouldin Index']:.4f}")