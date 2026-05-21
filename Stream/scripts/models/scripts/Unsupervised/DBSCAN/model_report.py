import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any
from sklearn.decomposition import PCA

# --- Helper to safely retrieve metrics ---

def _get_common_metrics(metrics_snapshot: Dict[str, Any]):
    """Extracts common clustering metrics."""
    # --- Clustering Metrics ---
    n_clusters = metrics_snapshot.get('N Clusters', 0)
    n_noise = metrics_snapshot.get('N Noise Points', 0)
    n_samples = metrics_snapshot.get('N Samples', 0)
    silhouette = metrics_snapshot.get('Silhouette Score', None)
    davies_bouldin = metrics_snapshot.get('Davies-Bouldin Index', None)
    calinski_harabasz = metrics_snapshot.get('Calinski-Harabasz Score', None)
    
    # Placeholder/Dummy for compatibility
    class_df = pd.DataFrame() 
    conf_matrix_array = np.array([]) 
    
    return n_clusters, n_noise, n_samples, silhouette, davies_bouldin, calinski_harabasz, class_df, conf_matrix_array

# --- Granular Report Asset Generation Functions (Input: metrics_snapshot) ---

def create_ml_summary_text(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Generates the main summary text block (Report Asset Type: text)."""
    n_clusters, n_noise, n_samples, silhouette, davies_bouldin, calinski_harabasz, _, _ = _get_common_metrics(metrics_snapshot)

    text = """
    ### DBSCAN Performance Summary
    - **Features:** {features}
    - **N Clusters Found:** **{n_clusters}**
    - **N Noise Points:** {n_noise}
    - **Total Samples:** {n_samples}
    - **Noise Percentage:** {noise_pct:.2f}%
    - **Best Parameters:** {best_params}
    - **Training Method:** {grid_search}
    """.format(
        features=", ".join(metrics_snapshot['features']),
        n_clusters=n_clusters,
        n_noise=n_noise,
        n_samples=n_samples,
        noise_pct=(n_noise / n_samples * 100) if n_samples > 0 else 0,
        best_params=metrics_snapshot['Best Parameters'],
        grid_search="Grid Search" if metrics_snapshot['use_grid_search'] else "Manual"
    )
    return {"type": "text", "content": text}

def create_classification_report_table(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Creates a summary table of clustering metrics."""
    
    n_clusters, n_noise, n_samples, silhouette, davies_bouldin, calinski_harabasz, _, _ = _get_common_metrics(metrics_snapshot)
    
    clustering_summary = {
        'Metric': ['N Clusters', 'N Noise Points', 'Total Samples', 'Noise Percentage'],
        'Value': [
            f"{n_clusters}", 
            f"{n_noise}", 
            f"{n_samples}",
            f"{(n_noise / n_samples * 100) if n_samples > 0 else 0:.2f}%"
        ]
    }
    
    # Add advanced metrics if available
    if silhouette is not None:
        clustering_summary['Metric'].extend(['Silhouette Score', 'Davies-Bouldin Index'])
        clustering_summary['Value'].extend([f"{silhouette:.4f}", f"{davies_bouldin:.4f}"])
    
    if calinski_harabasz is not None:
        clustering_summary['Metric'].append('Calinski-Harabasz Score')
        clustering_summary['Value'].append(f"{calinski_harabasz:.4f}")
    
    df = pd.DataFrame(clustering_summary)
    return {"type": "dataframe", "content": df}

def create_cluster_distribution_plot(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Generates cluster distribution plot using metrics snapshot data."""
    n_clusters, n_noise, n_samples, _, _, _, _, _ = _get_common_metrics(metrics_snapshot)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create cluster distribution data
    clusters = []
    counts = []
    colors = []
    
    # Add clusters
    for i in range(n_clusters):
        clusters.append(f'Cluster {i}')
        counts.append(max(1, (n_samples - n_noise) // max(1, n_clusters)))  # Estimate distribution
        colors.append(f'C{i}')
    
    # Add noise
    if n_noise > 0:
        clusters.append('Noise')
        counts.append(n_noise)
        colors.append('red')
    
    if not clusters:  # If no clusters found
        clusters = ['No Clusters']
        counts = [n_samples]
        colors = ['gray']
    
    bars = ax.bar(clusters, counts, color=colors)
    ax.set_ylabel('Number of Samples')
    ax.set_xlabel('Cluster')
    ax.set_title('Estimated Cluster Distribution')
    ax.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(count)}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.close(fig)
    return {"type": "plot", "content": fig}

def create_quality_metrics_plot(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Generates quality metrics plot (Report Asset Type: plot)."""
    n_clusters, n_noise, n_samples, silhouette, davies_bouldin, calinski_harabasz, _, _ = _get_common_metrics(metrics_snapshot)
    
    if silhouette is None:
        # Create a basic metrics plot even without advanced metrics
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metrics_names = ['N Clusters', 'N Noise Points']
        metrics_values = [n_clusters, n_noise]
        
        colors = ['#4CAF50', '#FF9800']
        bars = ax.bar(metrics_names, metrics_values, color=colors)
        ax.set_ylabel('Count')
        ax.set_title('Basic Clustering Metrics')
        
        # Add value labels
        for bar, value in zip(bars, metrics_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{int(value)}', ha='center', va='bottom', fontsize=10)
        
        plt.close(fig)
        return {"type": "plot", "content": fig}
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics_names = ['Silhouette\nScore', 'Davies-Bouldin\nIndex']
    metrics_values = [silhouette, davies_bouldin]
    
    if calinski_harabasz is not None:
        metrics_names.append('Calinski-Harabasz\nScore')
        metrics_values.append(calinski_harabasz)
    
    colors = ['#4CAF50', '#FF9800', '#2196F3']
    bars = ax.bar(metrics_names[:len(metrics_values)], metrics_values, color=colors[:len(metrics_values)])
    ax.set_ylabel('Score')
    ax.set_title('Clustering Quality Metrics')
    
    # Add value labels
    for bar, value in zip(bars, metrics_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.close(fig)
    return {"type": "plot", "content": fig}

def create_parameter_importance_plot(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Creates a visualization of parameter importance based on best parameters."""
    best_params = metrics_snapshot.get('Best Parameters', {})
    
    if not best_params:
        return {"type": "text", "content": "Parameter information not available"}
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    param_names = list(best_params.keys())
    param_values = []
    
    # Convert parameter values to numeric for visualization
    for param_name in param_names:
        value = best_params[param_name]
        if isinstance(value, (int, float)):
            param_values.append(value)
        elif isinstance(value, str) and value.replace('.', '').isdigit():
            param_values.append(float(value))
        else:
            # For categorical parameters, assign a numeric value
            param_values.append(1.0)
    
    # Create a bar chart showing relative parameter values
    bars = ax.bar(param_names, param_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    ax.set_ylabel('Parameter Value')
    ax.set_title('Best Parameter Configuration')
    ax.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, value, name in zip(bars, param_values, param_names):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.close(fig)
    return {"type": "plot", "content": fig}

# --- ML Page Display Function ---

def display_ml_report(metrics_snapshot: Dict[str, Any]):
    """
    Function to be called on the main ML page. It uses the granular functions
    to generate and display all components using Streamlit commands.
    """
    if not metrics_snapshot:
        st.error("No model data provided for display.")
        return

    st.markdown("---")
    st.markdown("## DBSCAN Analysis (Live View)")
    
    # Get all the report assets
    summary_asset = create_ml_summary_text(metrics_snapshot)
    table_asset = create_classification_report_table(metrics_snapshot)
    cluster_dist_asset = create_cluster_distribution_plot(metrics_snapshot)
    quality_plot_asset = create_quality_metrics_plot(metrics_snapshot)
    param_plot_asset = create_parameter_importance_plot(metrics_snapshot)
    
    # Display Summary
    st.markdown(summary_asset['content'], unsafe_allow_html=True)
    
    # Display Visualizations in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Cluster Distribution")
        if cluster_dist_asset['type'] == 'plot':
            st.pyplot(cluster_dist_asset['content'])
        else:
            st.info(cluster_dist_asset['content'])
    
    with col2:
        st.subheader("📈 Quality Metrics")
        if quality_plot_asset['type'] == 'plot':
            st.pyplot(quality_plot_asset['content'])
        else:
            st.info(quality_plot_asset['content'])
    
    # Display Parameter Configuration
    st.subheader("⚙️ Parameter Configuration")
    if param_plot_asset['type'] == 'plot':
        st.pyplot(param_plot_asset['content'])
    else:
        st.info(param_plot_asset['content'])
    
    # Display Clustering Metrics Table
    st.subheader("📋 Key Clustering Metrics")
    st.dataframe(table_asset['content'], use_container_width=True)
    
    # Performance Interpretation
    n_clusters, n_noise, n_samples, silhouette, davies_bouldin, calinski_harabasz, _, _ = _get_common_metrics(metrics_snapshot)
    
    st.subheader("📊 Performance Interpretation")
    
    if n_clusters == 0:
        st.error("**No Clusters Found** - All points classified as noise. Try adjusting eps and min_samples parameters.")
    elif n_clusters == 1:
        st.warning("**Single Cluster** - Only one cluster found. Consider adjusting parameters for better separation.")
    elif silhouette is not None:
        if silhouette >= 0.7:
            st.success("**Excellent Clustering** - Strong cluster structure with good separation!")
        elif silhouette >= 0.5:
            st.info("**Good Clustering** - Reasonable cluster structure detected.")
        elif silhouette >= 0.25:
            st.warning("**Fair Clustering** - Weak cluster structure. Consider parameter tuning.")
        else:
            st.error("**Poor Clustering** - Very weak cluster structure. Try different parameters or features.")
    else:
        st.info(f"**{n_clusters} Clusters Found** - {n_noise} points classified as noise ({(n_noise/n_samples*100) if n_samples > 0 else 0:.1f}%)")
        
    # Model Configuration Details
    st.subheader("🔧 Model Configuration")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.write("**Hyperparameters:**")
        best_params = metrics_snapshot['Best Parameters']
        if isinstance(best_params, dict):
            for param, value in best_params.items():
                st.write(f"- **{param}:** {value}")
        else:
            st.write(f"- {best_params}")
        
        st.write("**Training Method:**")
        st.write(f"- Grid Search: {'✅ Yes' if metrics_snapshot['use_grid_search'] else '❌ No'}")
    
    with col4:
        st.write("**Dataset Information:**")
        st.write(f"- Number of features: **{len(metrics_snapshot['features'])}**")
        st.write(f"- Total samples: **{n_samples}**")
        st.write(f"- Noise percentage: **{(n_noise/n_samples*100) if n_samples > 0 else 0:.2f}%**")
        
        if metrics_snapshot['use_grid_search']:
            cv_folds = metrics_snapshot.get('cv_folds', 'N/A')
            st.write(f"- CV Folds: **{cv_folds}**")
    
    # Enhanced DBSCAN-specific sections
    st.subheader("🎯 DBSCAN Insights")
    
    col5, col6 = st.columns(2)
    
    with col5:
        st.info("""
        **DBSCAN Advantages:**
        • Automatically determines cluster count
        • Identifies outliers as noise
        • Handles arbitrary cluster shapes
        • Robust to noise and outliers
        """)
    
    with col6:
        if n_noise > 0:
            noise_percentage = (n_noise / n_samples * 100) if n_samples > 0 else 0
            if noise_percentage > 20:
                st.warning(f"**High Noise Level:** {noise_percentage:.1f}% of data is noise")
            elif noise_percentage > 5:
                st.info(f"**Moderate Noise:** {noise_percentage:.1f}% of data is noise")
            else:
                st.success(f"**Low Noise:** {noise_percentage:.1f}% of data is noise")
        
        if n_clusters > 0:
            avg_cluster_size = (n_samples - n_noise) / n_clusters if n_clusters > 0 else 0
            st.write(f"**Average Cluster Size:** {avg_cluster_size:.1f} samples")

# For compatibility with your old system, we keep the original function name
def model_report():
    found = False 
    # Check for the updated model name
    for model_info in st.session_state.pipeline.get("ML", []):
        if model_info.get("model name") == 'DBSCAN_Cluster':
            if 'metrics_snapshot' in model_info:
                found = True
                model_results = model_info.get("metrics_snapshot", {})
                break
    
    # If the cluster wasn't found, try the old name for robustness
    if not found:
        for model_info in st.session_state.pipeline.get("ML", []):
            if model_info.get("model name") == 'DBSCAN': # Fallback check
                if 'metrics_snapshot' in model_info:
                    found = True
                    model_results = model_info.get("metrics_snapshot", {})
                    break

    if not found:
        st.error("No DBSCAN model results found. Please create a model first.")
        return
    
    # Pass the JSON-safe metrics snapshot 
    display_ml_report(model_results)

# Optional: Add a function to display metrics in a more compact way
def display_compact_metrics():
    """Alternative compact metrics display"""
    found = False
    model_results = {}
    
    # Check for the updated model name
    for model_info in st.session_state.pipeline.get("ML", []):
        if model_info.get("model name") == 'DBSCAN_Cluster':
            if 'metrics_snapshot' in model_info:
                found = True
                model_results = model_info.get("metrics_snapshot", {})
                break
    
    # If the cluster wasn't found, try the old name for robustness
    if not found:
        for model_info in st.session_state.pipeline.get("ML", []):
            if model_info.get("model name") == 'DBSCAN': # Fallback check
                if 'metrics_snapshot' in model_info:
                    found = True
                    model_results = model_info.get("metrics_snapshot", {})
                    break

    if not found:
        return
    
    n_clusters = model_results.get('N Clusters', 0)
    n_noise = model_results.get('N Noise Points', 0)
    silhouette = model_results.get('Silhouette Score')
    
    # Create a compact metrics card
    if silhouette is not None:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("N Clusters", n_clusters)
        with col2:
            st.metric("Noise Points", n_noise)
        with col3:
            st.metric("Silhouette", f"{silhouette:.4f}")
        with col4:
            davies_bouldin = model_results.get('Davies-Bouldin Index', 0)
            st.metric("Davies-Bouldin", f"{davies_bouldin:.4f}")
    else:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("N Clusters", n_clusters)
        with col2:
            st.metric("Noise Points", n_noise)
        with col3:
            n_samples = model_results.get('N Samples', 0)
            noise_pct = (n_noise / n_samples * 100) if n_samples > 0 else 0
            st.metric("Noise %", f"{noise_pct:.1f}%")