import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
from typing import Dict, Any
import models.model as m

def display_ml_report(metrics_snapshot: Dict[str, Any]):
    if not metrics_snapshot:
        st.error("No model data provided for display.")
        return
    
    matrices_asset = get_common_metrics(metrics_snapshot)
    cluster_dist_asset = create_cluster_ditribution_plot(metrics_snapshot)
    dendrogram_asset = create_dendrogram_plot(metrics_snapshot)
    pca_projection_asset = create_PCA_Projection_plot(metrics_snapshot)
    silhouette_asset = create_silhouette_plot(metrics_snapshot)

    

    st.subheader("📊 Cluster Size Distribution")
    st.pyplot(cluster_dist_asset['content'])

    st.subheader("🌳 Dendrogram Visualization")
    st.pyplot(dendrogram_asset['content'])
    st.markdown("""
        **How to read the dendrogram:**
        - **Vertical lines**: Represent clusters being merged
        - **Height**: Distance between clusters when merged (higher = less similar)
        - **Horizontal lines**: Show which clusters/points are connected
        - **Cutoff line**: Red dashed line shows where clusters are cut to get your chosen number
        """)
    
    st.subheader("📐 2D PCA Projection of Clusters")
    st.pyplot(pca_projection_asset['content'])

    st.subheader("🎭 Silhouette Analysis")
    st.pyplot(silhouette_asset['content'])
    


# This is the main function.
def model_report():
    found = False 
    # Check for the updated model name
    for model_info in st.session_state.pipeline.get("ML", []):
        if model_info.get("model name") == 'Hierarchical Clustering':
            if 'metrics_snapshot' in model_info:
                found = True
                model_results = model_info.get("metrics_snapshot", {})
                break
    


    if not found:
        st.error("No Hierarchical Clistering model results found. Please create a model first.")
        return
    # --- If we are here, the results are valid ---
    
    st.markdown("---")
    st.markdown("## Hierarchical Clustering Model Analysis (Live View)")
    
    
    # Call all the report components.
    display_ml_report(model_results)

################################################################################
# All functions below now (correctly) read from 'st.session_state.model_results'
################################################################################

def get_common_metrics(metrics_snapshot: Dict[str, Any]): 
    results = metrics_snapshot
    
    # Main report section
    text = st.markdown("""
                <div class="report-container">
                    <h3>Clustering Performance Report</h3>
                    <div class="metric-card">
                        <strong>Features Used:</strong> {features}<br>
                        <strong>Number of Clusters:</strong> {n_clusters}
                    </div>
                    <div class="metric-card">
                        <strong>Silhouette Score:</strong> {silhouette:.4f}<br>
                        <strong>Calinski-Harabasz Score:</strong> {calinski:.4f}<br>
                        <strong>Davies-Bouldin Score:</strong> {davies:.4f}
                    </div>
                    <div class="metric-card">
                        <strong>Cluster Sizes:</strong><br>
                        {cluster_sizes}
                    </div>
                    <div class="metric-card">
                        <strong>Algorithm Parameters:</strong><br>
                        <strong>Linkage Method:</strong> {linkage}<br>
                        <strong>Distance Metric:</strong> {metric}<br>
                        <strong>Compute Full Tree:</strong> {full_tree}
                    </div>
                </div>
                """.format(
                    features=", ".join(results['features']),
                    n_clusters=results['Number of Clusters'],
                    silhouette=results['Silhouette Score'] if results['Silhouette Score'] is not None else 0,
                    calinski=results['Calinski-Harabasz Score'] if results['Calinski-Harabasz Score'] is not None else 0,
                    davies=results['Davies-Bouldin Score'] if results['Davies-Bouldin Score'] is not None else 0,
                    cluster_sizes="<br>".join([f"&nbsp;&nbsp;Cluster {i}: {size} points" 
                                               for i, size in enumerate(results['Cluster Sizes'])]),
                    linkage=results['linkage'],
                    metric=results['metric'],
                full_tree=results['compute_full_tree']
                ), unsafe_allow_html=True)
    
    return {"type": "text", "content": " "}



def create_cluster_ditribution_plot(metrics_snapshot: Dict[str, Any]): 
    results = metrics_snapshot
    n_clusters = results['Number of Clusters']
    # st.write("**Cluster Size Distribution**")
    cluster_sizes = results['Cluster Sizes']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Pie chart
    colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
    wedges, texts, autotexts = ax1.pie(cluster_sizes, labels=[f'Cluster {i}' for i in range(n_clusters)], 
                                     autopct='%1.1f%%', colors=colors, startangle=90)
    ax1.set_title('Cluster Distribution')
    
    # Bar chart
    bars = ax2.bar(range(n_clusters), cluster_sizes, color=plt.cm.viridis(np.linspace(0, 1, n_clusters)))
    ax2.set_xlabel('Cluster')
    ax2.set_ylabel('Number of Points')
    ax2.set_title('Cluster Sizes')
    ax2.set_xticks(range(n_clusters))
    ax2.set_xticklabels([f'Cluster {i}' for i in range(n_clusters)])
    
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                 f'{int(height)}', ha='center', va='bottom')
    
    # plt.tight_layout()
    # st.pyplot(fig)
    return {"type": "plot", "content": fig}


def create_dendrogram_plot(metrics_snapshot: Dict[str, Any]):
    results = metrics_snapshot
    n_clusters = results['Number of Clusters']
    X = results['X_scaled']
    # st.write("**Dendrogram - Cluster Merging Hierarchy**")
        
    try:
        Z = linkage(X, method=results['linkage'], metric=results['metric'])
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        dendrogram(
            Z,
            truncate_mode='lastp',
            p=n_clusters,
            show_leaf_counts=True,
            leaf_rotation=90.,
            leaf_font_size=8.,
            show_contracted=True,
            ax=ax
        )
        
        if n_clusters > 1:
            last_merge_height = Z[-(n_clusters-1), 2]
            ax.axhline(y=last_merge_height, color='r', linestyle='--', 
                       label=f'Cutoff for {n_clusters} clusters')
        
        ax.set_title(f"Dendrogram ({results['linkage']} linkage)")
        ax.set_xlabel('Data points or cluster size')
        ax.set_ylabel('Distance')
        ax.legend()
        # plt.tight_layout()
        # st.pyplot(fig)
        
        # st.markdown("""
        # **How to read the dendrogram:**
        # - **Vertical lines**: Represent clusters being merged
        # - **Height**: Distance between clusters when merged (higher = less similar)
        # - **Horizontal lines**: Show which clusters/points are connected
        # - **Cutoff line**: Red dashed line shows where clusters are cut to get your chosen number
        # """)
        
    except Exception as e:
        st.error(f"Could not create dendrogram: {str(e)}")
        st.info("Note: Dendrograms work best with Euclidean distance and 'ward', 'complete', 'average', or 'single' linkage.")

    return {"type": "plot", "content": fig}    


def create_PCA_Projection_plot(metrics_snapshot: Dict[str, Any]):
    results = metrics_snapshot
    cluster_labels = results['cluster_labels']
    X = results['X_scaled']
    # st.write("**2D PCA Projection of Clusters**")
    if X.shape[1] >= 2:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7, s=50)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        ax.set_title('Cluster Visualization (PCA Projection)')
        # plt.colorbar(scatter, ax=ax, label='Cluster')
        # st.pyplot(fig)
    else:
        st.info("Need at least 2 features for 2D visualization")

    return {"type": "plot", "content": fig}    


def create_silhouette_plot(metrics_snapshot: Dict[str, Any]): 
    results = metrics_snapshot
    n_clusters = results['Number of Clusters']
    cluster_labels = results['cluster_labels']
    X = results['X_scaled']
    # st.write("**Silhouette Analysis**")
    if results['Silhouette Score'] is not None:
        from sklearn.metrics import silhouette_samples
        silhouette_vals = silhouette_samples(X, cluster_labels)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        y_ticks = []
        y_lower = y_upper = 0
        
        for i in range(n_clusters):
            cluster_silhouette_vals = silhouette_vals[cluster_labels == i]
            cluster_silhouette_vals.sort()
            y_upper += len(cluster_silhouette_vals)
            color = plt.cm.viridis(i / n_clusters)
            ax.barh(range(y_lower, y_upper), cluster_silhouette_vals, height=1.0, 
                   edgecolor='none', color=color)
            y_ticks.append((y_lower + y_upper) / 2)
            y_lower += len(cluster_silhouette_vals)
        
        ax.axvline(results['Silhouette Score'], color="red", linestyle="--", 
                   label=f'Average: {results["Silhouette Score"]:.3f}')
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f'Cluster {i}' for i in range(n_clusters)])
        ax.set_xlabel('Silhouette Coefficient')
        ax.set_ylabel('Cluster')
        ax.set_title('Silhouette Plot for Clusters')
        ax.legend()
        # st.pyplot(fig)
    else:
        st.info("Silhouette score not available (requires 2 or more clusters).")

    return {"type": "plot", "content": fig}    