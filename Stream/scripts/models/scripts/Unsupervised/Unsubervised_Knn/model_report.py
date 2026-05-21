import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def model_report():
    if 'model_results' not in st.session_state:
        st.error("No model results found. Please create a model first.")
        return
    
    results = st.session_state.model_results
    
    # Check if results contain clustering metrics
    if 'clusters' not in results:
        st.error("Invalid model results format for unsupervised KNN clustering")
        return
    
    # Extract metrics
    metrics = results['metrics']
    n_clusters = results['n_clusters']
    cluster_sizes = results['cluster_sizes']
    
    st.markdown("""
    <div class="report-container">
        <h3>Unsupervised KNN Clustering Report</h3>
        <div class="metric-card">
            <strong>Features:</strong> {features}<br>
            <strong>Clustering Method:</strong> {method}<br>
            <strong>Number of Clusters Found:</strong> {n_clusters}
        </div>
        <div class="metric-card">
            <strong>Silhouette Score:</strong> {silhouette:.4f}<br>
            <strong>Calinski-Harabasz Index:</strong> {calinski:.4f}<br>
            <strong>Davies-Bouldin Index:</strong> {davies:.4f}
        </div>
        <div class="metric-card">
            <strong>Parameters:</strong> {params}
        </div>
    </div>
    """.format(
        features=", ".join(results['features']),
        method=results['method_used'],
        n_clusters=n_clusters,
        silhouette=metrics.get('silhouette_score', 'N/A'),
        calinski=metrics.get('calinski_harabasz_score', 'N/A'),
        davies=metrics.get('davies_bouldin_score', 'N/A'),
        params=results['parameters']
    ), unsafe_allow_html=True)
    
    # Cluster Distribution
    st.subheader("📊 Cluster Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Cluster sizes bar chart
        fig, ax = plt.subplots(figsize=(8, 6))
        clusters = list(cluster_sizes.keys())
        sizes = list(cluster_sizes.values())
        
        bars = ax.bar([f'Cluster {c}' for c in clusters], sizes, 
                     color=plt.cm.Set3(np.linspace(0, 1, len(clusters))))
        ax.set_ylabel('Number of Points')
        ax.set_title('Cluster Sizes Distribution')
        
        # Add value labels on bars
        for bar, size in zip(bars, sizes):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{size}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    with col2:
        # Cluster sizes pie chart
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        labels = [f'Cluster {c}' for c in clusters]
        ax2.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90,
               colors=plt.cm.Set3(np.linspace(0, 1, len(clusters))))
        ax2.set_title('Cluster Proportion')
        st.pyplot(fig2)
    
    # Dimensionality Reduction for Visualization
    st.subheader("🌐 Cluster Visualization")
    
    # Prepare data for visualization
    X = results['df'][results['features']].values
    X_scaled = results['scaler'].transform(X) if 'scaler' in results else X
    clusters = results['clusters']
    
    # Use PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Create scatter plot
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    scatter = ax3.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='tab10', alpha=0.7)
    ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    ax3.set_title('Cluster Visualization (PCA)')
    plt.colorbar(scatter, ax=ax3, label='Cluster')
    st.pyplot(fig3)
    
    # t-SNE visualization (if dataset isn't too large)
    if len(X_scaled) <= 1000:
        st.subheader("🔍 t-SNE Visualization")
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_scaled)-1))
        X_tsne = tsne.fit_transform(X_scaled)
        
        fig4, ax4 = plt.subplots(figsize=(10, 8))
        scatter_tsne = ax4.scatter(X_tsne[:, 0], X_tsne[:, 1], c=clusters, cmap='tab10', alpha=0.7)
        ax4.set_xlabel('t-SNE Component 1')
        ax4.set_ylabel('t-SNE Component 2')
        ax4.set_title('Cluster Visualization (t-SNE)')
        plt.colorbar(scatter_tsne, ax=ax4, label='Cluster')
        st.pyplot(fig4)
    
    # Cluster Characteristics Analysis
    st.subheader("📈 Cluster Characteristics")
    
    # Add cluster labels to dataframe for analysis
    df_with_clusters = results['df'].copy()
    df_with_clusters['Cluster'] = clusters
    
    # Show mean values for each cluster
    numeric_features = [col for col in results['features'] 
                       if df_with_clusters[col].dtype in ['int64', 'float64']]
    
    if numeric_features:
        cluster_means = df_with_clusters.groupby('Cluster')[numeric_features].mean()
        
        st.write("**Average Feature Values by Cluster:**")
        st.dataframe(cluster_means.style.background_gradient(cmap='Blues'), use_container_width=True)
        
        # Feature distributions by cluster
        st.write("**Feature Distributions by Cluster:**")
        
        selected_feature = st.selectbox(
            "Select feature to visualize distribution:",
            options=numeric_features
        )
        
        fig5, ax5 = plt.subplots(figsize=(10, 6))
        for cluster in np.unique(clusters):
            cluster_data = df_with_clusters[df_with_clusters['Cluster'] == cluster][selected_feature]
            ax5.hist(cluster_data, alpha=0.7, label=f'Cluster {cluster}', bins=20)
        
        ax5.set_xlabel(selected_feature)
        ax5.set_ylabel('Frequency')
        ax5.set_title(f'Distribution of {selected_feature} by Cluster')
        ax5.legend()
        st.pyplot(fig5)
    
    # Metrics Interpretation
    st.subheader("📊 Clustering Quality Assessment")
    
    silhouette = metrics.get('silhouette_score')
    calinski = metrics.get('calinski_harabasz_score')
    davies = metrics.get('davies_bouldin_score')
    
    col3, col4, col5 = st.columns(3)
    
    with col3:
        if silhouette is not None:
            st.metric("Silhouette Score", f"{silhouette:.4f}")
            if silhouette > 0.7:
                st.success("Strong clustering structure")
            elif silhouette > 0.5:
                st.info("Reasonable clustering structure")
            elif silhouette > 0.25:
                st.warning("Weak clustering structure")
            else:
                st.error("No substantial clustering structure")
    
    with col4:
        if calinski is not None:
            st.metric("Calinski-Harabasz", f"{calinski:.4f}")
            # Higher is better for Calinski-Harabasz
    
    with col5:
        if davies is not None:
            st.metric("Davies-Bouldin", f"{davies:.4f}")
            # Lower is better for Davies-Bouldin
    
    # Cluster Insights
    st.subheader("💡 Cluster Insights")
    
    if n_clusters == 1:
        st.warning("Only one cluster found. The data may not have clear natural groupings.")
    elif n_clusters > 10:
        st.info(f"Found {n_clusters} clusters. Consider whether this level of granularity is meaningful.")
    else:
        st.success(f"Found {n_clusters} distinct clusters in the data.")
    
    # Outlier detection (if using density-based method)
    if -1 in cluster_sizes:  # -1 typically indicates outliers in DBSCAN
        outlier_count = cluster_sizes[-1]
        st.warning(f"Detected {outlier_count} potential outliers (noise points)")
    
    # Recommendations
    st.subheader("🎯 Recommendations")
    
    recommendations = []
    
    if silhouette is not None and silhouette < 0.3:
        recommendations.append("Consider trying different numbers of clusters")
        recommendations.append("Try different distance metrics or preprocessing")
    
    if n_clusters < 2:
        recommendations.append("The data may not have clear cluster structure")
        recommendations.append("Consider using outlier detection instead")
    
    if recommendations:
        st.info("**Suggestions for improvement:**")
        for rec in recommendations:
            st.write(f"• {rec}")
    else:
        st.success("Clustering results appear satisfactory!")
    
    # Export clustered data
    st.subheader("💾 Export Results")
    
    if st.button("Download Clustered Data as CSV"):
        csv = df_with_clusters.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="clustered_data.csv",
            mime="text/csv"
        )

def display_compact_metrics():
    """Alternative compact metrics display"""
    if 'model_results' not in st.session_state:
        return
    
    results = st.session_state.model_results
    metrics = results['metrics']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        silhouette = metrics.get('silhouette_score')
        if silhouette is not None:
            st.metric("Silhouette Score", f"{silhouette:.4f}")
    
    with col2:
        calinski = metrics.get('calinski_harabasz_score')
        if calinski is not None:
            st.metric("Calinski-Harabasz", f"{calinski:.4f}")
    
    with col3:
        davies = metrics.get('davies_bouldin_score')
        if davies is not None:
            st.metric("Davies-Bouldin", f"{davies:.4f}")