import streamlit as st
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN, SpectralClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from constants import DataManager

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from transformations.utils.robust_ml import clean_numeric_frame, make_columns_unique, scale_array

data_manager = DataManager()


def apply_knn_clustering(X, n_clusters, n_neighbors, metric, algorithm):
    """Apply KNN-based clustering method"""

    if n_clusters is not None:
        # Use spectral clustering with KNN graph
        try:
            nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, algorithm=algorithm)
            nn.fit(X)
            adjacency_matrix = nn.kneighbors_graph(X, mode='connectivity')
            spectral = SpectralClustering(
                n_clusters=n_clusters,
                affinity='precomputed',
                random_state=42
            )
            clusters = spectral.fit_predict(adjacency_matrix)
            method_used = "Spectral Clustering with KNN graph"
        except Exception as spec_err:
            # Fallback to DBSCAN density estimation
            nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, algorithm=algorithm)
            nn.fit(X)
            distances, _ = nn.kneighbors(X)
            avg_distances = distances.mean(axis=1)
            eps = np.percentile(avg_distances, 50)
            dbscan = DBSCAN(eps=eps, min_samples=n_neighbors)
            clusters = dbscan.fit_predict(X)
            method_used = f"DBSCAN fallback (Spectral failed: {spec_err})"
    else:
        # Use density-based clustering
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, algorithm=algorithm)
        nn.fit(X)
        distances, _ = nn.kneighbors(X)
        avg_distances = distances.mean(axis=1)
        eps = np.percentile(avg_distances, 50)
        dbscan = DBSCAN(eps=eps, min_samples=n_neighbors)
        clusters = dbscan.fit_predict(X)
        method_used = "DBSCAN with KNN density estimation"

    return clusters, method_used


def model_script(df, features, edit, use_grid_search, param_grid, manual_params, n_clusters):
    try:
        # --- Basic validation ---
        if df is None or not isinstance(df, pd.DataFrame) or len(df) == 0:
            st.error("Invalid or empty dataframe provided.")
            return None

        features = [f for f in features if f in df.columns]
        features = make_columns_unique(features)

        if len(features) == 0:
            st.error("Please select at least one valid feature column.")
            return None

        if len(df) < 4:
            st.error("Not enough rows for clustering (need at least 4).")
            return None

        # Guard n_clusters
        if n_clusters is not None:
            n_clusters = max(2, int(n_clusters))
            if n_clusters >= len(df):
                st.error("Number of clusters cannot be greater than or equal to number of samples.")
                return None

        # --- Build X ---
        X_df = df[features].copy()
        X_df = X_df.select_dtypes(include=[np.number])
        if X_df.empty:
            st.error("No numeric feature columns found.")
            return None

        X_df = clean_numeric_frame(X_df)
        valid_features = make_columns_unique(X_df.columns.tolist())
        X_df.columns = valid_features
        X_scaled = scale_array(X_df.values, "standard")

        if manual_params is None:
            manual_params = {}
        if param_grid is None:
            param_grid = {}

        # Clamp n_neighbors to be valid
        n_samples = len(X_scaled)
        max_neighbors = max(1, n_samples - 1)

        # --- Grid search or manual ---
        if use_grid_search:
            st.info("Finding optimal KNN parameters...")
            best_score = -1
            best_params = {'n_neighbors': 5, 'algorithm': 'auto', 'metric': 'euclidean'}
            best_clusters = None
            best_method = "Manual"

            for n_neighbors in param_grid.get('n_neighbors', [5]):
                n_neighbors = max(1, min(int(n_neighbors), max_neighbors))
                for algorithm in param_grid.get('algorithm', ['auto']):
                    for metric in param_grid.get('metric', ['euclidean']):
                        try:
                            clusters, method_used = apply_knn_clustering(
                                X_scaled, n_clusters, n_neighbors, metric, algorithm
                            )
                            if len(np.unique(clusters)) > 1:
                                score = float(silhouette_score(X_scaled, clusters))
                                if score > best_score:
                                    best_score = score
                                    best_params = {
                                        'n_neighbors': n_neighbors,
                                        'algorithm': algorithm,
                                        'metric': metric
                                    }
                                    best_clusters = clusters
                                    best_method = method_used
                        except Exception:
                            continue

            if best_clusters is None:
                st.warning("Grid search could not find valid clusters. Using default parameters.")
                n_neighbors_default = max(1, min(5, max_neighbors))
                best_clusters, best_method = apply_knn_clustering(
                    X_scaled, n_clusters, n_neighbors_default, 'euclidean', 'auto'
                )
                best_params = {'n_neighbors': n_neighbors_default, 'algorithm': 'auto', 'metric': 'euclidean'}

            st.success(f"Optimal parameters found: {best_params}")

        else:
            st.info("Applying KNN clustering with manual parameters...")
            n_neighbors = max(1, min(int(manual_params.get('n_neighbors', 5)), max_neighbors))
            try:
                best_clusters, best_method = apply_knn_clustering(
                    X_scaled, n_clusters,
                    n_neighbors,
                    manual_params.get('metric', 'euclidean'),
                    manual_params.get('algorithm', 'auto')
                )
                best_params = manual_params
            except Exception as e:
                st.error(f"Clustering failed: {e}")
                return None

        # --- Metrics ---
        silhouette = calinski = davies = None
        if len(np.unique(best_clusters)) > 1:
            try:
                silhouette = float(silhouette_score(X_scaled, best_clusters))
            except Exception:
                pass
            try:
                calinski = float(calinski_harabasz_score(X_scaled, best_clusters))
            except Exception:
                pass
            try:
                davies = float(davies_bouldin_score(X_scaled, best_clusters))
            except Exception:
                pass
        else:
            st.warning("Only one cluster found. Metrics may not be meaningful.")

        unique_clusters, counts = np.unique(best_clusters, return_counts=True)
        cluster_sizes = {str(int(k)): int(v) for k, v in zip(unique_clusters, counts)}

        model_results = {
            'clusters': best_clusters,
            'features': valid_features,
            'n_clusters': len(np.unique(best_clusters)),
            'metrics': {
                'silhouette_score': silhouette,
                'calinski_harabasz_score': calinski,
                'davies_bouldin_score': davies
            },
            'cluster_sizes': cluster_sizes,
            'method_used': best_method,
            'parameters': best_params
        }

        param_list = [
            {"name": "features", "value": [str(f) for f in valid_features]},
            {"name": "n_clusters", "value": n_clusters},
            {"name": "use_grid_search", "value": use_grid_search},
            {"name": "n_neighbors", "value": str(param_grid.get('n_neighbors', [])) if use_grid_search else int(manual_params.get('n_neighbors', 5))},
            {"name": "algorithm", "value": str(param_grid.get('algorithm', [])) if use_grid_search else manual_params.get('algorithm', 'auto')},
            {"name": "metric", "value": str(param_grid.get('metric', [])) if use_grid_search else manual_params.get('metric', 'euclidean')}
        ]

        try:
            UnsupKNN_model = DataManager.create_UnsupKNN_Model(
                "Unsupervised KNN",
                param_list,
                st.session_state.selected_trans
            )
        except Exception:
            UnsupKNN_model = data_manager.create_UnsupKNN_Model(
                "Unsupervised KNN",
                param_list,
                st.session_state.selected_trans
            )

        if 'pipeline' not in st.session_state or not isinstance(st.session_state.pipeline, dict):
            st.session_state.pipeline = {}
        if 'ML' not in st.session_state.pipeline or not isinstance(st.session_state.pipeline.get('ML'), list):
            st.session_state.pipeline['ML'] = []

        if edit:
            st.session_state.pipeline['ML'] = [
                item if item.get('name') != 'Unsupervised KNN' and item.get('model name') != 'Unsupervised KNN'
                else UnsupKNN_model
                for item in st.session_state.pipeline['ML']
            ]
        else:
            st.session_state.pipeline['ML'].append(UnsupKNN_model)

        st.success("Unsupervised KNN Clustering completed successfully!")
        return model_results

    except Exception as e:
        st.error(f"Error performing unsupervised KNN clustering: {str(e)}")
        return None


def validate_model(params):
    features = params.get('features', [])
    df = params.get('df')

    if df is None or (hasattr(df, '__len__') and len(df) == 0):
        st.error("No data available.")
        return False

    if len(features) == 0:
        st.error("Please select at least one feature column")
        return False

    if len(features) < 2:
        st.warning("Clustering with only one feature may not yield meaningful results")

    n_clusters = params.get('n_clusters')
    if n_clusters is None:
        st.error("Please specify the number of clusters")
        return False

    if df is not None:
        n_samples = len(df)
        if int(n_clusters) >= n_samples:
            st.error("Number of clusters cannot be greater than or equal to number of samples")
            return False

    return True