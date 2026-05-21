import streamlit as st
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from constants import DataManager

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from transformations.utils.robust_ml import clean_numeric_frame, make_columns_unique, scale_array


def model_script(df, features, n_clusters, linkage, metric, compute_full_tree, distance_threshold, edit):
    try:
        # --- Basic validation ---
        if df is None or not isinstance(df, pd.DataFrame) or len(df) == 0:
            st.error("Invalid or empty dataframe provided.")
            st.session_state.model_results = None
            return None

        features = [f for f in features if f in df.columns]
        features = make_columns_unique(features)

        if len(features) < 2:
            st.error("Please select at least two valid feature columns for clustering.")
            st.session_state.model_results = None
            return None

        if len(df) < 4:
            st.error("Not enough rows for clustering (need at least 4).")
            st.session_state.model_results = None
            return None

        # Guard n_clusters
        n_clusters = max(2, int(n_clusters))
        if n_clusters > len(df):
            st.error("Number of clusters cannot exceed the number of data points")
            st.session_state.model_results = None
            return None

        # --- Build X ---
        X_df = df[features].copy()
        X_df = X_df.select_dtypes(include=[np.number])
        if X_df.empty:
            st.error("No numeric features found. Please select numeric feature columns.")
            st.session_state.model_results = None
            return None

        X_df = clean_numeric_frame(X_df)
        valid_features = make_columns_unique(X_df.columns.tolist())
        X_df.columns = valid_features

        if len(valid_features) < 2:
            st.error("Need at least 2 numeric features for hierarchical clustering.")
            st.session_state.model_results = None
            return None

        X_scaled = scale_array(X_df.values, "standard")

        # Ward linkage only works with euclidean metric
        if linkage == 'ward' and metric != 'euclidean':
            st.warning("Ward linkage requires Euclidean distance. Switching metric to 'euclidean'.")
            metric = 'euclidean'

        # --- Fit model ---
        try:
            model = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric=metric,
                linkage=linkage,
                compute_full_tree=compute_full_tree,
                distance_threshold=distance_threshold
            )
            cluster_labels = model.fit_predict(X_scaled)
        except Exception as fit_err:
            st.error(f"Hierarchical Clustering fitting failed: {fit_err}")
            st.session_state.model_results = None
            return None

        # --- Metrics (safe, optional) ---
        silhouette_avg = None
        calinski_score = None
        davies_score = None

        if len(np.unique(cluster_labels)) >= 2:
            try:
                silhouette_avg = float(silhouette_score(X_scaled, cluster_labels))
            except Exception:
                pass
            try:
                calinski_score = float(calinski_harabasz_score(X_scaled, cluster_labels))
            except Exception:
                pass
            try:
                davies_score = float(davies_bouldin_score(X_scaled, cluster_labels))
            except Exception:
                pass

        cluster_sizes = [int(np.sum(cluster_labels == i)) for i in range(n_clusters)]

        # JSON-safe metrics snapshot (no raw arrays or model objects)
        metrics_snapshot = {
            'Silhouette Score': silhouette_avg,
            'Calinski-Harabasz Score': calinski_score,
            'Davies-Bouldin Score': davies_score,
            'Number of Clusters': int(n_clusters),
            'Cluster Sizes': cluster_sizes,
            'features': [str(f) for f in valid_features],
            'linkage': str(linkage),
            'metric': str(metric),
            'compute_full_tree': compute_full_tree
        }

        model_results = {
            'model': model,
            'metrics': {
                'Silhouette Score': silhouette_avg,
                'Calinski-Harabasz Score': calinski_score,
                'Davies-Bouldin Score': davies_score,
                'Number of Clusters': int(n_clusters),
                'Cluster Sizes': cluster_sizes
            },
            'cluster_labels': cluster_labels,
            'features': valid_features,
            'X_scaled': X_scaled
        }

        param_list = [
            {"name": "features", "value": [str(f) for f in valid_features]},
            {"name": "n_clusters", "value": int(n_clusters)},
            {"name": "linkage", "value": str(linkage)},
            {"name": "metric", "value": str(metric)},
            {"name": "compute_full_tree", "value": compute_full_tree},
            {"name": "distance_threshold", "value": distance_threshold}
        ]

        model_name_to_check = "Hierarchical Clustering"
        try:
            Hierarchy_model_pipeline_entry = DataManager.create_Hierarchical_Clustering_Model(
                model_name_to_check,
                param_list,
                st.session_state.selected_trans,
                model,
                metrics_snapshot
            )
        except Exception:
            Hierarchy_model_pipeline_entry = {
                'model name': model_name_to_check,
                'params': param_list,
                'metrics_snapshot': metrics_snapshot
            }

        if 'pipeline' not in st.session_state or not isinstance(st.session_state.pipeline, dict):
            st.session_state.pipeline = {}
        if 'ML' not in st.session_state.pipeline or not isinstance(st.session_state.pipeline.get('ML'), list):
            st.session_state.pipeline['ML'] = []

        if edit:
            st.session_state.pipeline['ML'] = [
                item if item.get('model name') != model_name_to_check and item.get('name') != model_name_to_check
                else Hierarchy_model_pipeline_entry
                for item in st.session_state.pipeline['ML']
            ]
        else:
            st.session_state.pipeline['ML'].append(Hierarchy_model_pipeline_entry)

        st.success("Hierarchical Clustering model created successfully!")
        st.session_state.model_results = model_results
        return model_results

    except Exception as e:
        st.error(f"An error occurred while creating the model: {e}")
        st.session_state.model_results = None
        return None


def validate_model(params):
    features = params.get('features', [])
    df = params.get('df')

    if df is None or (hasattr(df, '__len__') and len(df) == 0):
        st.error("No data available.")
        return False

    if len(features) < 2:
        st.error("Please select at least two feature columns for clustering")
        return False

    if params.get('linkage') == 'ward' and params.get('metric') != 'euclidean':
        st.error("Ward linkage can only be used with Euclidean distance metric")
        return False

    n_clusters = params.get('n_clusters', 2)
    if n_clusters < 2:
        st.error("Number of clusters must be at least 2")
        return False

    if df is not None and n_clusters > len(df):
        st.error("Number of clusters cannot exceed the number of data points")
        return False

    return True