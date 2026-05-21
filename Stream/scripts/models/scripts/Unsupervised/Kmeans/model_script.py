import streamlit as st
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from constants import DataManager

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from transformations.utils.robust_ml import clean_numeric_frame, make_columns_unique, scale_array

import json

data_manager = DataManager()

def detect_elbow(inertia_values):
    """Automatically detect elbow point for optimal K"""
    if len(inertia_values) < 2:
        return 2
    x = np.arange(len(inertia_values))
    y = np.array(inertia_values)
    p1 = np.array([x[0], y[0]])
    p2 = np.array([x[-1], y[-1]])
    norm = np.linalg.norm(p2 - p1)
    if norm == 0:
        return 2
    distances = []
    for i in range(len(x)):
        p = np.array([x[i], y[i]])
        distance = np.abs(np.cross(p2 - p1, p1 - p)) / norm
        distances.append(distance)
    return x[np.argmax(distances)] + 2

def model_script(df, features, k_method, auto_k, manual_k, max_k, edit):
    try:
        # --- Basic validation ---
        if df is None or not isinstance(df, pd.DataFrame) or len(df) == 0:
            st.error("Invalid or empty dataframe provided.")
            return {"status": "error", "message": "Invalid dataframe"}

        features = [f for f in features if f in df.columns]
        features = make_columns_unique(features)

        if len(features) < 1:
            st.error("Please select at least one valid feature column for clustering.")
            return {"status": "error", "message": "No valid features"}

        if len(df) < 4:
            st.error("Not enough rows for clustering (need at least 4).")
            return {"status": "error", "message": "Too few rows"}

        # --- Build X ---
        X_df = df[features].copy()

        # Keep only numeric columns
        X_df = X_df.select_dtypes(include=[np.number])
        if X_df.empty:
            st.error("No numeric feature columns found. Please select numeric features.")
            return {"status": "error", "message": "No numeric features"}

        X_df = clean_numeric_frame(X_df)
        valid_features = make_columns_unique(X_df.columns.tolist())
        X_df.columns = valid_features

        # Scale features (robust — handles outliers)
        X_scaled = scale_array(X_df.values, "standard")

        # --- Determine optimal K ---
        max_k = max(3, int(max_k)) if max_k else 10
        manual_k = max(2, int(manual_k)) if manual_k else 2
        max_possible_k = max(2, min(max_k, len(df) // 2))

        if k_method == "auto":
            inertia_values = []
            k_range = range(2, max_possible_k + 1)
            for k in k_range:
                try:
                    km = KMeans(n_clusters=k, random_state=42, n_init=10)
                    km.fit(X_scaled)
                    inertia_values.append(km.inertia_)
                except Exception:
                    break
            if not inertia_values:
                optimal_k = 2
            else:
                optimal_k = int(detect_elbow(inertia_values))
                optimal_k = max(2, min(optimal_k, max_possible_k))
            st.info(f"Auto-detected optimal K: {optimal_k} using Elbow Method")
        else:
            optimal_k = max(2, min(int(manual_k), max_possible_k))
            st.info(f"Using manual K: {optimal_k}")

        # --- Fit final model ---
        try:
            kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X_scaled)
        except Exception as fit_err:
            st.error(f"KMeans fitting failed: {fit_err}")
            return {"status": "error", "message": str(fit_err)}

        # --- Metrics ---
        inertia_value = float(kmeans.inertia_)

        silhouette_avg = None
        calinski_harabasz = None
        davies_bouldin = None

        unique_labels = np.unique(clusters)
        if len(unique_labels) >= 2:
            try:
                silhouette_avg = float(silhouette_score(X_scaled, clusters))
            except Exception:
                pass
            try:
                calinski_harabasz = float(calinski_harabasz_score(X_scaled, clusters))
            except Exception:
                pass
            try:
                davies_bouldin = float(davies_bouldin_score(X_scaled, clusters))
            except Exception:
                pass

        cluster_counts = pd.Series(clusters).value_counts().sort_index()
        cluster_distribution = {
            f"Cluster {int(cluster)}": f"{int(count)} samples ({float(count/len(df)*100):.1f}%)"
            for cluster, count in cluster_counts.items()
        }

        metrics_snapshot = {
            'Optimal K': int(optimal_k),
            'Inertia': inertia_value,
            'Silhouette Score': silhouette_avg,
            'Calinski-Harabasz Index': calinski_harabasz,
            'Davies-Bouldin Index': davies_bouldin,
            'Cluster Distribution': cluster_distribution,
            'k_method': str(k_method),
            'features': [str(f) for f in valid_features]
        }

        param_list = [
            {"name": "features", "value": [str(f) for f in valid_features]},
            {"name": "k_method", "value": str(k_method)},
            {"name": "auto_k", "value": bool(auto_k)},
            {"name": "manual_k", "value": int(manual_k)},
            {"name": "max_k", "value": int(max_k)}
        ]

        try:
            kmeans_model = DataManager.create_KMeans_Model(
                "K-Means",
                param_list,
                st.session_state.get("selected_trans", [])
            )
        except Exception:
            kmeans_model = {
                "model name": "K-Means",
                "model type": "KMeans Clustering",
                "model param": param_list,
                "transformations": st.session_state.get("selected_trans", []),
                "metrics_snapshot": metrics_snapshot
            }

        if 'pipeline' not in st.session_state or not isinstance(st.session_state.pipeline, dict):
            st.session_state.pipeline = {}
        if 'ML' not in st.session_state.pipeline or not isinstance(st.session_state.pipeline.get('ML'), list):
            st.session_state.pipeline['ML'] = []

        if edit:
            st.session_state.pipeline['ML'] = [
                item if item.get('name') != 'K-Means' and item.get('model name') != 'K-Means'
                else kmeans_model
                for item in st.session_state.pipeline['ML']
            ]
        else:
            st.session_state.pipeline['ML'].append(kmeans_model)

        st.success("K-Means Model created successfully!")
        return {"status": "success", "metrics_snapshot": metrics_snapshot}

    except Exception as e:
        st.error(f"Error training K-Means model: {str(e)}")
        return {"status": "error", "message": str(e)}


def validate_model(params):
    features = params.get('features', [])
    df = params.get('df')

    if df is None or (hasattr(df, '__len__') and len(df) == 0):
        st.error("No data available.")
        return False

    if len(features) < 1:
        st.error("Please select at least one feature column for clustering")
        return False

    if df is not None:
        numeric_features = [f for f in features if f in df.columns and pd.api.types.is_numeric_dtype(df[f])]
        if len(numeric_features) == 0:
            st.error("All selected features must be numeric for K-Means clustering")
            return False

    if params.get('k_method') == "manual" and int(params.get('manual_k', 2)) < 2:
        st.error("Number of clusters (K) must be at least 2")
        return False

    if int(params.get('max_k', 10)) < 3:
        st.error("Maximum K for auto-detection must be at least 3")
        return False

    return True