import itertools

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from constants import DataManager

data_manager = DataManager()


def model_script(df, features, edit, use_grid_search, param_grid, manual_params, cv_folds):
    try:
        # Prepare data
        X = df[features].copy()

        # Handle potential infinite values
        X.replace([np.inf, -np.inf], np.nan, inplace=True)

        numeric_features = X.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
        categorical_features = X.select_dtypes(exclude=['int64', 'float64', 'int32', 'float32']).columns.tolist()

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        X_scaled = preprocessor.fit_transform(X)

        # Train model with or without grid search
        if use_grid_search:
            st.info("Performing Grid Search for optimal parameters...")

            best_score = -1
            best_params = None
            best_labels = None
            best_model = None

            # Manual grid search for DBSCAN (GridSearchCV doesn't work well with clustering)
            param_combinations = list(itertools.product(
                param_grid.get('eps', [0.5]),
                param_grid.get('min_samples', [5]),
                param_grid.get('metric', ['euclidean'])
            ))

            progress_bar = st.progress(0)
            total_combinations = len(param_combinations)

            for idx, (eps, min_samples, metric) in enumerate(param_combinations):
                metric_params = {'p': 2} if metric == 'minkowski' else None
                model = DBSCAN(
                    eps=eps, min_samples=min_samples,
                    metric=metric, metric_params=metric_params, n_jobs=-1
                )
                labels = model.fit_predict(X_scaled)

                # Calculate number of clusters
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = list(labels).count(-1)

                # Only evaluate if we have at least 2 clusters and not all noise
                if n_clusters >= 2 and n_noise < len(labels):
                    mask = labels != -1
                    if np.sum(mask) > 1:
                        try:
                            sample_size = 5000 if np.sum(mask) > 5000 else None
                            kwargs = {} if sample_size is None else {'sample_size': sample_size}
                            score = silhouette_score(X_scaled[mask], labels[mask], **kwargs)
                            if score > best_score:
                                best_score = score
                                best_params = {'eps': eps, 'min_samples': min_samples, 'metric': metric}
                                best_labels = labels
                                best_model = model
                        except Exception as score_err:
                            st.warning(
                                f"Silhouette score failed for eps={eps}, "
                                f"min_samples={min_samples}: {score_err}"
                            )
                            continue

                progress_bar.progress((idx + 1) / total_combinations)

            progress_bar.empty()

            if best_model is None:
                st.warning("Grid search couldn't find optimal parameters. Using default parameters.")
                best_params = {'eps': 0.5, 'min_samples': 5, 'metric': 'euclidean'}
                best_model = DBSCAN(**best_params, n_jobs=-1)
                best_labels = best_model.fit_predict(X_scaled)

            st.success(f"Grid Search completed! Best parameters: {best_params}")

        else:
            st.info("Training with manual parameters...")

            # Extract manual parameters
            eps = manual_params.get('eps', 0.5)
            min_samples = manual_params.get('min_samples', 5)
            metric = manual_params.get('metric', 'euclidean')
            algorithm = manual_params.get('algorithm', 'auto')
            leaf_size = manual_params.get('leaf_size', 30)

            metric_params = {'p': 2} if metric == 'minkowski' else None

            best_model = DBSCAN(
                eps=eps,
                min_samples=min_samples,
                metric=metric,
                metric_params=metric_params,
                algorithm=algorithm,
                leaf_size=leaf_size,
                n_jobs=-1
            )

            best_labels = best_model.fit_predict(X_scaled)
            best_params = manual_params

        # Calculate metrics
        n_clusters = len(set(best_labels)) - (1 if -1 in best_labels else 0)
        n_noise = list(best_labels).count(-1)

        # --- CRITICAL: Create JSON-SAFE metrics snapshot for the pipeline ---
        metrics_snapshot = {
            'N Clusters': n_clusters,
            'N Noise Points': n_noise,
            'N Samples': len(best_labels),
            'Best Parameters': best_params,
            'features': features,
            'use_grid_search': use_grid_search,
            'cv_folds': cv_folds,
            # Pre-set to None; filled below if valid clusters exist
            'Silhouette Score': None,
            'Davies-Bouldin Index': None,
            'Calinski-Harabasz Score': None,
        }

        # Guard: warn explicitly if all points are noise
        if n_clusters == 0:
            st.warning(
                "DBSCAN found 0 clusters — all points are noise. "
                "Try increasing eps or reducing min_samples."
            )
        elif n_clusters >= 2 and n_noise < len(best_labels):
            # Calculate clustering metrics only if we have valid clusters
            mask = best_labels != -1
            if np.sum(mask) > 1:
                try:
                    sample_size = 5000 if np.sum(mask) > 5000 else None
                    kwargs = {} if sample_size is None else {'sample_size': sample_size}
                    silhouette = silhouette_score(X_scaled[mask], best_labels[mask], **kwargs)
                    davies_bouldin = davies_bouldin_score(X_scaled[mask], best_labels[mask])
                    calinski_harabasz = calinski_harabasz_score(X_scaled[mask], best_labels[mask])

                    metrics_snapshot['Silhouette Score'] = float(silhouette)
                    metrics_snapshot['Davies-Bouldin Index'] = float(davies_bouldin)
                    metrics_snapshot['Calinski-Harabasz Score'] = float(calinski_harabasz)
                except Exception as e:
                    st.warning(f"Could not calculate some metrics: {str(e)}")

        # Update st.session_state.model_results for the live-view ML page
        model_results = {
            'model': best_model,
            'scaler': preprocessor,
            'labels': best_labels,
            'metrics': metrics_snapshot,
            'features': features,
            'use_grid_search': use_grid_search,
            'cv_folds': cv_folds,
            'X_scaled': X_scaled,
            'df': df
        }
        st.session_state.model_results = model_results

        # Create parameter list for pipeline
        param_list = [
            {"name": "features", "value": features},
            {"name": "use_grid_search", "value": use_grid_search},
            {"name": "cv_folds", "value": cv_folds}
        ]

        # Add DBSCAN-specific parameters
        if use_grid_search:
            param_list.extend([
                {"name": "eps_range", "value": str(param_grid.get('eps', []))},
                {"name": "min_samples_range", "value": str(param_grid.get('min_samples', []))},
                {"name": "metric_options", "value": str(param_grid.get('metric', []))}
            ])
        else:
            param_list.extend([
                {"name": "eps", "value": manual_params['eps']},
                {"name": "min_samples", "value": manual_params['min_samples']},
                {"name": "metric", "value": manual_params['metric']},
                {"name": "algorithm", "value": manual_params['algorithm']},
                {"name": "leaf_size", "value": manual_params['leaf_size']}
            ])

        # Ensure manual_params values are proper types
        if not use_grid_search:
            manual_params = {
                'eps': float(manual_params['eps']),
                'min_samples': int(manual_params['min_samples']),
                'metric': manual_params['metric'],
                'algorithm': manual_params['algorithm'],
                'leaf_size': int(manual_params['leaf_size'])
            }

        # Create model entry for pipeline, including the JSON-safe snapshot
        DBSCAN_model_pipeline_entry = DataManager.create_DBSCAN_Model(
            "DBSCAN_Cluster",
            param_list,
            st.session_state.selected_trans,
            metrics_snapshot
        )

        # Robust pipeline/session key guard
        if 'pipeline' not in st.session_state or not isinstance(st.session_state.pipeline, dict):
            st.session_state.pipeline = {}
        if 'ML' not in st.session_state.pipeline or not isinstance(st.session_state.pipeline.get('ML'), list):
            st.session_state.pipeline['ML'] = []

        model_name_to_check = "DBSCAN_Cluster"

        if edit:
            st.session_state.pipeline['ML'] = [
                item if item.get('model name') != model_name_to_check else DBSCAN_model_pipeline_entry
                for item in st.session_state.pipeline['ML']
            ]
        else:
            st.session_state.pipeline['ML'].append(DBSCAN_model_pipeline_entry)

        st.success("DBSCAN Model created successfully and results saved to pipeline!")
        return model_results

    except Exception as e:
        st.error(f"Error training DBSCAN model: {str(e)}")
        return None


def validate_model(params):
    if len(params['features']) == 0:
        st.error("Please select at least one feature column")
        return False

    # Check if we have enough data points
    if len(params['df']) < 10:
        st.error("Insufficient data for clustering (minimum 10 samples required)")
        return False

    return True
