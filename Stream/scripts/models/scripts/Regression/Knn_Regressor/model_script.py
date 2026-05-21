import streamlit as st
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from constants import DataManager

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from transformations.utils.robust_ml import clean_numeric_frame, make_columns_unique, scale_array

data_manager = DataManager()

def model_script(df, features, target, edit, use_grid_search, param_grid, manual_params, cv_folds):
    try:
        # --- Basic validation ---
        if df is None or not isinstance(df, pd.DataFrame) or len(df) == 0:
            st.error("Invalid or empty dataframe provided.")
            return None

        features = [f for f in features if f in df.columns]
        features = make_columns_unique(features)

        if not features:
            st.error("None of the selected feature columns exist in the dataframe.")
            return None

        if target not in df.columns:
            st.error(f"Target column '{target}' not found in dataframe.")
            return None

        # --- Build X (KNN is distance-based, use robust scale) ---
        X_df = df[features].copy()
        X_df = clean_numeric_frame(X_df)
        X = scale_array(X_df.values, "robust")

        # --- Build y ---
        y_series = pd.to_numeric(df[target], errors='coerce')
        y_series = y_series.replace([np.inf, -np.inf], np.nan)
        valid_mask = y_series.notna()
        X = X[valid_mask.values]
        y = y_series[valid_mask].values.astype(float)

        if len(X) < 5:
            st.error("Not enough valid rows after cleaning target column.")
            return None

        # --- Split ---
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        if len(X_test) == 0:
            st.error("Test set is empty after splitting.")
            return None

        if manual_params is None:
            manual_params = {}
        if param_grid is None:
            param_grid = {}
        try:
            cv_folds = max(2, int(cv_folds))
        except Exception:
            cv_folds = 5

        # n_neighbors must not exceed training samples
        n_neighbors_default = max(1, min(5, len(X_train) - 1))

        # --- Train model ---
        if use_grid_search:
            st.info("Performing Grid Search for optimal parameters...")
            knn = KNeighborsRegressor()
            try:
                grid_search = GridSearchCV(
                    knn,
                    param_grid,
                    cv=cv_folds,
                    scoring='neg_mean_squared_error',
                    n_jobs=-1
                )
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                st.success(f"Grid Search completed! Best parameters: {best_params}")
            except Exception as gs_err:
                st.warning(f"Grid search failed ({gs_err}). Falling back to manual parameters.")
                best_model = KNeighborsRegressor(
                    n_neighbors=int(manual_params.get('n_neighbors', n_neighbors_default)),
                    weights=manual_params.get('weights', 'uniform'),
                    algorithm=manual_params.get('algorithm', 'auto')
                )
                best_model.fit(X_train, y_train)
                best_params = manual_params
        else:
            st.info("Training with manual parameters...")
            try:
                best_model = KNeighborsRegressor(
                    n_neighbors=int(manual_params.get('n_neighbors', n_neighbors_default)),
                    weights=manual_params.get('weights', 'uniform'),
                    algorithm=manual_params.get('algorithm', 'auto')
                )
                best_model.fit(X_train, y_train)
                best_params = manual_params
            except Exception as fit_err:
                st.error(f"Model fit failed: {fit_err}")
                return None

        # --- Metrics ---
        y_pred = best_model.predict(X_test)
        r2 = float(r2_score(y_test, y_pred))
        mae = float(mean_absolute_error(y_test, y_pred))
        mse = float(mean_squared_error(y_test, y_pred))

        if not use_grid_search:
            best_params = {
                'n_neighbors': int(manual_params.get('n_neighbors', n_neighbors_default)),
                'weights': manual_params.get('weights', 'uniform'),
                'algorithm': manual_params.get('algorithm', 'auto')
            }

        metrics_snapshot = {
            'R2 Score': r2,
            'MAE': mae,
            'MSE': mse,
            'Best Parameters': best_params,
            'features': features,
            'target': target,
            'use_grid_search': use_grid_search,
            'cv_folds': cv_folds,
            'y_test': y_test.tolist(),
            'y_pred': y_pred.tolist()
        }

        target_encoder = None
        model_results = {
            'model': best_model,
            'scaler': None,
            'target_encoder': target_encoder,
            'metrics': metrics_snapshot,
            'features': features,
            'target': target,
            'use_grid_search': use_grid_search,
            'cv_folds': cv_folds,
        }
        st.session_state.model_results = model_results

        param_list = [
            {"name": "features", "value": features},
            {"name": "target", "value": target},
            {"name": "use_grid_search", "value": use_grid_search},
            {"name": "n_neighbors_range", "value": str(param_grid.get('n_neighbors', [])) if use_grid_search else int(manual_params.get('n_neighbors', n_neighbors_default))},
            {"name": "weights", "value": str(param_grid.get('weights', [])) if use_grid_search else manual_params.get('weights', 'uniform')},
            {"name": "algorithm", "value": str(param_grid.get('algorithm', [])) if use_grid_search else manual_params.get('algorithm', 'auto')},
            {"name": "cv_folds", "value": cv_folds}
        ]

        if not use_grid_search:
            manual_params = {
                'n_neighbors': int(manual_params.get('n_neighbors', n_neighbors_default)),
                'weights': manual_params.get('weights', 'uniform'),
                'algorithm': manual_params.get('algorithm', 'auto')
            }

        try:
            KNN_model_pipeline_entry = DataManager.create_KNN_Regressor_Model(
                "KNN_Regressor",
                param_list,
                st.session_state.selected_trans,
                best_model,
                metrics_snapshot
            )
        except Exception:
            KNN_model_pipeline_entry = data_manager.create_KNN_Regressor_Model(
                "KNN_Regressor",
                param_list,
                st.session_state.selected_trans,
                best_model,
                metrics_snapshot
            )

        if 'pipeline' not in st.session_state or not isinstance(st.session_state.pipeline, dict):
            st.session_state.pipeline = {}
        if 'ML' not in st.session_state.pipeline or not isinstance(st.session_state.pipeline.get('ML'), list):
            st.session_state.pipeline['ML'] = []

        model_name_to_check = "KNN_Regressor"
        if edit:
            st.session_state.pipeline['ML'] = [
                item if item.get('model name') != model_name_to_check and item.get('name') != model_name_to_check
                else KNN_model_pipeline_entry
                for item in st.session_state.pipeline['ML']
            ]
        else:
            st.session_state.pipeline['ML'].append(KNN_model_pipeline_entry)

        st.success("KNN Regressor Model created successfully and results saved to pipeline!")
        return model_results

    except Exception as e:
        st.error(f"Error training KNN Regressor model: {str(e)}")
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

    if params.get('target') in features:
        st.error("Target column cannot be one of the features")
        return False

    target = params.get('target')
    if target not in df.columns:
        st.error(f"Target column '{target}' not found in dataframe.")
        return False

    missing = [f for f in features if f not in df.columns]
    if missing:
        st.error(f"Features not found in dataframe: {missing}")
        return False

    y_numeric = pd.to_numeric(df[target], errors='coerce')
    if y_numeric.isna().all():
        st.error("Target column must be numeric for regression")
        return False

    return True