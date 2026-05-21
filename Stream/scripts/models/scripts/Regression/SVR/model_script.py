import streamlit as st
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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

        # --- Build X ---
        X_df = df[features].copy()
        _cat_cols = X_df.select_dtypes(include=['object', 'category']).columns.tolist()
        if _cat_cols:
            X_df = pd.get_dummies(X_df, columns=_cat_cols, drop_first=True)
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

        # --- Train model ---
        if use_grid_search:
            st.info("Performing Grid Search for optimal SVR parameters...")
            svr = SVR()
            try:
                grid_search = GridSearchCV(
                    svr,
                    param_grid or {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear'], 'epsilon': [0.1, 0.2]},
                    cv=int(cv_folds or 5),
                    scoring='neg_mean_squared_error',
                    n_jobs=-1
                )
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                st.success(f"Grid Search completed! Best parameters: {best_params}")
            except Exception as gs_err:
                st.warning(f"Grid search failed ({gs_err}). Falling back to manual parameters.")
                best_model = SVR(
                    C=float(manual_params.get('C', 1.0)),
                    kernel=manual_params.get('kernel', 'rbf'),
                    gamma=manual_params.get('gamma', 'scale'),
                    epsilon=float(manual_params.get('epsilon', 0.1))
                )
                best_model.fit(X_train, y_train)
                best_params = manual_params
        else:
            st.info("Training with manual parameters...")
            try:
                best_model = SVR(
                    C=float(manual_params.get('C', 1.0)),
                    kernel=manual_params.get('kernel', 'rbf'),
                    gamma=manual_params.get('gamma', 'scale'),
                    epsilon=float(manual_params.get('epsilon', 0.1))
                )
                best_model.fit(X_train, y_train)
                best_params = manual_params
            except Exception as fit_err:
                st.error(f"Model fit failed: {fit_err}")
                return None

        # --- Metrics ---
        y_pred = best_model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        metrics_snapshot = {
            'R2 Score': float(r2),
            'MAE': float(mae),
            'MSE': float(mse),
            'RMSE': float(rmse),
            'Best Parameters': best_params,
            'features': list(X_df.columns),
            'target': target,
            'use_grid_search': use_grid_search,
            'cv_folds': cv_folds,
            'y_test': y_test.tolist(),
            'y_pred': y_pred.tolist(),
        }

        model_results = {
            'model': best_model,
            'metrics': metrics_snapshot,
            'features': list(X_df.columns),
            'target': target,
        }
        st.session_state.model_results = model_results

        param_list = [
            {"name": "features", "value": features},
            {"name": "target", "value": target},
            {"name": "best_params", "value": best_params}
        ]

        try:
            SVR_model = DataManager.create_SVM_Model(
                "Support Vector Regression",
                param_list,
                st.session_state.get('selected_trans', []),
                metrics_snapshot
            )
        except Exception:
            SVR_model = data_manager.create_SVM_Model(
                "Support Vector Regression",
                param_list,
                st.session_state.get('selected_trans', []),
                metrics_snapshot
            )

        if 'pipeline' not in st.session_state: st.session_state.pipeline = {}
        if 'ML' not in st.session_state.pipeline: st.session_state.pipeline['ML'] = []

        model_key = "Support Vector Regression"
        if edit:
            st.session_state.pipeline['ML'] = [
                item if item.get('model type') != model_key and item.get('model name') != model_key
                else SVR_model
                for item in st.session_state.pipeline['ML']
            ]
        else:
            st.session_state.pipeline['ML'].append(SVR_model)

        st.success("Support Vector Regression model created successfully!")
        return model_results

    except Exception as e:
        st.error(f"Error training Support Vector Regression model: {str(e)}")
        return None

def validate_model(params):
    features = params.get('features', [])
    df = params.get('df')
    if df is None or len(df) == 0:
        st.error("No data available.")
        return False
    if not features:
        st.error("Please select at least one feature column")
        return False
    target = params.get('target')
    if not target or target not in df.columns:
        st.error("Please select a valid target column")
        return False
    return True
