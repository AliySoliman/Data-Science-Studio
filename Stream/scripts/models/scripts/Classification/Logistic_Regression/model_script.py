import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from constants import DataManager

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from transformations.utils.robust_ml import clean_numeric_frame, make_columns_unique, scale_array

data_manager = DataManager()

def model_script(df, features, target, edit, use_grid_search, param_grid, manual_params, cv_folds, solver, max_iter):
    try:
        # --- Basic validation ---
        if df is None or not isinstance(df, pd.DataFrame) or len(df) == 0:
            st.error("Invalid or empty dataframe provided.")
            return None

        # Filter to features that actually exist in df
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
        # Drop any remaining non-numeric columns; fallback to get_dummies if needed
        _cat_cols = X_df.select_dtypes(include=['object', 'category']).columns.tolist()
        if _cat_cols:
            X_df = pd.get_dummies(X_df, columns=_cat_cols, drop_first=True)
        X_df = X_df.select_dtypes(include=[np.number])
        X_df = clean_numeric_frame(X_df)
        X = scale_array(X_df.values, "robust")

        # --- Build y ---
        y_raw = df[target].replace([np.inf, -np.inf], np.nan)
        valid_mask = y_raw.notna()
        X = X[valid_mask.values]
        y_raw = y_raw[valid_mask]

        if len(X) < 5:
            st.error("Not enough valid rows after cleaning target column.")
            return None

        # Encode target
        le = LabelEncoder()
        y = le.fit_transform(y_raw.astype(str))
        target_encoder = le

        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            st.error("Target column has only 1 unique class after encoding. Classification requires at least 2 classes.")
            return None

        # --- Stratified split with fallback ---
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        except ValueError:
            st.warning(
                "Stratified split failed (likely a class with too few samples). "
                "Falling back to random split."
            )
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

        if len(X_test) == 0:
            st.error("Test set is empty after splitting. Provide more data.")
            return None

        # Ensure safe params
        if manual_params is None:
            manual_params = {}
        if param_grid is None:
            param_grid = {}
        try:
            cv_folds = max(2, int(cv_folds))
        except Exception:
            cv_folds = 5

        # --- Train model ---
        if use_grid_search:
            st.info("Performing Grid Search for optimal parameters...")
            lr = LogisticRegression(random_state=42, max_iter=1000)
            try:
                grid_search = GridSearchCV(
                    lr,
                    param_grid,
                    cv=cv_folds,
                    scoring='accuracy',
                    n_jobs=-1
                )
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                st.success(f"Grid Search completed! Best parameters: {best_params}")
            except Exception as gs_err:
                st.warning(f"Grid search failed ({gs_err}). Falling back to manual parameters.")
                best_model = LogisticRegression(
                    C=float(manual_params.get('C', 1.0)),
                    solver=manual_params.get('solver', 'lbfgs'),
                    max_iter=int(manual_params.get('max_iter', 1000)),
                    random_state=42
                )
                best_model.fit(X_train, y_train)
                best_params = manual_params
        else:
            st.info("Training with manual parameters...")
            try:
                best_model = LogisticRegression(
                    C=float(manual_params.get('C', 1.0)),
                    solver=manual_params.get('solver', 'lbfgs'),
                    max_iter=int(manual_params.get('max_iter', 1000)),
                    random_state=42
                )
                best_model.fit(X_train, y_train)
                best_params = manual_params
            except Exception as fit_err:
                st.error(f"Model fit failed: {fit_err}")
                return None

        # --- Predictions & metrics ---
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)

        # AUC-ROC for binary
        auc_score = None
        fpr_list = []
        tpr_list = []
        try:
            y_pred_proba = best_model.predict_proba(X_test)
            if len(best_model.classes_) == 2:
                auc_score = float(roc_auc_score(y_test, y_pred_proba[:, 1]))
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
                fpr_list = fpr.tolist()
                tpr_list = tpr.tolist()
        except Exception:
            pass

        metrics_snapshot = {
            'Accuracy': float(accuracy),
            'Classification Report': class_report,
            'Confusion Matrix': conf_matrix.tolist(),
            'Best Parameters': best_params,
            'AUC-ROC': auc_score,
            'fpr': fpr_list,
            'tpr': tpr_list,
            'features': features,
            'target': target,
            'use_grid_search': use_grid_search,
            'cv_folds': cv_folds,
        }

        model_results = {
            'model': best_model,
            'scaler': None,
            'target_encoder': target_encoder,
            'X_test': X_test,
            'y_test': y_test,
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
            {"name": "c_values", "value": str(param_grid.get('C', [])) if use_grid_search else manual_params.get('C', 1.0)},
            {"name": "solver", "value": str(param_grid.get('solver', [])) if use_grid_search else manual_params.get('solver', 'lbfgs')},
            {"name": "max_iter", "value": manual_params.get('max_iter', 1000) if not use_grid_search else str(param_grid.get('max_iter', [100, 200]))},
            {"name": "cv_folds", "value": cv_folds}
        ]

        try:
            LR_model_pipeline_entry = DataManager.create_Logistic_Regression_Model(
                "LR",
                param_list,
                st.session_state.selected_trans
            )
        except Exception:
            LR_model_pipeline_entry = data_manager.create_Logistic_Regression_Model(
                "LR",
                param_list,
                st.session_state.selected_trans
            )

        LR_model_pipeline_entry["metrics_snapshot"] = metrics_snapshot

        # Robust pipeline/session key guard
        if 'pipeline' not in st.session_state or not isinstance(st.session_state.pipeline, dict):
            st.session_state.pipeline = {}
        if 'ML' not in st.session_state.pipeline or not isinstance(st.session_state.pipeline.get('ML'), list):
            st.session_state.pipeline['ML'] = []

        model_name_to_check = "LR"
        if edit:
            st.session_state.pipeline['ML'] = [
                item if item.get('model name') != model_name_to_check and item.get('name') != model_name_to_check
                else LR_model_pipeline_entry
                for item in st.session_state.pipeline['ML']
            ]
        else:
            st.session_state.pipeline['ML'].append(LR_model_pipeline_entry)

        st.success("Logistic Regression Model created successfully and results saved to pipeline!")
        return model_results

    except Exception as e:
        st.error(f"Error training Logistic Regression model: {str(e)}")
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

    # Check that features exist
    missing = [f for f in features if f not in df.columns]
    if missing:
        st.error(f"Features not found in dataframe: {missing}")
        return False

    target_values = df[target].nunique()
    if target_values < 2:
        st.error("Target column must have at least 2 unique classes for classification")
        return False

    if not params.get('use_grid_search', False):
        manual_params = params.get('manual_params', {})
        if 'solver' in manual_params and manual_params['solver'] in ['newton-cg', 'lbfgs', 'sag']:
            if df[target].nunique() > 2:
                st.warning(f"Solver '{manual_params['solver']}' may not support multiclass. Consider 'saga'.")

    return True