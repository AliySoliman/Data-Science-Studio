import streamlit as st
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from constants import DataManager

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from transformations.utils.robust_ml import clean_numeric_frame, make_columns_unique, scale_array

data_manager = DataManager()

def model_script(df, features, target, edit, model_type, use_grid_search, param_grid, manual_params, cv_folds):
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

        # Drop any remaining non-numeric columns (e.g. if encoding didn't fully cover all cols)
        X_df = X_df.select_dtypes(include=[np.number])
        if X_df.empty:
            # Fallback: try get_dummies on all features for any remaining object columns
            X_df = pd.get_dummies(df[features], drop_first=True)
            X_df = X_df.select_dtypes(include=[np.number])
        if X_df.empty:
            st.error("No numeric features available for Naive Bayes after encoding. Please encode your categorical features first.")
            return None

        X_df = clean_numeric_frame(X_df)
        numeric_features = X_df.columns.tolist()

        # Normalize model_type to lowercase handle 'GaussianNB', 'gaussian', 'Gaussian' etc.
        model_type_lower = str(model_type).lower().replace('nb', '').replace('naive_bayes', '').strip()
        if 'gauss' in model_type_lower or model_type_lower == 'gaussian':
            model_type_lower = 'gaussian'
        elif 'multi' in model_type_lower:
            model_type_lower = 'multinomial'
        else:
            model_type_lower = 'bernoulli'
        model_type = model_type_lower

        # GaussianNB: standard scale; MultinomialNB: no negatives (clip); BernoulliNB: none
        if model_type == 'gaussian':
            X = scale_array(X_df.values, "standard")
            scaler_used = "standard"
        elif model_type == 'multinomial':
            # MultinomialNB requires non-negative features
            X_np = X_df.values.astype(float)
            col_mins = X_np.min(axis=0)
            neg_mask = col_mins < 0
            if neg_mask.any():
                X_np[:, neg_mask] -= col_mins[neg_mask]
            X = X_np
            scaler_used = "none"
        else:
            X = X_df.values
            scaler_used = "none"

        # --- Build y ---
        y_raw = df[target].replace([np.inf, -np.inf], np.nan)
        valid_mask = y_raw.notna()
        X = X[valid_mask.values]
        y_raw = y_raw[valid_mask]

        if len(X) < 5:
            st.error("Not enough valid rows after cleaning target column.")
            return None

        le = LabelEncoder()
        y = le.fit_transform(y_raw.astype(str))
        target_encoder = le

        if len(np.unique(y)) < 2:
            st.error("Target column has only 1 unique class. Classification requires at least 2 classes.")
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

        if manual_params is None:
            manual_params = {}
        if param_grid is None:
            param_grid = {}
        try:
            cv_folds = max(2, int(cv_folds))
        except Exception:
            cv_folds = 5

        # --- Train model ---
        def _build_model(mp):
            if model_type == 'gaussian':
                return GaussianNB(var_smoothing=float(mp.get('var_smoothing', 1e-9)))
            elif model_type == 'multinomial':
                return MultinomialNB(
                    alpha=float(mp.get('alpha', 1.0)),
                    fit_prior=bool(mp.get('fit_prior', True))
                )
            else:  # bernoulli
                return BernoulliNB(
                    alpha=float(mp.get('alpha', 1.0)),
                    fit_prior=bool(mp.get('fit_prior', True)),
                    binarize=float(mp.get('binarize', 0.0))
                )

        if use_grid_search:
            st.info(f"Performing Grid Search for {model_type.upper()} Naive Bayes...")
            base_model = _build_model(manual_params)
            try:
                grid_search = GridSearchCV(
                    base_model,
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
                best_model = _build_model(manual_params)
                best_model.fit(X_train, y_train)
                best_params = manual_params
        else:
            st.info(f"Training {model_type.upper()} Naive Bayes with manual parameters...")
            try:
                best_model = _build_model(manual_params)
                best_model.fit(X_train, y_train)
                best_params = manual_params
            except Exception as fit_err:
                st.error(f"Model fit failed: {fit_err}")
                return None

        # --- Metrics ---
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)

        metrics_snapshot = {
            'Accuracy': float(accuracy),
            'Precision': float(precision),
            'Recall': float(recall),
            'F1 Score': float(f1),
            'Classification Report': class_report,
            'Confusion Matrix': cm.tolist(),
            'Best Parameters': best_params,
            'features': features,
            'target': target,
            'use_grid_search': use_grid_search,
            'cv_folds': cv_folds,
        }

        model_results = {
            'model': best_model,
            'scaler': scaler_used,
            'target_encoder': target_encoder,
            'metrics': metrics_snapshot,
            'features': features,
            'target': target,
            'model_type': model_type,
            'use_grid_search': use_grid_search,
            'cv_folds': cv_folds,
        }
        st.session_state.model_results = model_results

        param_list = [
            {"name": "features", "value": features},
            {"name": "target", "value": target},
            {"name": "model_type", "value": model_type},
            {"name": "use_grid_search", "value": use_grid_search},
            {"name": "cv_folds", "value": cv_folds}
        ]
        if use_grid_search:
            param_list.append({"name": "param_grid", "value": str(param_grid)})
        else:
            for pname, pval in manual_params.items():
                param_list.append({"name": pname, "value": pval})

        try:
            nb_model = DataManager.create_NB_Model(
                'Naive Bayes Classifier',
                param_list,
                st.session_state.selected_trans,
                metrics_snapshot
            )
        except Exception:
            nb_model = {
                "model name": "Naive Bayes Classifier",
                "model type": "Naive Bayes Classifier",
                "model param": param_list,
                "transformations": st.session_state.get("selected_trans", []),
                "metrics_snapshot": metrics_snapshot
            }

        if 'pipeline' not in st.session_state or not isinstance(st.session_state.pipeline, dict):
            st.session_state.pipeline = {}
        if 'ML' not in st.session_state.pipeline or not isinstance(st.session_state.pipeline.get('ML'), list):
            st.session_state.pipeline['ML'] = []

        model_name = 'Naive Bayes Classifier'
        if edit:
            st.session_state.pipeline['ML'] = [
                item if item.get('name') != model_name and item.get('model name') != model_name
                else nb_model
                for item in st.session_state.pipeline['ML']
            ]
        else:
            st.session_state.pipeline['ML'].append(nb_model)

        st.success(f"{model_type.upper()} Naive Bayes Model created successfully!")
        return model_results

    except Exception as e:
        st.error(f"Error training {model_type} Naive Bayes model: {str(e)}")
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

    if len(df) < 10:
        st.error("Insufficient data for Naive Bayes (minimum 10 samples required)")
        return False

    target_unique = df[target].nunique()
    if target_unique < 2:
        st.error("Target must have at least 2 classes for classification")
        return False

    return True