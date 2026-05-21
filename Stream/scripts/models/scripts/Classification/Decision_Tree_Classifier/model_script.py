import streamlit as st
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from constants import DataManager
import pandas as pd
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from transformations.utils.robust_ml import clean_numeric_frame, make_columns_unique

data_manager = DataManager()

DT_PARAM_GRID = {
    'max_depth': [None, 5, 10],
    'min_samples_leaf': [1, 5, 10],
    'criterion': ['gini', 'entropy']
}

def model_script(df, features, target, edit, use_grid_search, max_depth, min_samples_leaf, random_state):
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

        # --- Prepare X: use get_dummies for categorical, then clean ---
        df_copy = df.copy()
        X_raw = df_copy[features]
        X_encoded = pd.get_dummies(X_raw, drop_first=True)
        X_encoded = clean_numeric_frame(X_encoded)
        encoded_features = make_columns_unique(X_encoded.columns.tolist())
        X_encoded.columns = encoded_features

        # --- Build y ---
        y_raw = df_copy[target].replace([np.inf, -np.inf], np.nan)
        valid_mask = y_raw.notna()
        X_final = X_encoded[valid_mask.values].values
        y_raw = y_raw[valid_mask]

        if len(X_final) < 5:
            st.error("Not enough valid rows after cleaning target column.")
            return None

        le = LabelEncoder()
        if y_raw.dtype == 'object' or y_raw.dtype.name == 'category':
            y_final = le.fit_transform(y_raw.astype(str))
        else:
            y_final = le.fit_transform(y_raw.astype(str))

        if len(np.unique(y_final)) < 2:
            st.error("Target column has only 1 unique class. Classification requires at least 2 classes.")
            return None

        # --- Train/test split with stratified fallback ---
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_final, y_final, test_size=0.2, random_state=42, stratify=y_final
            )
        except ValueError:
            st.warning(
                "Stratified split failed (likely a class with too few samples). "
                "Falling back to random split."
            )
            X_train, X_test, y_train, y_test = train_test_split(
                X_final, y_final, test_size=0.2, random_state=42
            )

        if len(X_test) == 0:
            st.error("Test set is empty after splitting. Provide more data.")
            return None

        # --- Train model ---
        if use_grid_search:
            st.subheader("Grid Search Optimization")
            st.write("Running Grid Search for best Decision Tree hyperparameters...")
            dt_model = DecisionTreeClassifier(random_state=42)
            try:
                grid_search = GridSearchCV(
                    estimator=dt_model,
                    param_grid=DT_PARAM_GRID,
                    cv=5,
                    scoring='accuracy',
                    n_jobs=-1,
                    verbose=1
                )
                grid_search.fit(X_train, y_train)
                model = grid_search.best_estimator_
                best_model = model
                st.success(f"Grid Search complete. Best parameters found: {grid_search.best_params_}")
            except Exception as gs_err:
                st.warning(f"Grid search failed ({gs_err}). Falling back to manual parameters.")
                model = DecisionTreeClassifier(
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    random_state=random_state or 42
                )
                best_model = model.fit(X_train, y_train)
                model = best_model  # keep reference consistent
        else:
            st.subheader("Default Model Training")
            try:
                model = DecisionTreeClassifier(
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    random_state=random_state or 42
                )
                best_model = model.fit(X_train, y_train)
                st.success("Default Decision Tree Classifier trained.")
            except Exception as fit_err:
                st.error(f"Model fit failed: {fit_err}")
                return None

        # --- Metrics ---
        y_pred = model.predict(X_test)
        unique_classes = np.unique(y_test)
        average_type = 'binary' if len(unique_classes) <= 2 else 'weighted'

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average=average_type, zero_division=0)
        recall = recall_score(y_test, y_pred, average=average_type, zero_division=0)
        f1 = f1_score(y_test, y_pred, average=average_type, zero_division=0)
        conf_matrix = confusion_matrix(y_test, y_pred)

        metrics_snapshot = {
            'Accuracy': float(accuracy),
            'Precision': float(precision),
            'Recall': float(recall),
            'F1 Score': float(f1),
            'Confusion Matrix': conf_matrix.tolist(),
            'features': encoded_features,
            'feature_importances': model.feature_importances_.tolist() if hasattr(model, 'feature_importances_') else [],
            'target': target,
            'use_grid_search': use_grid_search,
        }

        model_results = {
            'model': best_model,
            'target_encoder': le,
            'metrics': metrics_snapshot,
            'features': encoded_features,
            'target': target,
            'use_grid_search': use_grid_search
        }
        st.session_state.model_results = model_results

        param_list = [
            {"name": "features", "value": features},
            {"name": "target", "value": target},
            {"name": "use_grid_search", "value": use_grid_search}
        ]

        try:
            DT_model = DataManager.create_DecisionTree_Model(
                "Decision Tree Classifier",
                param_list,
                st.session_state.selected_trans,
                best_model,
                metrics_snapshot=metrics_snapshot
            )
        except Exception:
            DT_model = data_manager.create_DecisionTree_Model(
                "Decision Tree Classifier",
                param_list,
                st.session_state.selected_trans,
                best_model,
                metrics_snapshot=metrics_snapshot
            )

        if 'pipeline' not in st.session_state or not isinstance(st.session_state.pipeline, dict):
            st.session_state.pipeline = {}
        if 'ML' not in st.session_state.pipeline or not isinstance(st.session_state.pipeline.get('ML'), list):
            st.session_state.pipeline['ML'] = []

        if edit:
            st.session_state.pipeline['ML'] = [
                item if item.get('name') != 'Decision Tree Classifier' and item.get('model name') != 'Decision Tree Classifier'
                else DT_model
                for item in st.session_state.pipeline['ML']
            ]
        else:
            st.session_state.pipeline['ML'].append(DT_model)

        st.success("Decision Tree model created successfully!")
        return model_results

    except Exception as e:
        st.error(f"Error training Decision Tree model: {str(e)}")
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

    target_data = df[target]
    unique_classes = target_data.nunique()
    if unique_classes < 2:
        st.error("Target column must have at least 2 unique classes for classification")
        return False

    return True