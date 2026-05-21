import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from constants import DataManager

data_manager = DataManager()


def model_script(df, features, target, edit, use_grid_search, param_grid, manual_params, cv_folds):
    try:
        # Clean features
        X = df[features].copy()
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X = X.fillna(X.mean())  # Gradient Boosting does NOT handle NaN natively
        X = X.values

        y = df[target].values

        if df[target].dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
            target_encoder = le
        else:
            target_encoder = None
            unique_y = np.unique(y)
            if not np.array_equal(unique_y, np.arange(len(unique_y))):
                le = LabelEncoder()
                y = le.fit_transform(y)
                target_encoder = le

        # Stratified split with fallback
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

        if use_grid_search:
            st.info("Performing Grid Search for optimal parameters...")

            gbc = GradientBoostingClassifier(random_state=42)
            grid_search = GridSearchCV(
                gbc,
                param_grid,
                cv=cv_folds,
                scoring='accuracy',
                n_jobs=1
            )
            grid_search.fit(X_train, y_train)

            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_

            st.success(f"Grid Search completed! Best parameters: {best_params}")

        else:
            st.info("Training with manual parameters...")

            manual_params_with_defaults = {
                'n_estimators': manual_params.get('n_estimators', 100),
                'learning_rate': manual_params.get('learning_rate', 0.1),
                'max_depth': manual_params.get('max_depth', 3),
                'min_samples_split': manual_params.get('min_samples_split', 2),
                'min_samples_leaf': manual_params.get('min_samples_leaf', 1),
                'subsample': manual_params.get('subsample', 1.0),
                'random_state': 42
            }

            best_model = GradientBoostingClassifier(**manual_params_with_defaults)
            best_model.fit(X_train, y_train)

            filtered_params = manual_params_with_defaults.copy()
            filtered_params.pop('random_state', None)
            best_params = filtered_params

        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)

        feature_importances = best_model.feature_importances_.tolist()

        metrics_snapshot = {
            'Accuracy': float(accuracy),
            'Classification Report': class_report,
            'Confusion Matrix': conf_matrix.tolist(),
            'Best Parameters': best_params,
            'features': features,
            'target': target,
            'use_grid_search': use_grid_search,
            'cv_folds': cv_folds,
            'feature_importances': feature_importances
        }

        model_results = {
            'model': best_model,
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
            {"name": "n_estimators_range", "value": str(param_grid.get('n_estimators', [])) if use_grid_search else manual_params.get('n_estimators', 100)},
            {"name": "learning_rate_range", "value": str(param_grid.get('learning_rate', [])) if use_grid_search else manual_params.get('learning_rate', 0.1)},
            {"name": "max_depth_range", "value": str(param_grid.get('max_depth', [])) if use_grid_search else manual_params.get('max_depth', 3)},
            {"name": "min_samples_split_range", "value": str(param_grid.get('min_samples_split', [])) if use_grid_search else manual_params.get('min_samples_split', 2)},
            {"name": "min_samples_leaf_range", "value": str(param_grid.get('min_samples_leaf', [])) if use_grid_search else manual_params.get('min_samples_leaf', 1)},
            {"name": "subsample_range", "value": str(param_grid.get('subsample', [])) if use_grid_search else manual_params.get('subsample', 1.0)},
            {"name": "cv_folds", "value": cv_folds}
        ]

        try:
            GBC_model_pipeline_entry = DataManager.create_gbc_Model(
                "Gradient Boosting Classifier",
                param_list,
                st.session_state.get('selected_trans', []),
                best_model,
                metrics_snapshot
            )
        except Exception:
            GBC_model_pipeline_entry = data_manager.create_gbc_Model(
                "Gradient Boosting Classifier",
                param_list,
                st.session_state.get('selected_trans', []),

                best_model,
                metrics_snapshot
            )

        if 'pipeline' not in st.session_state or not isinstance(st.session_state.pipeline, dict):
            st.session_state.pipeline = {}
        if 'ML' not in st.session_state.pipeline or not isinstance(st.session_state.pipeline.get('ML'), list):
            st.session_state.pipeline['ML'] = []

        if edit:
            st.session_state.pipeline['ML'] = [
                item if item.get('name') != 'Gradient Boosting Classifier' and item.get('model name') != 'Gradient Boosting Classifier'
                else GBC_model_pipeline_entry
                for item in st.session_state.pipeline['ML']
            ]
        else:
            st.session_state.pipeline['ML'].append(GBC_model_pipeline_entry)

        st.success("Gradient Boosting Classifier Model created successfully and results saved to pipeline!")
        return model_results

    except Exception as e:
        st.error(f"Error training Gradient Boosting Classifier model: {str(e)}")
        return None


def validate_model(params):
    if len(params.get('features', [])) == 0:
        st.error("Please select at least one feature column")
        return False
    elif params.get('target') in params.get('features', []):
        st.error("Target column cannot be one of the features")
        return False

    target_values = params.get('df')[params.get('target')].nunique()
    if target_values < 2:
        st.error("Target column must have at least 2 unique classes for classification")
        return False

    return True