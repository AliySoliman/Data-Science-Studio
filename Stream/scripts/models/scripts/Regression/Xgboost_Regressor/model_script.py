import streamlit as st
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from constants import DataManager

data_manager = DataManager()

def _safe_int(x, default):
    try:
        return int(x)
    except Exception:
        return int(default)

def _safe_float(x, default):
    try:
        return float(x)
    except Exception:
        return float(default)

def model_script(df, features, target, edit, use_grid_search=False, param_grid=None, manual_params=None, cv_folds=5):
    try:
        if df is None or not isinstance(df, pd.DataFrame):
            st.error("Invalid dataframe provided.")
            return None

        if not features or len(features) == 0:
            st.error("Please select at least one feature column.")
            return None

        if target is None or target == "":
            st.error("Please select a target column.")
            return None

        if target in features:
            st.error("Target column cannot be one of the features.")
            return None

        X = df[features].values
        X = pd.DataFrame(X, columns=features)
        X.replace([np.inf, -np.inf], np.nan, inplace=True)

        y = pd.to_numeric(df[target], errors="coerce").values
        if np.isnan(y).any():
            target_mask = ~np.isnan(y)
            X = X[target_mask]
            y = y[target_mask]

        if len(X) < 5:
            st.error("Not enough valid rows after cleaning targets.")
            return None

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

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        use_grid_search = bool(use_grid_search)
        cv_folds = _safe_int(cv_folds, 5)
        if cv_folds < 2:
            cv_folds = 2

        if use_grid_search:
            st.info("Performing Grid Search for optimal parameters...")

            base_model = XGBRegressor(random_state=42)
            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=cv_folds,
                scoring="neg_mean_squared_error",
                n_jobs=1,
            )
            grid_search.fit(X_train, y_train)

            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_

            st.success(f"Grid Search completed! Best parameters: {best_params}")
        else:
            st.info("Training with manual parameters...")

            best_model = XGBRegressor(
                n_estimators=_safe_int(manual_params.get("n_estimators"), 100),
                learning_rate=_safe_float(manual_params.get("learning_rate"), 0.1),
                max_depth=_safe_int(manual_params.get("max_depth"), 3),
                min_child_weight=_safe_int(manual_params.get("min_child_weight"), 1),
                gamma=_safe_float(manual_params.get("gamma"), 0.0),
                random_state=42,
            )
            best_model.fit(X_train, y_train)
            best_params = manual_params

        y_pred = best_model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = float(np.sqrt(mse))
        
        # Store feature importances as a dictionary
        feature_importance_dict = {
            feat: float(imp) for feat, imp in zip(features, best_model.feature_importances_)
        }

        metrics_snapshot = {
            "R2 Score": float(r2),
            "MAE": float(mae),
            "MSE": float(mse),
            "RMSE": float(rmse),
            "Best Parameters": best_params,
            "features": features,
            "target": target,
            "use_grid_search": use_grid_search,
            "cv_folds": cv_folds,
            "feature_importance": feature_importance_dict,
            "y_test": y_test.tolist(),
            "y_pred": y_pred.tolist()
        }

        model_results = {
            "model": best_model,
            "scaler": preprocessor,
            "target_encoder": None,
            "metrics": metrics_snapshot,
            "features": features,
            "target": target,
            "use_grid_search": use_grid_search,
            "cv_folds": cv_folds,
        }

        st.session_state.model_results = model_results

        param_list = [
            {"name": "features", "value": features},
            {"name": "target", "value": target},
            {"name": "use_grid_search", "value": use_grid_search},
            {"name": "cv_folds", "value": cv_folds},
        ]

        if use_grid_search:
            param_list.extend([
                {"name": "n_estimators_range", "value": str(param_grid.get("n_estimators", []))},
                {"name": "learning_rate_range", "value": str(param_grid.get("learning_rate", []))},
                {"name": "max_depth_range", "value": str(param_grid.get("max_depth", []))},
                {"name": "min_child_weight_range", "value": str(param_grid.get("min_child_weight", []))},
                {"name": "gamma_range", "value": str(param_grid.get("gamma", []))}
            ])
        else:
            param_list.extend([
                {"name": "n_estimators", "value": manual_params.get("n_estimators")},
                {"name": "learning_rate", "value": manual_params.get("learning_rate")},
                {"name": "max_depth", "value": manual_params.get("max_depth")},
                {"name": "min_child_weight", "value": manual_params.get("min_child_weight")},
                {"name": "gamma", "value": manual_params.get("gamma")}
            ])

        if "pipeline" not in st.session_state or not isinstance(st.session_state.pipeline, dict):
            st.session_state.pipeline = {}

        if "ML" not in st.session_state.pipeline or not isinstance(st.session_state.pipeline.get("ML"), list):
            st.session_state.pipeline["ML"] = []

        selected_trans = st.session_state.get("selected_trans", None)

        try:
            XGB_model_pipeline_entry = DataManager.create_XGBoost_REG_Model(
                "XGBoost Regressor",
                param_list,
                selected_trans,
                best_model,
                metrics_snapshot,
            )
        except Exception:
            XGB_model_pipeline_entry = data_manager.create_XGBoost_REG_Model(
                "XGBoost Regressor",
                param_list,
                selected_trans,
                best_model,
                metrics_snapshot,
            )

        model_name_to_check = "XGBoost Regressor"

        if edit:
            st.session_state.pipeline["ML"] = [
                item if item.get("model name") != model_name_to_check and item.get("name") != model_name_to_check else XGB_model_pipeline_entry
                for item in st.session_state.pipeline["ML"]
            ]
        else:
            st.session_state.pipeline["ML"].append(XGB_model_pipeline_entry)

        st.success("XGBoost Regressor Model created successfully and results saved to pipeline!")
        return model_results

    except Exception as e:
        st.error(f"Error training XGBoost Regressor model: {str(e)}")
        return None

def validate_model(params):
    if len(params.get("features", [])) == 0:
        st.error("Please select at least one feature column")
        return False
    elif params.get("target") in params.get("features", []):
        st.error("Target column cannot be one of the features")
        return False
    return True
