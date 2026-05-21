import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
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


def _default_manual_params():
    # Sensible defaults for HistGradientBoostingRegressor
    return {
        "max_iter": 200,
        "learning_rate": 0.05,
        "max_depth": 3,
        "min_samples_leaf": 20,
        "l2_regularization": 0.0,
    }


def _default_param_grid():
    return {
        "max_iter": [100, 200, 400],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5, None],
        "min_samples_leaf": [10, 20, 40],
        "l2_regularization": [0.0, 0.1, 1.0]
    }


def model_script(
    df,
    features,
    target,
    edit,
    use_grid_search=False,
    param_grid=None,
    manual_params=None,
    cv_folds=5,
):
    """
    Robust GBR script:
    - Works even if caller only passes df/features/target/edit
    - Supports optional GridSearchCV
    - Safeguards session_state pipeline keys
    """
    try:
        # --- Basic validation ---
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

        # --- Prepare data ---
        X = df[features].values

        # Handle infinite values
        X = pd.DataFrame(X, columns=features)
        X.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Target must be numerical for regression
        y = pd.to_numeric(df[target], errors="coerce").values
        if np.isnan(y).any():
            # Simple target imputation: drop rows where target is NaN
            target_mask = ~np.isnan(y)
            X = X[target_mask]
            y = y[target_mask]

        if len(X) < 5:
            st.error("Not enough valid rows after cleaning targets.")
            return None

        # --- Scale features & Impute ---
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

        # --- Train/test split ---
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        # --- Normalize config inputs ---
        use_grid_search = bool(use_grid_search)

        cv_folds = _safe_int(cv_folds, 5)
        if cv_folds < 2:
            cv_folds = 2

        if param_grid is None or (isinstance(param_grid, dict) and len(param_grid) == 0):
            param_grid = _default_param_grid()

        if manual_params is None or not isinstance(manual_params, dict):
            manual_params = _default_manual_params()
        else:
            # Fill missing keys with defaults; do NOT fall back to n_estimators (XGBoost key)
            defaults = _default_manual_params()
            for k, v in defaults.items():
                manual_params.setdefault(k, v)

        # --- Train model ---
        if use_grid_search:
            st.info("Performing Grid Search for optimal parameters...")

            base_model = HistGradientBoostingRegressor(random_state=42)
            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=cv_folds,
                scoring="neg_mean_squared_error",
                n_jobs=-1,
            )
            grid_search.fit(X_train, y_train)

            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_

            st.success(f"Grid Search completed! Best parameters: {best_params}")
        else:
            st.info("Training with manual parameters...")

            best_model = HistGradientBoostingRegressor(
                max_iter=_safe_int(manual_params.get("max_iter", 200), 200),
                learning_rate=_safe_float(manual_params.get("learning_rate", 0.05), 0.05),
                max_depth=_safe_int(manual_params.get("max_depth", 3), 3),
                min_samples_leaf=_safe_int(manual_params.get("min_samples_leaf", 20), 20),
                l2_regularization=_safe_float(manual_params.get("l2_regularization", 0.0), 0.0),
                random_state=42,
            )
            best_model.fit(X_train, y_train)
            best_params = manual_params

        # --- Predict ---
        y_pred = best_model.predict(X_test)

        # --- Metrics ---
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = float(np.sqrt(mse))

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

        # --- Build param list for pipeline ---
        param_list = [
            {"name": "features", "value": features},
            {"name": "target", "value": target},
            {"name": "use_grid_search", "value": use_grid_search},
            {"name": "cv_folds", "value": cv_folds},
        ]

        if use_grid_search:
            param_list.extend([
                {"name": "max_iter_range", "value": str(param_grid.get("max_iter", []))},
                {"name": "learning_rate_range", "value": str(param_grid.get("learning_rate", []))},
                {"name": "max_depth_range", "value": str(param_grid.get("max_depth", []))},
                {"name": "min_samples_leaf_range", "value": str(param_grid.get("min_samples_leaf", []))},
                {"name": "l2_regularization_range", "value": str(param_grid.get("l2_regularization", []))}
            ])
        else:
            param_list.extend([
                {"name": "max_iter", "value": manual_params.get("max_iter", 200)},
                {"name": "learning_rate", "value": manual_params.get("learning_rate", 0.05)},
                {"name": "max_depth", "value": manual_params.get("max_depth", 3)},
                {"name": "min_samples_leaf", "value": manual_params.get("min_samples_leaf", 20)},
                {"name": "l2_regularization", "value": manual_params.get("l2_regularization", 0.0)}
            ])

        # --- Robust pipeline/session key guard ---
        if "pipeline" not in st.session_state or not isinstance(st.session_state.pipeline, dict):
            st.session_state.pipeline = {}
        if "ML" not in st.session_state.pipeline or not isinstance(st.session_state.pipeline.get("ML"), list):
            st.session_state.pipeline["ML"] = []

        selected_trans = st.session_state.get("selected_trans", None)

        # --- Create pipeline entry (support both classmethod and instance method) ---
        try:
            GBR_model_pipeline_entry = DataManager.create_Gradient_Boosting_REG_Model(
                "GBR_Regressor",
                param_list,
                selected_trans,
                metrics_snapshot,
            )
        except Exception:
            GBR_model_pipeline_entry = data_manager.create_Gradient_Boosting_REG_Model(
                "GBR_Regressor",
                param_list,
                selected_trans,
                metrics_snapshot,
            )

        model_name_to_check = "GBR_Regressor"

        if edit:
            st.session_state.pipeline["ML"] = [
                item if item.get("model name") != model_name_to_check else GBR_model_pipeline_entry
                for item in st.session_state.pipeline["ML"]
            ]
        else:
            st.session_state.pipeline["ML"].append(GBR_model_pipeline_entry)

        st.success("Gradient Boosting Regressor Model created successfully and results saved to pipeline!")
        return model_results

    except Exception as e:
        st.error(f"Error training Gradient Boosting Regressor model: {str(e)}")
        return None


def validate_model(params):
    if len(params.get("features", [])) == 0:
        st.error("Please select at least one feature column")
        return False
    elif params.get("target") in params.get("features", []):
        st.error("Target column cannot be one of the features")
        return False
    return True
