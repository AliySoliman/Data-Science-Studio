import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from constants import DataManager

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from transformations.utils.robust_ml import clean_numeric_frame, make_columns_unique, scale_array

data_manager = DataManager()

def model_script(df, features, target, model_type, alpha=1.0, edit=False):
    """
    Regression model script with default parameters to avoid missing arguments
    """
    try:
        # --- Basic validation ---
        if df is None or not isinstance(df, pd.DataFrame) or len(df) == 0:
            st.error("Invalid or empty dataframe provided.")
            return {"status": "error", "message": "Invalid dataframe"}

        if not features or len(features) == 0:
            st.error("Please select features and target")
            return {"status": "error", "message": "Missing features or target"}

        if not target:
            st.error("Please select a target column")
            return {"status": "error", "message": "Missing target"}

        if target in features:
            st.error("Target cannot be in features")
            return {"status": "error", "message": "Target in features"}

        # Filter features that exist
        features = [f for f in features if f in df.columns]
        features = make_columns_unique(features)

        if not features:
            st.error("None of the selected feature columns exist in the dataframe.")
            return {"status": "error", "message": "No valid features"}

        if target not in df.columns:
            st.error(f"Target column '{target}' not found in dataframe.")
            return {"status": "error", "message": "Target not found"}

        if alpha is None:
            alpha = 1.0

        # --- Build X ---
        X_df = df[features].copy()

        # Handle categorical features with LabelEncoder
        feature_encoders = {}
        categorical_features = X_df.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_features:
            st.info(f"Encoding categorical features: {categorical_features}")
            for col in categorical_features:
                le = LabelEncoder()
                X_df[col] = le.fit_transform(X_df[col].astype(str))
                feature_encoders[col] = le

        X_df = clean_numeric_frame(X_df)

        # Scale for regularization models
        if model_type in ['ridge', 'lasso']:
            X = scale_array(X_df.values, "standard")
        else:
            X = X_df.values

        # --- Build y ---
        y_series = pd.to_numeric(df[target], errors='coerce')
        y_series = y_series.replace([np.inf, -np.inf], np.nan)
        valid_mask = y_series.notna()
        X = X[valid_mask.values]
        y = y_series[valid_mask].values.astype(float)

        if len(X) < 5:
            st.error("Not enough valid rows after cleaning target column.")
            return {"status": "error", "message": "Too few rows"}

        # --- Split ---
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        if len(X_test) == 0:
            st.error("Test set is empty after splitting.")
            return {"status": "error", "message": "Empty test set"}

        # --- Train model ---
        if model_type == 'linear':
            model = LinearRegression()
            st.info("Training Linear Regression model...")
        elif model_type == 'ridge':
            model = Ridge(alpha=float(alpha))
            st.info(f"Training Ridge Regression model (alpha={alpha})...")
        elif model_type == 'lasso':
            model = Lasso(alpha=float(alpha), max_iter=5000)
            st.info(f"Training Lasso Regression model (alpha={alpha})...")
        else:
            st.error(f"Unknown model type: {model_type}")
            return {"status": "error", "message": f"Unknown model type: {model_type}"}

        try:
            best_model = model.fit(X_train, y_train)
        except Exception as fit_err:
            st.error(f"Model fit failed: {fit_err}")
            return {"status": "error", "message": str(fit_err)}

        # --- Predictions & metrics ---
        y_pred = model.predict(X_test)

        r2 = float(r2_score(y_test, y_pred)) if len(y_test) > 0 else 0.0
        mae = float(mean_absolute_error(y_test, y_pred)) if len(y_test) > 0 else 0.0
        mse = float(mean_squared_error(y_test, y_pred)) if len(y_test) > 0 else 0.0
        rmse = float(np.sqrt(mse)) if mse > 0 else 0.0

        if len(y_test) > 0 and not np.all(y_test == 0):
            with np.errstate(divide='ignore', invalid='ignore'):
                mape = float(np.nanmean(np.abs(np.where(y_test != 0, (y_test - y_pred) / y_test, 0.0))) * 100)
        else:
            mape = 0.0

        if len(y_test) > len(features) + 1:
            adjusted_r2 = float(1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - len(features) - 1))
        else:
            adjusted_r2 = r2

        coefficients = {}
        if hasattr(model, 'coef_'):
            coefficients = {str(feat): float(coef) for feat, coef in zip(features, model.coef_)}
        intercept = float(model.intercept_) if hasattr(model, 'intercept_') else 0.0

        metrics_snapshot = {
            'R² Score': r2,
            'Adjusted R²': adjusted_r2,
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape,
            'Coefficients': coefficients,
            'Intercept': intercept,
            'model_type': str(model_type),
            'features': [str(f) for f in features],
            'target': str(target),
            'alpha': float(alpha),
            'y_test': y_test.tolist(),
            'y_pred': y_pred.tolist()
        }

        model_results = {
            'model': best_model,
            'target_encoder': None,
            'metrics': metrics_snapshot,
            'features': features,
            'target': target,
        }
        st.session_state.model_results = model_results

        param_list = [
            {"name": "features", "value": [str(f) for f in features]},
            {"name": "target", "value": str(target)},
            {"name": "model_type", "value": str(model_type)},
            {"name": "alpha", "value": float(alpha)}
        ]

        try:
            regression_model = DataManager.create_REG_Model(
                "Linear Regression",
                param_list,
                st.session_state.get('selected_trans', []),
                best_model,
                metrics_snapshot=metrics_snapshot
            )
        except Exception:
            regression_model = data_manager.create_REG_Model(
                "Linear Regression",
                param_list,
                st.session_state.get('selected_trans', []),
                best_model,
                metrics_snapshot=metrics_snapshot
            )

        if 'pipeline' not in st.session_state or not isinstance(st.session_state.pipeline, dict):
            st.session_state.pipeline = {}
        if 'ML' not in st.session_state.pipeline or not isinstance(st.session_state.pipeline.get('ML'), list):
            st.session_state.pipeline['ML'] = []

        if edit:
            model_replaced = False
            for i, item in enumerate(st.session_state.pipeline['ML']):
                if item.get('model name') == "Linear Regression" or item.get('name') == "Linear Regression":
                    st.session_state.pipeline['ML'][i] = regression_model
                    model_replaced = True
                    break
            if not model_replaced:
                st.session_state.pipeline['ML'].append(regression_model)
        else:
            st.session_state.pipeline['ML'].append(regression_model)

        st.success("Regression Model created successfully!")
        return {"status": "success", "metrics_snapshot": metrics_snapshot}

    except Exception as e:
        st.error(f"Error training Regression model: {str(e)}")
        return {"status": "error", "message": str(e)}


def validate_model(params):
    """Validate model parameters with safe defaults"""
    features = params.get('features', [])
    df = params.get('df')

    if df is None or (hasattr(df, '__len__') and len(df) == 0):
        st.error("No data available.")
        return False

    if not features or len(features) == 0:
        st.error("Please select at least one feature column")
        return False

    target = params.get('target')
    if not target:
        st.error("Please select a target column")
        return False

    if target in features:
        st.error("Target column cannot be one of the features")
        return False

    if target not in df.columns:
        st.error(f"Target column '{target}' not found in dataframe.")
        return False

    missing = [f for f in features if f not in df.columns]
    if missing:
        st.error(f"Features not found in dataframe: {missing}")
        return False

    # Check target is numeric or can be coerced
    y_numeric = pd.to_numeric(df[target], errors='coerce')
    if y_numeric.isna().all():
        st.error("Target column must be numeric for regression")
        return False

    model_type = params.get('model_type', 'linear')
    if model_type in ['ridge', 'lasso']:
        alpha = params.get('alpha', 1.0)
        try:
            alpha_val = float(alpha)
            if alpha_val <= 0:
                st.error("Alpha value must be positive for Ridge and Lasso regression")
                return False
        except (ValueError, TypeError):
            st.error("Alpha value must be a valid number")
            return False

    return True