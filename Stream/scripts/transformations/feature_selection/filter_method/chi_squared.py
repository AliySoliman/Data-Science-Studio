import traceback

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2


def build_chi_squared(df, edit_values=None):
    """Build chi-squared feature selection UI"""
    st.write("### Chi-Squared Feature Selection")

    # Check if we have both features and target
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if len(numeric_columns) == 0:
        st.error("No numeric features found for chi-squared test")
        return {"k": 0, "target": None, "features": []}

    if len(categorical_columns) == 0:
        st.error("No categorical target variable found for chi-squared test")
        return {"k": 0, "target": None, "features": []}

    # Target selection
    target_options = categorical_columns
    default_target = edit_values.get("target", target_options[0]) if edit_values and target_options else target_options[0]

    target = st.selectbox(
        "Select target categorical variable",
        target_options,
        index=target_options.index(default_target) if default_target in target_options else 0
    )

    # Feature selection
    feature_options = numeric_columns
    default_features = edit_values.get("features", feature_options) if edit_values else feature_options

    selected_features = st.multiselect(
        "Select numeric features for chi-squared test",
        feature_options,
        default=default_features
    )

    # Number of features to select
    max_k = min(len(selected_features), 20) if selected_features else 1
    default_k = edit_values.get("k", min(5, max_k)) if edit_values else min(5, max_k)

    k = st.number_input(
        "Number of top features to select",
        min_value=1,
        max_value=max_k,
        value=default_k,
        key="chi2_k"
    )

    return {
        "k": k,
        "target": target,
        "features": selected_features
    }


def apply_chi_squared_selection(df, step):
    """Apply chi-squared feature selection"""
    df_copy = df.copy()

    try:
        k = step["k"]
        target = step["target"]
        features = step["features"]

        if not features or not target:
            st.warning("No features or target selected for chi-squared test")
            return df_copy

        if k > len(features):
            st.warning(f"k ({k}) cannot be greater than number of features ({len(features)})")
            k = len(features)

        # Prepare data — fill missing values first
        X = df_copy[features].fillna(0)
        y = df_copy[target]

        # chi2 requires non-negative values — shift columns that contain negatives
        col_mins = X.min()
        if (col_mins < 0).any():
            st.warning(
                "Chi-squared requires non-negative feature values. "
                "Shifting negative columns by their absolute minimum automatically."
            )
            # Only shift columns that are actually negative
            shift = col_mins.clip(upper=0).abs()
            X = X + shift

        # Ensure target is encoded if categorical
        if y.dtype == 'object':
            y = pd.factorize(y)[0]

        # Apply chi-squared test
        selector = SelectKBest(score_func=chi2, k=k)
        X_new = selector.fit_transform(X, y)

        # Get selected features
        selected_mask = selector.get_support()
        selected_features = [features[i] for i in range(len(features)) if selected_mask[i]]

        # Keep only selected features and target
        df_copy = df_copy[selected_features + [target]]

        # Display feature scores
        scores = selector.scores_
        feature_scores = {features[i]: scores[i] for i in range(len(features))}
        sorted_scores = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)

        st.info("Chi-squared feature scores:")
        for feature, score in sorted_scores[:k]:
            st.write(f"  - {feature}: {score:.4f}")

    except Exception as e:
        st.error(f"Error in chi-squared feature selection: {str(e)}")
        st.error("Make sure your target is categorical and features are numeric.")
        with st.expander("Details"):
            st.code(traceback.format_exc())
        return df_copy  # never crash the pipeline

    return df_copy
