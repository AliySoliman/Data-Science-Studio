import traceback

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.feature_selection import RFE, VarianceThreshold
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

try:
    from transformations.utils.robust_ml import clean_numeric_frame, infer_task_type
except ImportError:
    # Fallback if project path is not available
    def clean_numeric_frame(df):
        X = df.replace([float('inf'), float('-inf')], float('nan'))
        return X.apply(lambda col: col.fillna(col.median()), axis=0).fillna(0)

    def infer_task_type(y):
        if y is None:
            return "unsupervised", None
        y = y.replace([float('inf'), float('-inf')], float('nan')).dropna()
        if y.empty:
            return "unsupervised", None
        if not pd.api.types.is_numeric_dtype(y):
            y_enc = pd.Series(pd.factorize(y)[0], index=y.index)
            return "classification", y_enc
        nunique = y.nunique(dropna=True)
        if nunique <= 20:
            return "classification", y.astype(int, errors="ignore")
        return "regression", y.astype(float)


def build_ref_selection(df, edit_values=None):
    st.write("### Recursive Feature Elimination (RFE) — Regression")

    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_columns:
        st.error("No numeric features found for RFE.")
        return {"n_features": 0, "target": None, "features": [], "estimator_type": "linear"}

    target_options = df.columns.tolist()
    default_target = (edit_values.get("target") if edit_values else None)
    if default_target not in target_options:
        default_target = target_options[-1]  # often target is last

    target = st.selectbox(
        "Select target variable (regression target)",
        target_options,
        index=target_options.index(default_target),
        key="rfe_target"
    )

    feature_options = [c for c in numeric_columns if c != target]
    default_features = (edit_values.get("features", feature_options) if edit_values else feature_options)
    default_features = [c for c in default_features if c in feature_options]

    selected_features = st.multiselect(
        "Select features for RFE",
        feature_options,
        default=default_features,
        key="rfe_features"
    )

    max_features = max(1, min(len(selected_features), 50)) if selected_features else 1
    default_n = (edit_values.get("n_features") if edit_values else None)
    if default_n is None:
        default_n = min(5, max_features)
    default_n = int(np.clip(default_n, 1, max_features))

    n_features = st.number_input(
        "Number of features to select",
        min_value=1,
        max_value=int(max_features),
        value=int(default_n),
        step=1,
        key="rfe_n_features"
    )

    estimator_type = (edit_values.get("estimator_type", "linear") if edit_values else "linear")
    estimator_type = "tree" if estimator_type == "tree" else "linear"

    estimator_type = st.selectbox(
        "Estimator type",
        ["linear", "tree"],
        index=0 if estimator_type == "linear" else 1,
        key="rfe_estimator"
    )

    return {
        "n_features": int(n_features),
        "target": target,
        "features": selected_features,
        "estimator_type": estimator_type
    }


def apply_ref_selection(df, step):
    df_copy = df.copy()
    try:
        target = step.get("target")                 # can be None
        features = step.get("features", [])
        n_features = int(step.get("n_features", 5))
        estimator_family = step.get("estimator_type", "linear")  # linear|tree
        unsup_fallback = step.get("unsupervised_fallback", "variance")  # variance|skip
        keep_target = bool(step.get("keep_target", True))

        if not features:
            st.warning("No features selected for RFE.")
            return df_copy

        valid = [c for c in features if c in df_copy.columns]
        valid = df_copy[valid].select_dtypes(include=[np.number]).columns.tolist()
        if not valid:
            st.error("RFE requires numeric features (encode categoricals first).")
            return df_copy

        # Clamp n_features to the number of valid features actually available
        n_features = min(max(1, n_features), len(valid))
        if n_features < 1:
            st.error("No valid numeric features remain for RFE after filtering.")
            return df_copy

        X = clean_numeric_frame(df_copy[valid])

        y = df_copy[target] if (target and target in df_copy.columns) else None
        task, y_clean = infer_task_type(y)

        # --- Unsupervised case ---
        if task == "unsupervised":
            if unsup_fallback == "skip":
                st.warning("No valid target provided; skipping RFE (supervised).")
                return df_copy

            # Variance-based selection as fallback
            vt = VarianceThreshold()  # remove zero-variance features
            X_v = vt.fit_transform(X)
            kept = [f for f, keep in zip(valid, vt.get_support()) if keep]

            # If still too many, keep top-n by variance
            if len(kept) > n_features:
                variances = X[kept].var().sort_values(ascending=False)
                kept = variances.index[:n_features].tolist()

            out_cols = kept
            out = df_copy[out_cols].copy()
            st.info(f"✅ Unsupervised fallback selected {len(kept)} features (variance-based).")
            return out

        # --- Supervised case (classification/regression) ---
        # Align X/y (drop rows where y is NaN after cleaning)
        y2 = y_clean
        common_idx = X.index.intersection(y2.index)
        X2 = X.loc[common_idx]
        y2 = y2.loc[common_idx]

        if len(X2) < 3:
            st.error("Not enough rows after cleaning target for supervised RFE.")
            return df_copy

        n_features = min(max(1, n_features), len(valid))

        if estimator_family == "tree":
            estimator = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1) if task == "classification" \
                        else RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
        else:
            estimator = LogisticRegression(max_iter=2000, random_state=42) if task == "classification" \
                        else Ridge(alpha=1.0)

        selector = RFE(estimator=estimator, n_features_to_select=n_features)
        selector.fit(X2, y2)

        support = selector.get_support()
        chosen = [f for f, keep in zip(valid, support) if keep]

        out_cols = chosen + ([target] if (keep_target and target in df_copy.columns) else [])
        out = df_copy[out_cols].copy()

        # Ranking display
        rank_df = pd.DataFrame({"feature": valid, "rank": selector.ranking_}).sort_values("rank")
        st.info(f"✅ RFE ({task}) selected {len(chosen)} features.")
        st.dataframe(rank_df, use_container_width=True)

        return out

    except Exception as e:
        st.error(f"RFE failed: {e}")
        with st.expander("Details"):
            st.code(traceback.format_exc())
        return df_copy
