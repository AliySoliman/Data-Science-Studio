import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif


def apply_ANOVA(df, step):
    """Apply ANOVA dimensionality reduction (supervised feature selection)."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df if df is not None else pd.DataFrame()

    target_col = step.get("target_column")
    k = int(step.get("k", 5))

    if not target_col or target_col not in df.columns:
        st.warning(f"Target column '{target_col}' not found for ANOVA. Skipping.")
        return df

    try:
        X_df = df.select_dtypes(include=['number']).copy()

        # Remove target from features if it's numeric
        if target_col in X_df.columns:
            X_df = X_df.drop(columns=[target_col])

        if X_df.empty:
            st.warning("No numeric feature columns found for ANOVA.")
            return df

        # Remove inf/nan from features
        X_df = X_df.replace([np.inf, -np.inf], np.nan)
        X_df = X_df.fillna(X_df.median())

        y = df[target_col]

        # Align rows where target is not null
        valid_mask = y.notna()
        X_valid = X_df[valid_mask]
        y_valid = y[valid_mask]

        if len(X_valid) < 5:
            st.warning("Not enough rows after cleaning target for ANOVA.")
            return df

        # Encode target if categorical
        if not pd.api.types.is_numeric_dtype(y_valid):
            y_valid = pd.factorize(y_valid)[0]

        # Clamp k
        k = max(1, min(k, X_valid.shape[1]))

        selector = SelectKBest(score_func=f_classif, k=k)
        selector.fit(X_valid, y_valid)

        selected_columns = X_df.columns[selector.get_support(indices=True)].tolist()
        df_reduced = df[[c for c in selected_columns + [target_col] if c in df.columns]]

        st.success(f"ANOVA selected {len(selected_columns)} features: {selected_columns}")
        return df_reduced

    except Exception as e:
        st.error(f"Error in ANOVA feature selection: {str(e)}")
        return df


def build_anova_transf(df, edit_values=None):
    st.subheader("ANOVA Dimensionality Reduction Parameters")

    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        st.warning("No data available for ANOVA.")
        return {"category": "ANOVA", "target_column": None, "k": 1}

    columns = df.select_dtypes(include=['number']).columns.tolist()
    target_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if not target_columns:
        # Fall back to low-cardinality numeric columns
        target_columns = [c for c in df.columns if df[c].nunique() <= 20 and c not in columns]

    if not target_columns:
        st.warning("No suitable target columns found for ANOVA (need categorical or low-cardinality column).")
        return {"category": "ANOVA", "target_column": None, "k": 1}

    if not columns:
        st.warning("No numeric feature columns found for ANOVA.")
        return {"category": "ANOVA", "target_column": None, "k": 1}

    edit_values = edit_values or {}
    default_target = edit_values.get("target_column", target_columns[0])
    target_column = st.selectbox(
        "Select Target Column",
        target_columns,
        index=target_columns.index(default_target) if default_target in target_columns else 0
    )

    max_k = max(1, len(columns))
    default_k = int(np.clip(edit_values.get("k", min(5, max_k)), 1, max_k))
    k = st.number_input("Number of Features to Select (k)", min_value=1, max_value=max_k, value=default_k)

    return {
        "category": "ANOVA",
        "target_column": target_column,
        "k": int(k)
    }

