import streamlit as st
import pandas as pd


def apply_target_encoding(df, step):
    """Apply target encoding: replace each category with the mean of the numeric target."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df if df is not None else pd.DataFrame()
    df_encoded = df.copy()
    target_column = step.get('target_column')
    columns = step.get('columns', [])

    if not target_column or target_column not in df_encoded.columns:
        st.warning(f"Target column '{target_column}' not found. Skipping target encoding.")
        return df_encoded

    if not pd.api.types.is_numeric_dtype(df_encoded[target_column]):
        st.warning(f"Target column '{target_column}' is not numeric. Skipping target encoding.")
        return df_encoded

    for col in columns:
        if col not in df_encoded.columns:
            st.warning(f"Column '{col}' not found in dataframe. Skipping.")
            continue
        try:
            mean_map = df_encoded.groupby(col)[target_column].mean()
            df_encoded[col] = df_encoded[col].map(mean_map)
        except Exception as e:
            st.warning(f"Target encoding failed for column '{col}': {str(e)}")
    return df_encoded


def build_target_encoding_transf(df, edit_values=None) -> dict:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        st.warning("No data available for target encoding.")
        return {"columns": [], "target_column": None}

    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    numeric_cols = df.select_dtypes(include='number').columns.tolist()

    if not categorical_cols:
        st.warning("No categorical columns found for target encoding.")
        return {"columns": [], "target_column": None}
    if not numeric_cols:
        st.warning("No numeric target columns found for target encoding.")
        return {"columns": [], "target_column": None}

    default_cols = edit_values.get("columns", []) if edit_values else []
    default_cols = [c for c in default_cols if c in categorical_cols]

    selected_columns = st.multiselect(
        "Select columns for Target Encoding",
        options=categorical_cols,
        default=default_cols
    )

    default_target = edit_values.get("target_column") if edit_values else None
    target_idx = numeric_cols.index(default_target) if default_target in numeric_cols else 0
    target_column = st.selectbox(
        "Select target column",
        options=numeric_cols,
        index=target_idx
    )

    return {
        "columns": selected_columns,
        "target_column": target_column
    }
