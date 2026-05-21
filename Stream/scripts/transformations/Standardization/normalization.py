import streamlit as st
import pandas as pd
import numpy as np

def apply_normalization(df, step):
    """Mean normalization standardization"""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df if df is not None else pd.DataFrame()
    df = df.copy()
    col = step.get("column")
    if not col or col not in df.columns:
        st.warning(f"Column '{col}' not found for normalization. Skipping.")
        return df
    if not pd.api.types.is_numeric_dtype(df[col]):
        st.warning(f"Column '{col}' is not numeric. Skipping normalization.")
        return df
    try:
        col_range = df[col].max() - df[col].min()
        if col_range == 0:
            st.warning(f"Column '{col}' has zero range (all values identical). Skipping normalization.")
            return df
        df[col] = (df[col] - df[col].mean()) / col_range
    except Exception as e:
        st.error(f"Error in mean normalization: {str(e)}")
    return df


def build_normalization_transf(df, edit_values):
    st.subheader("Normalization Standardization Parameters")
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        st.warning("No data available for normalization.")
        return {"category": "Mean Normalization", "column": None}
    columns = df.select_dtypes(include=['number']).columns.tolist()
    if not columns:
        st.warning("No numeric columns found for normalization.")
        return {"category": "Mean Normalization", "column": None}
    default_column = edit_values.get("column", columns[0]) if edit_values and edit_values.get("column") in columns else columns[0]
    column = st.selectbox("Select Column to Normalize", columns,
                          index=columns.index(default_column) if default_column in columns else 0)
    return {
        "category": "Mean Normalization",
        "column": column
    }

