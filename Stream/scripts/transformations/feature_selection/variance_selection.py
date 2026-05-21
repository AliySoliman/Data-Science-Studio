import streamlit as st
import pandas as pd
import numpy as np


def reduce_by_variance(df, step):
    """Remove low-variance numeric features."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df if df is not None else pd.DataFrame()

    # Accept both a raw threshold float (legacy) and a step dict
    if isinstance(step, dict):
        threshold = float(step.get("threshold", 0.01))
    else:
        threshold = float(step)

    try:
        numeric_df = df.select_dtypes(include='number')
        if numeric_df.empty:
            st.warning("No numeric columns available for variance-based feature selection.")
            return df

        variances = numeric_df.var()
        low_var_cols = variances[variances < threshold].index.tolist()
        df = df.drop(columns=[c for c in low_var_cols if c in df.columns])
        if low_var_cols:
            st.info(f"Dropped {len(low_var_cols)} low-variance columns: {low_var_cols}")
    except Exception as e:
        st.error(f"Error in variance feature selection: {str(e)}")

    return df


def build_variance_reduction_transf(df, edit_values=None) -> dict:
    default_threshold = float(edit_values.get("threshold", 0.01)) if edit_values else 0.01

    threshold = st.number_input(
        "Variance Threshold",
        min_value=0.0,
        value=default_threshold,
        step=0.001
    )

    return {
        "category": "Variance Feature Selection",
        "threshold": threshold
    }

