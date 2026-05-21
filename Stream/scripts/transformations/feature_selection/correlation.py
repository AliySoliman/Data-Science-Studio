import streamlit as st
import pandas as pd
import numpy as np


def reduce_by_correlation(df, step):
    """Remove highly correlated numeric features above a correlation threshold."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df if df is not None else pd.DataFrame()

    # Accept both a raw threshold float (legacy) and a step dict
    if isinstance(step, dict):
        threshold = float(step.get("threshold", 0.9))
    else:
        threshold = float(step)

    try:
        numeric_df = df.select_dtypes(include='number')
        if numeric_df.shape[1] < 2:
            st.warning("Not enough numeric columns for correlation-based feature selection.")
            return df

        corr_matrix = numeric_df.corr().abs()
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
        df = df.drop(columns=[c for c in to_drop if c in df.columns])
        if to_drop:
            st.info(f"Dropped {len(to_drop)} highly correlated columns: {to_drop}")
    except Exception as e:
        st.error(f"Error in correlation feature selection: {str(e)}")

    return df


def build_corr_reduction_transf(df, edit_values=None) -> dict:
    default_threshold = float(edit_values.get("threshold", 0.9)) if edit_values else 0.9

    threshold = st.slider(
        "Correlation Threshold",
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        value=default_threshold
    )

    return {
        "category": "Correlation Feature Selection",
        "threshold": threshold
    }


