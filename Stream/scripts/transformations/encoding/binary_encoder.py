import streamlit as st
import pandas as pd
try:
    import category_encoders as ce
    _CE_AVAILABLE = True
except ImportError:
    _CE_AVAILABLE = False


def apply_binary_encoding(df, step):
    """Apply binary encoding to a categorical column."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df if df is not None else pd.DataFrame()
    df = df.copy()
    col = step.get("column")
    if not col or col not in df.columns:
        st.warning(f"Column '{col}' not found for binary encoding. Skipping.")
        return df
    if not _CE_AVAILABLE:
        st.error("category_encoders package is not installed. Cannot apply binary encoding.")
        return df
    try:
        encoder = ce.BinaryEncoder(cols=[col])
        df_encoded = encoder.fit_transform(df[col])
        df = pd.concat([df.drop(columns=[col]), df_encoded], axis=1)
    except Exception as e:
        st.error(f"Binary encoding failed for column '{col}': {str(e)}")
    return df


def build_binary_encoding_transf(df, edit_values=None):
    st.subheader("Binary Encoding Parameters")
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        st.warning("No data available for binary encoding.")
        return {"column": None}
    columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if not columns:
        st.warning("No categorical columns found for binary encoding.")
        return {"column": None}
    default_column = edit_values.get("column", columns[0]) if edit_values and edit_values.get("column") in columns else columns[0]
    column = st.selectbox(
        "Select Column for Binary Encoding",
        columns,
        index=columns.index(default_column) if default_column in columns else 0
    )
    return {"column": column}