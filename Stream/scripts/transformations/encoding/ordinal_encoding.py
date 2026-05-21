import traceback

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder


def build_ordinal_encoding(df, edit_values=None):
    """Build ordinal encoding UI"""
    st.write("### Ordinal Encoding for Categorical Columns")

    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if not categorical_columns:
        st.error("No categorical columns found for ordinal encoding")
        return {"columns": [], "mappings": {}, "handle_unknown": "use_encoded_value", "unknown_value": -1}

    default_columns = edit_values.get("columns", categorical_columns) if edit_values else categorical_columns
    # Safety: only keep defaults that still exist as categorical columns
    default_columns = [c for c in default_columns if c in categorical_columns]

    selected_columns = st.multiselect(
        "Select categorical columns for ordinal encoding",
        categorical_columns,
        default=default_columns
    )

    encoding_mappings = {}
    for col in selected_columns:
        st.write(f"**{col}**")
        unique_values = df[col].dropna().unique()

        if edit_values and "mappings" in edit_values and col in edit_values["mappings"]:
            default_order = edit_values["mappings"][col]
        else:
            default_order = sorted(unique_values) if len(unique_values) < 20 else unique_values[:20]

        order_input = st.text_area(
            f"Enter order for '{col}' (comma-separated, found {len(unique_values)} unique values)",
            value=", ".join(map(str, default_order)),
            key=f"ordinal_{col}"
        )

        try:
            order_list = [x.strip() for x in order_input.split(",") if x.strip()]
            if len(order_list) > 0:
                encoding_mappings[col] = order_list
        except Exception as e:
            st.error(f"Error parsing order for {col}: {str(e)}")

    # Fixed: two valid options, safe index lookup so edit mode never throws IndexError
    handle_unknown_options = ["use_encoded_value", "error"]
    default_handle_unknown = (
        edit_values.get("handle_unknown", "use_encoded_value") if edit_values else "use_encoded_value"
    )
    safe_index = (
        handle_unknown_options.index(default_handle_unknown)
        if default_handle_unknown in handle_unknown_options
        else 0
    )

    handle_unknown = st.selectbox(
        "Handle unknown categories",
        handle_unknown_options,
        index=safe_index
    )

    unknown_value = -1
    if handle_unknown == "use_encoded_value":
        unknown_value = st.number_input(
            "Value for unknown categories",
            value=edit_values.get("unknown_value", -1) if edit_values else -1,
            key="unknown_value"
        )

    return {
        "columns": selected_columns,
        "mappings": encoding_mappings,
        "handle_unknown": handle_unknown,
        "unknown_value": unknown_value
    }


def apply_ordinal_encoding(df, step):
    """Apply ordinal encoding to selected columns"""
    df_copy = df.copy()

    try:
        for col in step["columns"]:
            if col not in df_copy.columns:
                st.warning(f"Column '{col}' not found for ordinal encoding")
                continue

            # Guard: skip if the column has already been numerically encoded by a prior step
            if df_copy[col].dtype not in ['object', 'category']:
                st.warning(
                    f"Column '{col}' is not categorical (dtype: {df_copy[col].dtype}). "
                    f"Skipping — it may have already been encoded by a prior step."
                )
                continue

            if col not in step["mappings"]:
                st.warning(f"No mapping defined for column '{col}'")
                continue

            categories = step["mappings"][col]

            # Create and fit ordinal encoder
            encoder = OrdinalEncoder(
                categories=[categories],
                handle_unknown=step["handle_unknown"],
                unknown_value=step["unknown_value"]
            )

            # Encode the column
            encoded_values = encoder.fit_transform(df_copy[[col]])
            df_copy[col] = encoded_values.flatten()

            # Store mapping information
            mapping_info = {cat: idx for idx, cat in enumerate(categories)}
            st.info(f"Ordinal encoding applied to '{col}': {mapping_info}")

    except Exception as e:
        st.error(f"Error in ordinal encoding: {str(e)}")
        with st.expander("Details"):
            st.code(traceback.format_exc())
        return df_copy  # never crash the pipeline

    return df_copy
