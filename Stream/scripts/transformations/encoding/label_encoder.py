import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
# =============================================================================
# Individual Encoding Function Builders
# =============================================================================

def show_column_info(df, selected_columns):
    """Display information about selected columns"""
    if not selected_columns:
        return
        
    st.write("📊 **Current Column Information:**")
    
    for col in selected_columns:
        if col in df.columns:
            unique_vals = df[col].nunique()
            sample_vals = df[col].dropna().unique()[:5]  # Show first 5 unique values
            dtype = df[col].dtype
            
            st.write(f"   - `{col}`: {unique_vals} unique values, dtype: {dtype}")
            if len(sample_vals) > 0:
                st.write(f"     Sample: {list(sample_vals)}")

def build_label_encoding(df, edit_values=None):
    """Build Label Encoding UI"""
    st.write("### Label Encoding")
    
    # Get categorical columns
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if not categorical_columns:
        st.warning("No categorical columns found for label encoding")
        return {"columns": []}
    
    # Column selection
    default_columns = edit_values.get("columns", categorical_columns) if edit_values else categorical_columns
    selected_columns = st.multiselect(
        "Select categorical columns for label encoding",
        categorical_columns,
        default=default_columns
    )
    
    # Initialize result parameters
    result_params = {"columns": []}
    
    # Show column information
    if selected_columns:
        show_column_info(df, selected_columns)
        
        # Encoding options
        st.write("### Encoding Options")
        
        handle_unknown = st.selectbox(
            "Handle unknown values",
            ["error", "use_encoded_value"],
            index=0,
            help="How to handle unknown categories during transformation"
        )
        
        if handle_unknown == "use_encoded_value":
            unknown_value = st.number_input(
                "Value to assign to unknown categories",
                value=-1,
                help="Value to use when encountering unknown categories"
            )
        else:
            unknown_value = -1
        
        # Preview encoding
        if st.checkbox("Preview encoding effect"):
            preview_label_encoding(df, selected_columns)
        
        # Update result parameters
        result_params = {
            "columns": selected_columns,
            "handle_unknown": handle_unknown,
            "unknown_value": unknown_value
        }
    else:
        st.warning("Please select at least one categorical column for label encoding")
    
    return result_params



def preview_label_encoding(df, selected_columns):
    """Preview Label Encoding effect on selected columns"""
    if not selected_columns:
        return
        
    st.write("### Label Encoding Preview")
    
    preview_df = df[selected_columns].head().copy()
    st.write("**Original values (first 5 rows):**")
    st.dataframe(preview_df)
    
    try:
        # Apply label encoding for preview
        encoded_df = preview_df.copy()
        encoders = {}
        
        for col in selected_columns:
            if col in encoded_df.columns:
                le = LabelEncoder()
                encoded_df[col] = le.fit_transform(encoded_df[col].astype(str))
                encoders[col] = le
        
        st.write("**After Label Encoding:**")
        st.dataframe(encoded_df)
        
        # Show encoding mapping
        st.write("**Encoding mapping:**")
        for col, le in encoders.items():
            classes = le.classes_
            st.write(f"   - `{col}`: {list(classes)} → {list(range(len(classes)))}")
            
    except Exception as e:
        st.error(f"Error in preview: {str(e)}")



# =============================================================================
# Core Encoding Functions
# =============================================================================

def apply_label_encoding(df, step):
    """Apply Label Encoding to selected columns"""
    df_copy = df.copy()
    
    selected_columns = step.get("columns", [])
    handle_unknown = step.get("handle_unknown", "error")
    unknown_value = step.get("unknown_value", -1)
    
    if not selected_columns:
        st.warning("No columns selected for label encoding")
        return df_copy
    
    # Filter only categorical columns that exist in dataframe
    valid_columns = [col for col in selected_columns if col in df_copy.columns and df_copy[col].dtype in ['object', 'category']]
    
    if not valid_columns:
        st.warning("No valid categorical columns found for label encoding")
        return df_copy
    
    try:
        # Store encoders for potential inverse transformation
        if 'encoders' not in st.session_state:
            st.session_state.encoders = {}
        
        # Apply Label Encoding
        for col in valid_columns:
            le = LabelEncoder()
            df_copy[col] = le.fit_transform(df_copy[col].astype(str))
            st.session_state.encoders[col] = le
        
        # Show transformation summary
        st.success(f"✅ Applied Label Encoding to {len(valid_columns)} columns: {', '.join(valid_columns)}")
        st.info(f"📊 Converted categorical values to numeric labels")
        
    except Exception as e:
        st.error(f"Error during Label Encoding: {str(e)}")
        return df
    
    return df_copy



def inverse_label_encoding(df, encoder_info):
    """Apply inverse Label Encoding to revert transformation"""
    df_copy = df.copy()
    
    try:
        for col, le in encoder_info.items():
            if col in df_copy.columns:
                # Apply inverse transformation
                df_copy[col] = le.inverse_transform(df_copy[col].astype(int))
        
    except Exception as e:
        st.error(f"Error during inverse Label Encoding: {str(e)}")
    
    return df_copy