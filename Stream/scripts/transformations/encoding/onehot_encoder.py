import streamlit as st
import pandas as pd
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

def build_onehot_encoding(df, edit_values=None):
    """Build One-Hot Encoding UI"""
    st.write("### One-Hot Encoding")
    
    # Get categorical columns
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if not categorical_columns:
        st.warning("No categorical columns found for one-hot encoding")
        return {"columns": []}
    
    # Column selection
    default_columns = edit_values.get("columns", categorical_columns) if edit_values else categorical_columns
    selected_columns = st.multiselect(
        "Select categorical columns for one-hot encoding",
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
        
        drop_first = st.checkbox(
            "Drop first category",
            value=False,
            help="Drop the first category to avoid multicollinearity"
        )
        
        handle_unknown = st.selectbox(
            "Handle unknown values",
            ["error", "ignore"],
            index=1,
            help="How to handle unknown categories during transformation"
        )
        
        # Preview encoding
        if st.checkbox("Preview encoding effect"):
            preview_onehot_encoding(df, selected_columns, drop_first)
        
        # Update result parameters
        result_params = {
            "columns": selected_columns,
            "drop_first": drop_first,
            "handle_unknown": handle_unknown
        }
    else:
        st.warning("Please select at least one categorical column for one-hot encoding")
    
    return result_params
def preview_onehot_encoding(df, selected_columns, drop_first):
    """Preview One-Hot Encoding effect on selected columns"""
    if not selected_columns:
        return
        
    st.write("### One-Hot Encoding Preview")
    
    preview_df = df[selected_columns].head().copy()
    st.write("**Original values (first 5 rows):**")
    st.dataframe(preview_df)
    
    try:
        # Apply one-hot encoding for preview
        encoded_df = pd.get_dummies(preview_df, columns=selected_columns, drop_first=drop_first)
        
        st.write("**After One-Hot Encoding:**")
        st.dataframe(encoded_df)
        
        # Show created columns
        st.write(f"**Created {len(encoded_df.columns)} new columns:**")
        st.write(f"   {list(encoded_df.columns)}")
            
    except Exception as e:
        st.error(f"Error in preview: {str(e)}")



def apply_onehot_encoding(df, step):
    """Apply One-Hot Encoding to selected columns"""
    df_copy = df.copy()
    
    selected_columns = step.get("columns", [])
    drop_first = step.get("drop_first", False)
    handle_unknown = step.get("handle_unknown", "ignore")
    
    if not selected_columns:
        st.warning("No columns selected for one-hot encoding")
        return df_copy
    
    # Filter only categorical columns that exist in dataframe
    valid_columns = [col for col in selected_columns if col in df_copy.columns and df_copy[col].dtype in ['object', 'category']]
    
    if not valid_columns:
        st.warning("No valid categorical columns found for one-hot encoding")
        return df_copy
    
    try:
        # Apply One-Hot Encoding
        original_columns = len(df_copy.columns)
        df_copy = pd.get_dummies(df_copy, columns=valid_columns, drop_first=drop_first)
        new_columns = len(df_copy.columns) - original_columns + len(valid_columns)
        
        # Show transformation summary
        st.success(f"✅ Applied One-Hot Encoding to {len(valid_columns)} columns: {', '.join(valid_columns)}")
        st.info(f"📊 Created {new_columns} new binary columns")
        
    except Exception as e:
        st.error(f"Error during One-Hot Encoding: {str(e)}")
        return df
    
    return df_copy