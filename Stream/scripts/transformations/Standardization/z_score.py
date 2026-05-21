# Z-Score Standardization Functions
import streamlit as st
from sklearn.preprocessing import StandardScaler

def build_zscore_config(edit_values=None):
    """Build Z-score standardization configuration UI"""
    st.markdown("""
    **Z-score Standardization (Standard Scaler)**
    - Scales data to have mean=0 and standard deviation=1
    - Formula: (x - mean) / std
    - Best for normally distributed data
    - Sensitive to outliers
    """)
    
    df = st.session_state.df_to_show if not st.session_state.df_to_show.empty else st.session_state.df_original
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if not numeric_cols:
        st.error("No numeric columns found for standardization")
        return {"columns": []}
    
    # Column selection
    default_cols = edit_values.get("columns", []) if edit_values else []
    selected_cols = st.multiselect(
        "Select numeric columns to standardize:",
        options=numeric_cols,
        default=default_cols,
        help="Select numeric columns to apply Z-score standardization"
    )
    
    # Advanced options
    # with st.expander("Advanced Options"):
    col1, col2 = st.columns(2)
    with col1:
        with_mean = st.checkbox(
            "Center to mean (with_mean)", 
            value=edit_values.get("with_mean", True) if edit_values else True,
            help="If True, center the data before scaling"
        )
    with col2:
        with_std = st.checkbox(
            "Scale to unit variance (with_std)", 
            value=edit_values.get("with_std", True) if edit_values else True,
            help="If True, scale the data to unit variance"
        )
    
    # Preview
    if selected_cols:
        st.info(f"Selected columns: {', '.join(selected_cols)}")
        
        # Show before stats
        st.write("**Current Statistics:**")
        stats_before = df[selected_cols].describe().round(4)
        st.dataframe(stats_before)
    
    return {
        "columns": selected_cols,
        "with_mean": with_mean,
        "with_std": with_std
    }

def apply_zscore_standardization(df, step):
    """Apply Z-score standardization to selected columns"""
    columns = step.get("columns", [])
    with_mean = step.get("with_mean", True)
    with_std = step.get("with_std", True)
    
    if not columns:
        st.warning("No columns selected for Z-score standardization")
        return df
    
    # Create a copy to avoid modifying original
    df_transformed = df.copy()
    
    # Initialize scaler
    scaler = StandardScaler(with_mean=with_mean, with_std=with_std)
    
    # Apply standardization
    try:
        df_transformed[columns] = scaler.fit_transform(df_transformed[columns])
        
        # Show transformation results
        st.success(f"✅ Z-score standardization applied to: {', '.join(columns)}")
        
        # Display before/after comparison
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Before Standardization:**")
            st.dataframe(df[columns].describe().round(4))
        with col2:
            st.write("**After Standardization:**")
            st.dataframe(df_transformed[columns].describe().round(4))
            
    except Exception as e:
        st.error(f"Error in Z-score standardization: {str(e)}")
        return df
    
    return df_transformed
