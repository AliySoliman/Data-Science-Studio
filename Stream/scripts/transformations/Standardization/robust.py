# Robust Scaler Functions
import streamlit as st
from sklearn.preprocessing import  RobustScaler

def build_robust_config(edit_values=None):
    """Build Robust Scaler configuration UI"""
    st.markdown("""
    **Robust Scaler**
    - Scales data using statistics that are robust to outliers
    - Uses IQR (Interquartile Range) instead of standard deviation
    - Formula: (x - median) / IQR
    - Best for data with outliers
    """)
    
    df = st.session_state.df_to_show if not st.session_state.df_to_show.empty else st.session_state.df_original
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if not numeric_cols:
        st.error("No numeric columns found for scaling")
        return {"columns": []}
    
    # Column selection
    default_cols = edit_values.get("columns", []) if edit_values else []
    selected_cols = st.multiselect(
        "Select numeric columns for Robust scaling:",
        options=numeric_cols,
        default=default_cols,
        help="Select numeric columns to apply Robust scaling"
    )
    
    # Quantile range options
    # with st.expander("Advanced Options"):
    st.write("Advanced Options:")
    col1, col2 = st.columns(2)
    with col1:
        quantile_range_low = st.number_input(
            "Lower quantile",
            min_value=0.0,
            max_value=50.0,
            value=edit_values.get("quantile_range_low", 25.0) if edit_values else 25.0,
            help="Lower quantile for IQR calculation (0-50)"
        )
    with col2:
        quantile_range_high = st.number_input(
            "Upper quantile", 
            min_value=50.0,
            max_value=100.0,
            value=edit_values.get("quantile_range_high", 75.0) if edit_values else 75.0,
            help="Upper quantile for IQR calculation (50-100)"
        )
    
    with_centering = st.checkbox(
        "Center to median",
        value=edit_values.get("with_centering", True) if edit_values else True,
        help="If True, center the data to the median"
    )
    with_scaling = st.checkbox(
        "Scale to IQR",
        value=edit_values.get("with_scaling", True) if edit_values else True,
        help="If True, scale the data to the IQR"
    )

    # Preview
    if selected_cols:
        st.info(f"Selected columns: {', '.join(selected_cols)}")
        
        # Show before stats with IQR
        st.write("**Current Statistics (with IQR):**")
        stats = df[selected_cols].describe().round(4)
        st.dataframe(stats)
    
    return {
        "columns": selected_cols,
        "quantile_range": (quantile_range_low, quantile_range_high),
        "with_centering": with_centering,
        "with_scaling": with_scaling
    }

def apply_robust_scaler(df, step):
    """Apply Robust Scaler to selected columns"""
    columns = step.get("columns", [])
    quantile_range = step.get("quantile_range", (25.0, 75.0))
    with_centering = step.get("with_centering", True)
    with_scaling = step.get("with_scaling", True)
    
    if not columns:
        st.warning("No columns selected for Robust scaling")
        return df
    
    # Create a copy to avoid modifying original
    df_transformed = df.copy()
    
    # Initialize scaler
    scaler = RobustScaler(
        quantile_range=quantile_range,
        with_centering=with_centering,
        with_scaling=with_scaling
    )
    
    # Apply scaling
    try:
        df_transformed[columns] = scaler.fit_transform(df_transformed[columns])
        
        st.success(f"✅ Robust scaling applied to: {', '.join(columns)}")
        
        # Display before/after comparison
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Before Robust Scaling:**")
            st.dataframe(df[columns].describe().round(4))
        with col2:
            st.write("**After Robust Scaling:**")
            st.dataframe(df_transformed[columns].describe().round(4))
            
    except Exception as e:
        st.error(f"Error in Robust scaling: {str(e)}")
        return df
    
    return df_transformed

# Utility function to get available standardization types
def get_standardization_types():
    """Return list of available standardization methods"""
    return ["Z-score Standardization", "Robust Scaler"]