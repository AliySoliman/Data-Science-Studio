import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def show_column_stats(df, selected_columns):
    """Display statistical information for selected columns"""
    if not selected_columns:
        return
        
    st.write("📊 **Current Data Statistics:**")
    
    for col in selected_columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            min_val = df[col].min()
            max_val = df[col].max()
            mean_val = df[col].mean()
            std_val = df[col].std()
            
            st.write(f"   - `{col}`: Min={min_val:.2f}, Max={max_val:.2f}, Mean={mean_val:.2f}, Std={std_val:.2f}")

def build_minmax_standardization(df, edit_values=None):
    """Build MinMax standardization UI"""
    st.write("### MinMax Standardization")
    
    # Initialize with safe default values
    result_params = {
        "columns": [],
        "feature_range": (0, 1)
    }
    
    # Get numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_columns:
        st.warning("No numeric columns found for standardization")
        return result_params
    
    # Column selection
    default_columns = edit_values.get("columns", numeric_columns) if edit_values else numeric_columns
    selected_columns = st.multiselect(
        "Select numeric columns for MinMax scaling",
        numeric_columns,
        default=default_columns
    )
    
    # Show current statistics
    if selected_columns:
        show_column_stats(df, selected_columns)
        
        # Feature range selection
        st.write("### Scaling Range")
        col1, col2 = st.columns(2)
        
        with col1:
            default_min = edit_values.get("feature_range", (0, 1))[0] if edit_values else 0
            min_range = st.number_input(
                "Minimum value after scaling",
                value=default_min,
                key="minmax_min"
            )
        
        with col2:
            default_max = edit_values.get("feature_range", (0, 1))[1] if edit_values else 1
            max_range = st.number_input(
                "Maximum value after scaling",
                value=default_max,
                key="minmax_max"
            )
        
        # Validate range
        if min_range >= max_range:
            st.error("Maximum value must be greater than minimum value")
            return result_params
        
        # Preview transformation
        if st.checkbox("Preview transformation effect"):
            preview_minmax_scaling(df, selected_columns, (min_range, max_range))
        
        # Update result parameters
        result_params = {
            "columns": selected_columns,
            "feature_range": (min_range, max_range)
        }
    else:
        st.warning("Please select at least one numeric column for standardization")
    
    return result_params

def preview_minmax_scaling(df, selected_columns, feature_range):
    """Preview MinMax scaling effect on selected columns"""
    if not selected_columns:
        return
        
    st.write("### Transformation Preview")
    
    preview_df = df[selected_columns].head().copy()
    st.write("**Original values (first 5 rows):**")
    st.dataframe(preview_df)
    
    try:
        # Apply MinMax scaling for preview
        scaler = MinMaxScaler(feature_range=feature_range)
        scaled_values = scaler.fit_transform(preview_df)
        scaled_df = pd.DataFrame(scaled_values, columns=selected_columns, index=preview_df.index)
        
        st.write(f"**After MinMax scaling (range {feature_range}):**")
        st.dataframe(scaled_df)
        
        # Show scaling parameters
        st.write("**Scaling parameters:**")
        for i, col in enumerate(selected_columns):
            data_min = df[col].min()
            data_max = df[col].max()
            st.write(f"   - `{col}`: Will transform from [{data_min:.2f}, {data_max:.2f}] to {feature_range}")
    except Exception as e:
        st.error(f"Error in preview: {str(e)}")

# =============================================================================
# Core Standardization Functions
# =============================================================================

def apply_minmax_scaling(df, step):
    """Apply MinMax scaling to selected columns"""
    df_copy = df.copy()
    
    selected_columns = step.get("columns", [])
    feature_range = step.get("feature_range", (0, 1))
    
    if not selected_columns:
        st.warning("No columns selected for MinMax scaling")
        return df_copy
    
    # Filter only numeric columns that exist in dataframe
    valid_columns = [col for col in selected_columns if col in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy[col])]
    
    if not valid_columns:
        st.warning("No valid numeric columns found for MinMax scaling")
        return df_copy
    
    try:
        # Apply MinMax scaling
        scaler = MinMaxScaler(feature_range=feature_range)
        df_copy[valid_columns] = scaler.fit_transform(df_copy[valid_columns])
        
        # Store scaler information for potential inverse transformation
        if 'scalers' not in st.session_state:
            st.session_state.scalers = {}
        
        st.session_state.scalers['minmax'] = {
            'scaler': scaler,
            'columns': valid_columns,
            'feature_range': feature_range
        }
        
        # Show transformation summary
        st.success(f"✅ Applied MinMax scaling to {len(valid_columns)} columns: {', '.join(valid_columns)}")
        st.info(f"📊 Values scaled to range {feature_range}")
        
    except Exception as e:
        st.error(f"Error during MinMax scaling: {str(e)}")
        return df
    
    return df_copy

def inverse_minmax_scaling(df, scaler_info):
    """Apply inverse MinMax scaling to revert transformation"""
    df_copy = df.copy()
    
    try:
        scaler = scaler_info.get('scaler')
        columns = scaler_info.get('columns', [])
        
        if scaler and columns:
            # Apply inverse transformation
            df_copy[columns] = scaler.inverse_transform(df_copy[columns])
        
    except Exception as e:
        st.error(f"Error during inverse MinMax scaling: {str(e)}")
    
    return df_copy