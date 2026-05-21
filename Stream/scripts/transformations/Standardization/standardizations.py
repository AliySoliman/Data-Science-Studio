import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from transformations.Standardization.z_score import build_zscore_config, apply_zscore_standardization
from transformations.Standardization.robust import build_robust_config, apply_robust_scaler
from transformations.Standardization.min_max_Standardization import apply_minmax_scaling, build_minmax_standardization
from transformations.Standardization.normalization import build_normalization_transf, apply_normalization
def standardizations_config(choice, edit_values=None):
    """Build the standardization transformation UI based on choice"""
    # Always initialize with default value
    transformation_params = {}
    
    # Check if dataframe exists in session state
    if not hasattr(st.session_state, 'df_original') or st.session_state.df_original is None:
        st.warning("No dataset loaded. Please upload a dataset first.")
        return transformation_params
    
    df = st.session_state.df_to_show if not st.session_state.df_to_show.empty else st.session_state.df_original
    
    # Handle different standardization choices
    if choice == "MinMax Standardization":
        transformation_params = build_minmax_standardization(df, edit_values)
    elif choice == "Z-score Standardization":
        transformation_params = build_zscore_config(edit_values)
    elif choice == "Robust Scaler":
        transformation_params = build_robust_config(edit_values)
    elif choice == "Mean Normalization":
            transformation_params = build_normalization_transf(
                df,
                edit_values
            )
    else:
        st.warning(f"Standardization type '{choice}' is not implemented yet")
        return transformation_params
    
    # Ensure we always return a dictionary with the expected structure
    if transformation_params:
        transformation_params["category"] = choice
    return transformation_params

def apply_standardization_transformation(df, step):
    """Apply the selected standardization transformation to the dataframe"""
    df_copy = df.copy()
    
    try:
        category = step.get("category")
        if category == "MinMax Standardization":
            df_copy = apply_minmax_scaling(df_copy, step)
        elif category == "Z-score Standardization":
            df_copy = apply_zscore_standardization(df_copy, step)
        elif category == "Robust Scaler":
            df_copy = apply_robust_scaler(df_copy, step)
        elif category == "Mean Normalization":
            df_copy = apply_normalization(df_copy, step)
        else:
            st.warning(f"Unknown standardization category: {category}")
            
    except Exception as e:
        st.error(f"Error applying standardization transformation: {str(e)}")
        return df
    
    return df_copy

# =============================================================================
# Individual Standardization Function Builders
# =============================================================================

