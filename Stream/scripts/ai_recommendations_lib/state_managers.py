import streamlit as st
import pandas as pd
from typing import Any, Optional

# ___________________________________________________________________________________________________________________________________________________________

def load_dataset(uploaded_file: Any) -> pd.DataFrame:
    """
    Load a dataset from an uploaded file and store it in session state.
    
    This function reads either a CSV or Excel file provided by the user. It also 
    handles persisting the dataset properties in Streamlit's session state and 
    clears any stale recommendations if a new dataset is uploaded.

    Args:
        uploaded_file (Any): The Streamlit file uploader object.

    Returns:
        pd.DataFrame: The loaded dataset as a Pandas DataFrame.
    """
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)
        else:
            raise ValueError("Unsupported file type.")
        
        # Store in session state for cross-tab persistence
        st.session_state.current_dataset = df
        st.session_state.dataset_name = uploaded_file.name
        st.session_state.dataset_loaded = True
        
        # Clear recommendations if dataset changes
        if 'current_dataset_hash' not in st.session_state or st.session_state.current_dataset_hash != hash(str(df.shape) + uploaded_file.name):
            st.session_state.recommendations = None
            st.session_state.current_dataset_hash = hash(str(df.shape) + uploaded_file.name)
            
        return df
    except Exception as e:
        st.error(f"❌ Failed to load: {str(e)}")
        return pd.DataFrame()

# ___________________________________________________________________________________________________________________________________________________________

def get_current_dataset() -> Optional[pd.DataFrame]:
    """
    Get the currently active dataset from the session state.
    
    This provides a safe and easy retriever for the active dataset, returning None
    if no dataset has been successfully loaded yet.

    Returns:
        Optional[pd.DataFrame]: The active Pandas DataFrame, or None if empty.
    """
    return st.session_state.get('current_dataset', None)
