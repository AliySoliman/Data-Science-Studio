import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from constants import DataManager
import transform
from viz_lib import visualization_ui, visualization_save, visualization_execution

data_manager = DataManager()

# def run():
#     # Initialize session state
#     if 'visualizations' not in st.session_state:
#         st.session_state.visualizations = []
#     if 'dashboard_layout' not in st.session_state:
#         st.session_state.dashboard_layout = "Single Column"
#     if 'current_viz_id' not in st.session_state:
#         st.session_state.current_viz_id = 0
    
#     st.markdown(data_manager.label_style, unsafe_allow_html=True)
#     st.markdown("""
#     <div class="custom-container">
#         <h2 class="custom-header">📊 Interactive Visualizations Dashboard</h2>
#     </div>
#     """, unsafe_allow_html=True)
    
#     # Show save file status
#     if 'save_file_upload' in st.session_state and st.session_state.save_file_upload:
#         st.info(f"📁 Project save file: {st.session_state.save_file_upload}")
#     else:
#         st.warning("⚠️ No save file configured. Go to Transform page to set a save file path.")
    
#     # Check if data is loaded
#     if st.session_state.load or st.session_state.df_original is None:
#         st.warning("Please upload data first in the Transform page")
#         return
    
#     # Main visualization interface
#     col1, col2 = st.columns([3, 1])
    
#     with col1:
#         visualization_ui.show_visualization_creator(st.session_state.df_original)
        
#         # Show edit modal when needed
#         visualization_ui.show_edit_modal(st.session_state.df_original)
    
#     with col2:
#         visualization_ui.show_dashboard_controls()
    
#     # Display dashboard
#     st.markdown("---")
#     st.subheader("📈 Visualization Dashboard")
#     visualization_ui.show_dashboard(st.session_state.df_original)
    
#     # Export options
#     st.markdown("---")
#     visualization_save.show_export_options()
    
#     # Add load button
#     if st.button("📂 Load from Project", type="secondary"):
#         visualization_save.load_dashboard_from_project()
def run():
    # Initialize session state
    if 'visualizations' not in st.session_state:
        st.session_state.visualizations = []
    if 'dashboard_layout' not in st.session_state:
        st.session_state.dashboard_layout = "Single Column"
    if 'current_viz_id' not in st.session_state:
        st.session_state.current_viz_id = 0
    # ADD: Initialize global transformations
    if 'global_transformations' not in st.session_state:
        st.session_state.global_transformations = ["original"]
    
    st.markdown(data_manager.label_style, unsafe_allow_html=True)
    st.markdown("""
    <div class="custom-container">
        <h2 class="custom-header">📊 Interactive Visualizations Dashboard</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Show save file status
    if 'save_file_upload' in st.session_state and st.session_state.save_file_upload:
        st.info(f"📁 Project save file: {st.session_state.save_file_upload}")
    else:
        st.warning("⚠️ No save file configured. Go to Transform page to set a save file path.")
    
    # Check if data is loaded
    if st.session_state.load or st.session_state.df_original is None:
        st.warning("Please upload data first in the Transform page")
        return
    
    # ADD: Global Transformations Section
    st.subheader("🌐 Global Data Transformations")
    st.info("These transformations will be applied to ALL visualizations below")
    
    available_transformations = ["original"] + [step["name"] for step in st.session_state.pipeline['transformations']]
    global_transformations = st.multiselect(
        "Select global transformations to apply to all charts:",
        options=available_transformations,
        default=st.session_state.global_transformations,
        key="global_transformations_selector"
    )
    
    # Update global transformations
    st.session_state.global_transformations = global_transformations
    
    # Show preview of global transformed data
    global_data = apply_global_transformations()
    with st.expander("📋 Global Data Preview (after transformations)"):
        st.dataframe(global_data.head(10))
        st.write(f"Shape: {global_data.shape}")
        st.write(f"Columns: {', '.join(global_data.columns.tolist())}")
    
    # Rest of the function remains the same...
    # Main visualization interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        visualization_ui.show_visualization_creator(st.session_state.df_original)
        
        # Show edit modal when needed
        visualization_ui.show_edit_modal(st.session_state.df_original)
    
    with col2:
        visualization_ui.show_dashboard_controls()
    
    # Display dashboard
    st.markdown("---")
    st.subheader("📈 Visualization Dashboard")
    visualization_ui.show_dashboard(st.session_state.df_original)
    
    # Export options
    st.markdown("---")
    visualization_save.show_export_options()
    
    # Add load button
    if st.button("📂 Load from Project", type="secondary"):
        visualization_save.load_dashboard_from_project()

# ADD: Helper function for global transformations
def apply_global_transformations():
    """Apply global transformations to the base data"""
    if not st.session_state.global_transformations or ("original" in st.session_state.global_transformations and len(st.session_state.global_transformations) == 1):
        return st.session_state.df_original.copy()
    
    df = st.session_state.df_original.copy()
    
    for step_name in st.session_state.global_transformations:
        if step_name == "original":
            continue
            
        # Find the transformation step
        step = next((s for s in st.session_state.pipeline['transformations'] if s["name"] == step_name), None)
        if step:
            try:
                from transformations import transformation_execution
                df = transformation_execution.execute_transformation(
                    step["type"], 
                    "execution", 
                    {"df": df, "step": step}
                )
            except Exception as e:
                st.error(f"Error applying global transformation '{step_name}': {str(e)}")
                continue
    
    return df