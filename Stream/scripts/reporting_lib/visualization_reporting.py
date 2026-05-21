import streamlit as st
import uuid
from typing import List, Dict, Any

# Define available report assets for visualizations
VISUALIZATION_REPORT_ASSETS = {
    "create_visualization_plot": "Visualization Plot",
    "create_visualization_comments": "Visualization Comments"
}

def save_visualization_report_item_to_pipeline(viz_id: str, viz_config: Dict, viz_comments: str, final_choices: List[Dict[str, str]]):
    """Saves the visualization report item to the pipeline."""
    # Find the visualization index
    viz_index = None
    for i, viz in enumerate(st.session_state.visualizations):
        if viz['id'] == viz_id:
            viz_index = i
            break
    
    if viz_index is None:
        st.error("Visualization not found.")
        return

    # Create the final Report Item Token
    new_report_item = {
        "id": str(uuid.uuid4()),  # Unique ID for caching
        "type": "Visualization",
        "pipeline_source_key": "visualizations",  # Key in st.session_state
        "pipeline_source_index": viz_index,  # Index of the visualization in the list
        "pipeline_step_name": viz_config.get('title', f"Visualization {viz_index + 1}"),
        "viz_id": viz_id,  # Store the visualization ID for reference
        "viz_config": viz_config,  # Store the visualization configuration
        "viz_comments": viz_comments,  # Store the comments
        "choices": final_choices
    }
    
    # Add to the global report items list
    st.session_state.pipeline.setdefault("report_items", [])
    st.session_state.pipeline["report_items"].append(new_report_item)
    
    # Clear temporary dialog state
    if 'selected_viz_choices_temp' in st.session_state:
        del st.session_state.selected_viz_choices_temp
    if 'final_ordered_viz_choices' in st.session_state:
        del st.session_state.final_ordered_viz_choices
    
    # Close the expander by resetting the flag
    st.session_state.show_viz_dialogue = False
    st.success(f"Visualization '{viz_config.get('title', 'Untitled')}' added to the Report Page!")
    st.rerun()

def display_visualization_reporting_ui(viz_id: str, viz_config: Dict, viz_comments: str):
    """
    Displays the Report Configuration UI for a visualization.
    """
    viz_title = viz_config.get('title', f"Visualization {viz_id}")
    
    with st.container():
        # Check the flag to determine if the expander should be expanded
        is_expanded = st.session_state.get('show_viz_dialogue', False)
        
        with st.expander(f"⚙️ Add Visualization to Report: {viz_title}", expanded=is_expanded):
            
            # --- Initialize Temporary State ---
            if 'selected_viz_choices_temp' not in st.session_state:
                # Initialize with all assets selected by default
                st.session_state.selected_viz_choices_temp = list(VISUALIZATION_REPORT_ASSETS.keys())
            
            # --- 1. Available Components (Selection) ---
            st.markdown("#### 1. Select Components to Include")
            
            available_keys = list(VISUALIZATION_REPORT_ASSETS.keys())
            
            selected_keys = []
            
            for key in available_keys:
                default_state = key in st.session_state.selected_viz_choices_temp
                checkbox_key = f"viz_report_choice_{key}_{viz_id}"
                
                if st.checkbox(VISUALIZATION_REPORT_ASSETS[key], value=default_state, key=checkbox_key):
                    selected_keys.append(key)

            st.session_state.selected_viz_choices_temp = selected_keys
            
            # --- 2. Ordering and Customization ---
            st.markdown("#### 2. Order and Rename Components")
            
            # Create a mutable list of choices with default titles
            temp_ordered_choices: List[Dict[str, str]] = []
            for key in selected_keys:
                temp_ordered_choices.append({
                    "function_name": key,
                    "user_title": st.text_input(
                        f"Title for '{VISUALIZATION_REPORT_ASSETS[key]}'",
                        value=VISUALIZATION_REPORT_ASSETS[key],
                        key=f"viz_title_{key}_{viz_id}"
                    )
                })

            # Initialize/Update final_ordered_choices state
            if 'final_ordered_viz_choices' not in st.session_state or st.session_state.get('last_selected_viz_keys') != selected_keys:
                # Reset if selection has changed or on first run
                st.session_state.final_ordered_viz_choices = temp_ordered_choices
                st.session_state.last_selected_viz_keys = selected_keys
            else:
                # Update titles in the state list based on text inputs
                for i in range(len(temp_ordered_choices)):
                    if i < len(st.session_state.final_ordered_viz_choices):
                        st.session_state.final_ordered_viz_choices[i]['user_title'] = temp_ordered_choices[i]['user_title']

            
            # Simple list display and reordering buttons
            st.markdown("**Current Order:**")
            
            reordered = False
            
            # Use the state list for display and reordering logic
            for i, choice in enumerate(st.session_state.final_ordered_viz_choices):
                col1, col2, col3 = st.columns([8, 1, 1])
                
                with col1:
                    st.markdown(f"**{i + 1}.** {choice['user_title']} *({VISUALIZATION_REPORT_ASSETS[choice['function_name']]})*")
                
                with col2:
                    # Up button
                    if i > 0 and st.button("⬆️", key=f"viz_up_{viz_id}_{i}"):
                        st.session_state.final_ordered_viz_choices[i], st.session_state.final_ordered_viz_choices[i-1] = st.session_state.final_ordered_viz_choices[i-1], st.session_state.final_ordered_viz_choices[i]
                        reordered = True
                        
                with col3:
                    # Down button
                    if i < len(st.session_state.final_ordered_viz_choices) - 1 and st.button("⬇️", key=f"viz_down_{viz_id}_{i}"):
                        st.session_state.final_ordered_viz_choices[i], st.session_state.final_ordered_viz_choices[i+1] = st.session_state.final_ordered_viz_choices[i+1], st.session_state.final_ordered_viz_choices[i]
                        reordered = True
            
            if reordered:
                st.rerun()

            # --- 3. Action Buttons ---
            st.markdown("---")
            col_save, col_cancel = st.columns(2)
            
            with col_save:
                if st.button(f"Add '{viz_title}' to Report", type="primary", use_container_width=True):
                    final_ordered_choices = st.session_state.final_ordered_viz_choices
                    
                    if not final_ordered_choices:
                        st.error("Please select at least one component to include in the report.")
                        return

                    save_visualization_report_item_to_pipeline(viz_id, viz_config, viz_comments, final_ordered_choices)
            
            with col_cancel:
                if st.button("Cancel", use_container_width=True):
                    # Close the expander
                    st.session_state.show_viz_dialogue = False
                    st.rerun()