import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu 
import numpy as np
# Assuming you have a file named 'statistics_execution.py' 
# Make sure to adjust the import path if it's different in your project structure
from stats_lib.statistics_execution import execute_statistics 

# Helper class/vars needed (adjust as per your project's `constants.py` and `DataManager`)

# --- Utility Functions ---

def get_initials(name):
    """Get initials from step name"""
    words = name.split()
    if len(words) >= 2:
        return f"{words[0][0]}{words[-1][0]}".upper()
    return name[:2].upper() if len(name) >= 2 else name[0].upper() * 2

def get_color_from_category(category):
    """Get a consistent color based on statistical category"""
    colors = {
        "descriptive_stats": "#1f77b4", # Blue
        "t_test_two_sample": "#ff7f0e", # Orange
        "confidence_interval": "#2ca02c", # Green
        "correlation_regression": "#d62728", # Red
        "default": "#0c0b0e" # Purple
    }
    return colors.get(category.lower(), colors["default"])

def save_statistics_step(name, category, params, is_edit=False):
    """Save the statistical step to the pipeline"""
    new_step = {"name": name, "type": "statistics", "category": category, **params}

    if is_edit:
        # Find and replace the existing step
        for i, step in enumerate(st.session_state.pipeline['statistics']):
            if step["name"] == st.session_state.editing_statistics["name"]:
                st.session_state.pipeline['statistics'][i] = new_step
                break
    else:
        # Add new step
        st.session_state.pipeline['statistics'].append(new_step)
    
    st.session_state.show_stats_modal = False
    st.session_state.editing_statistics = None
    st.rerun()


# --- Display Functions ---

def display_statistics_dashboard():
    """Display the list of executed statistical steps and their results."""
    
    if st.session_state.get("df_original") is None:
        st.info("Please upload a dataset on the main data page first.")
        return

    st.markdown("---")
    st.subheader("Results Dashboard")

    if not st.session_state.pipeline.get("statistics"):
        st.info("No statistical analyses have been added yet.")
        return

    # Iterate through all saved steps and execute them for display
    for i, step in enumerate(st.session_state.pipeline['statistics']):
        st.markdown(f"## 📊 Step {i+1}: {step['name']}")
        st.markdown(f"*{step['category'].replace('_', ' ').title()}*")
        
        # Execute the statistics step to display its output
        try:
            # We pass the currently displayed/transformed DF for context, though stats typically use original data
            # For simplicity and adhering to your prompt, we use the original/full data frame here
            with st.expander("View Results", expanded=False):
                execute_statistics("statistics", "execution", {"df": st.session_state.df_original, "step": step})
        except Exception as e:
            st.error(f"Error executing step {step['name']}: {e}")
        
        # Action buttons for list
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("✏️ Edit Step", key=f"edit_stat_{i}"):
                st.session_state.editing_statistics = step
                st.session_state.show_stats_modal = True
                st.rerun()
        with col2:
            if st.button("🗑️ Delete Step", key=f"delete_stat_{i}", type='primary'):
                st.session_state.pipeline['statistics'].pop(i)
                st.rerun()
        st.markdown("---")


def display_statistics_modal():
    """Display the modal for adding/editing statistical steps."""
    with st.container():
        st.header("Add Statistical Step" if not st.session_state.editing_statistics else "Edit Statistical Step")
        
        df = st.session_state.df_original
        
        # Set default values for editing
        edit_values = st.session_state.editing_statistics if st.session_state.editing_statistics else {}
        
        # Step name
        default_name = edit_values.get("name", f"Stat Step {len(st.session_state.pipeline.get('statistics', []))+1}")
        stat_name = st.text_input("Analysis Name", value=default_name)
        
        # Statistical Analysis Type Selector
        stat_choices = [
            "descriptive_stats", 
            "t_test_two_sample", 
            "confidence_interval", 
            "correlation_regression"
        ]

        try:
            default_index = stat_choices.index(edit_values.get("category", stat_choices[0]))
        except ValueError:
            default_index = 0

        stat_choice = option_menu(
            menu_title=None,
            options=[c.replace('_', ' ').title() for c in stat_choices],
            icons=["bar-chart", "equal", "activity", "hash"],  # Appropriate icons
            default_index=default_index,
            orientation="horizontal",
            styles={
                "container": {"padding": "0!important", "background-color": "#0A0A0A", "margin-bottom": "1rem"},
                "nav-link": {"font-size": "14px", "text-align": "center", "margin": "0px", "padding": "8px 12px", "border-radius": "4px"},
                "nav-link-selected": {"background-color": "#1f77b4", "color": "white"} # Use the blue color from stats.py
            }
        )
        
        # Map the pretty name back to the internal key
        internal_choice = stat_choices[[c.replace('_', ' ').title() for c in stat_choices].index(stat_choice)]
        
        with st.expander("Analysis Settings", expanded=True):
            # Dynamic configuration UI via the execution router
            try:
                transformation_params = execute_statistics(
                    "statistics",
                    "config",
                    {"choice": internal_choice, "df": df, "edit_values": edit_values}
                )
            except Exception as e:
                transformation_params = {}
                st.error(f"Error loading configuration: {e}")
        
        # Action buttons
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Apply and Run"):
                # Basic validation for name
                if not stat_name:
                    st.error("Please provide a name for the analysis step.")
                else:
                    save_statistics_step(
                        stat_name,
                        internal_choice,
                        transformation_params,
                        is_edit=bool(st.session_state.editing_statistics)
                    )
        with col2:
            if st.button("Cancel"):
                st.session_state.show_stats_modal = False
                st.session_state.editing_statistics = None
                st.rerun()

# --- Main Page Runner ---

def run():
    # 1. Initialize Session State for Statistics
    if 'show_stats_modal' not in st.session_state:
        st.session_state.show_stats_modal = False
    if 'editing_statistics' not in st.session_state:
        st.session_state.editing_statistics = None
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = {}

    # Ensure the 'statistics' key exists in the pipeline dictionary
    st.session_state.pipeline.setdefault("statistics", [])

    
    if st.session_state.get("df_original") is None:
        st.warning("Please navigate to the main data page and upload a dataset to begin statistical analysis.")
        return

    # 2. Add Analysis Button
    if st.button("➕ Add New Statistical Analysis"):
        st.session_state.show_stats_modal = True
        st.session_state.editing_statistics = None
        st.rerun()

    # 3. Display Modal
    if st.session_state.get("show_stats_modal", False):
        display_statistics_modal()

    # 4. Display Results Dashboard
    display_statistics_dashboard()

if __name__ == "__main__":
    # This block is for local testing only, you'd integrate `run()` into your Streamlit multi-page app
    # For a full multi-page app, this entire file would be `pages/stats_page.py`
    st.title("Statistics Page (Demo)")
    
    # Dummy setup for local test
    if "df_original" not in st.session_state:
        st.session_state.df_original = pd.DataFrame({
            'A': np.random.rand(100) * 10,
            'B': np.random.rand(100) * 5 + 5,
            'Group': ['G1', 'G2'] * 50,
            'C': np.random.randint(1, 5, 100)
        })
        st.session_state.pipeline = {}

    run()