import streamlit as st
import pandas as pd
import re
from typing import Any
from streamlit_elements import elements, mui, html
import save
from constants import DataManager
from collections import defaultdict
from streamlit_option_menu import option_menu 
from transformations import transformation_execution
data_manager = DataManager()

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

def run():
    # Initialize session state
    if 'load' not in st.session_state :
        st.session_state.load =True
    if 'save_load' not in st.session_state :
        st.session_state.save_load =False
    if 'edit' not in st.session_state:
        st.session_state.edit = False
    if 'editing_transformation' not in st.session_state:
        st.session_state.editing_transformation = None
    if "dataset_loaded" not in st.session_state:
        st.session_state.dataset_loaded = False
    if "current_dataset" not in st.session_state:
        st.session_state.current_dataset = None
    if "dataset_name" not in st.session_state:
        st.session_state.dataset_name = ""
    if "df_original" not in st.session_state:
        st.session_state.df_original = None
    if "pipeline" not in st.session_state:
            st.session_state.pipeline = {}

    if "transformations" not in st.session_state.pipeline:
        st.session_state.pipeline["transformations"] = []   
        st.session_state.pipeline.setdefault("ML", [])
    if "cached_items" not in st.session_state:
        st.session_state.cached_items = {}
   
    if "transformed_dfs" not in st.session_state:
        st.session_state.transformed_dfs = {"original": None}
    if "df_to_show" not in st.session_state:
        st.session_state.df_to_show = pd.DataFrame()

    # UI Layout
    st.markdown(data_manager.label_style, unsafe_allow_html=True)
    st.markdown(data_manager.label_data, unsafe_allow_html=True)

    # st.title("Dynamic Table with Transformation Pipeline")
    # Upload dataset
    data_col , save_col,submit_col = st.columns([4,1,1])
    with data_col:
        # File upload
        uploaded_file = st.file_uploader("📂 Upload your dataset", type=["csv", "xlsx", "xls"])
        
        # Use existing dataset or load new one
        current_df = None
        if uploaded_file is not None:
            # New file uploaded
            current_df = load_dataset(uploaded_file)
        elif st.session_state.dataset_loaded and st.session_state.current_dataset is not None:
            # Use existing dataset from session state
            current_df = st.session_state.current_dataset
            st.info(f"📁 Using previously loaded dataset: **{st.session_state.dataset_name}**")

    with save_col :
            save_file = st.text_input("save file path No quotations",key = 'save_upload')
            if st.button('upload'):
                # CLEAN PATH FIX
                clean_path = save_file.strip().replace('"', '').replace("'", "")
                try:
                    loaded_content = load_data(clean_path)
                    if loaded_content:
                        st.session_state.pipeline = loaded_content
                        st.session_state.save_file_upload = clean_path
                        st.success("Pipeline reloaded successfully!")
                        st.rerun() # UI REFRESH FIX
                    else:
                        st.error("Could not load data from path.")
                except Exception as e:
                    st.error(f"Failed to load: {e}")

    # RESTORED BLACK BOXES (st.write)
    st.write(st.session_state.pipeline)
    st.write(st.session_state.cached_items)
    # st.write(st.session_state.df_to_show.info())

    # Setup the original dataframe for the pipeline
    if current_df is None or current_df.empty:
        # UI DECOUPLING FIX: No return here, just info.
        st.info("👆 Upload a dataset to begin.")
    else:
        st.session_state.df_original = current_df
        data_manager.current_df = current_df.copy()
        st.session_state.transformed_dfs["original"] = current_df.copy()
        st.session_state.load = False


    # list_changed = current_pipeline != st.session_state.pipeline
    # Display transformations selector and table
    # try:
    display_transformations_and_table()
    # except  : st.error("Upload Data")
        # Transformation modal
    if st.button("Add Transformation"):
            st.session_state.show_modal = True
            st.session_state.editing_transformation = None
            st.rerun()

    if st.session_state.get("show_modal", False):
        display_transformation_modal()

def display_transformations_and_table():
    """Display the transformations selector and resulting table"""
    
    # Render table only if data exists
    if st.session_state.df_original is not None and st.session_state.pipeline.get('transformations'):
        # Select which transformations to apply
        selected_steps = st.multiselect(
            "Choose pipeline steps to view result", 
            ["original"] + [step["name"] for step in st.session_state.pipeline['transformations']]
        )
        st.session_state.df_to_show = data_manager.current_df.copy()
        if selected_steps:

                for step in st.session_state.pipeline['transformations']:
                    if step['name'] in selected_steps:
                # Apply selected transformations     
                        if step['name']  in st.session_state.cached_items and step['type'] =='dimensionality reduction':
                            st.session_state.df_to_show = st.session_state.cached_items[step['name']].copy()
                        else :
                            st.session_state.df_to_show = transformation_execution.execute_transformation(step['type'],"execution",{"df":st.session_state.df_to_show,"step":step})
                            if step['type'] =='dimensionality reduction':
                                st.session_state.cached_items[step['name']] = st.session_state.df_to_show.copy()

        # Display table 
        st.dataframe(st.session_state.df_to_show)
    elif st.session_state.pipeline.get('transformations'):
         st.warning("⚠️ Pipeline loaded successfully. Upload a dataset to calculate and see results.")

    # SIDEBAR: Matches User's Script Exactly
    with st.sidebar:
            st.markdown(
                """
                <style>
<style>
    /* Define color variables */
    :root {
        --oxford-blue: #112343;
        --oxford-blue-2: #0f1533;
        --indigo-dye: #123b5f;
        --violet-blue: #3941aa;
        --vista-blue: #7b99c4;
    }

    /* Example: change sidebar background */
    [data-testid="stSidebar"] {
        background: var(--gradient-right);
        color: white;
    }

    /* Example: add gradient background */
    body {
        background: linear-gradient(135deg, var(--oxford-blue), var(--indigo-dye), var(--vista-blue));
    }


    /* Define gradients */
    :root {
        --gradient-right: linear-gradient(90deg, #0a0f20, #0d1230, #0e2440, #2c3275, #5a7290);
    }
                </style>
                """,
                unsafe_allow_html=True
            )
            st.header("🛠️ Transformations")
            st.markdown('<div class="custom-container">', unsafe_allow_html=True)
            display_transformations_list()



def display_transformations_list():
    st.markdown(data_manager.app_style, unsafe_allow_html=True)

    if st.session_state.pipeline:
        # Create a container with scrollable CSS

            for i, step in enumerate(st.session_state.pipeline.get('transformations', [])):
                col1, col2 = st.columns([5, 1])
                with col1:
                    initials = get_initials(step["name"])
                    color = get_color_from_type(step["type"])
                    
                    st.markdown(f"""
                    <div class="transformation-card" title="{step['name']} ({step['type']})">
                        <div class="initials-badge" style="background-color: {color}">
                            {initials}
                        </div>
                        <div class="card-content">
                            <div class="card-title">{step['name']}</div>
                            <div class="card-type">{step['type']}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col2:
                    if st.button("✏️", key=f"edit_{i}"):
                        st.session_state.editing_transformation = step
                        st.session_state.show_modal = True
                        st.rerun()
                        
                    if st.button("🗑️", key=f"delete_{i}",type= 'primary'):
                        st.session_state.pipeline['transformations'].pop(i)
                        if step["name"] in st.session_state.cached_items:
                            del st.session_state.cached_items[step["name"]]
                        st.rerun()

            # Close scrollable div



def get_initials(name):
    """Get initials from transformation name"""
    words = name.split()
    if len(words) >= 2:
        return f"{words[0][0]}{words[-1][0]}".upper()
    return name[:2].upper() if len(name) >= 2 else name[0].upper() * 2

def get_color_from_type(transformation_type):
    """Get a consistent color based on transformation type"""
    colors = {
        "delete": "#e15759",
        "computation": "#4e79a7", 
        "binning": "#59a14f",
        "modify": "#edc948",
        "categorize": "#af7aa1",
        "default": "#76b7b2"
    }
    return colors.get(transformation_type.lower(), colors["default"])


def display_transformation_modal():
    """Display the modal for adding/editing transformations"""
    with st.container():
        st.header("Add Transformation" if not st.session_state.editing_transformation else "Edit Transformation")
        
        # Set default values for editing
        edit_values = st.session_state.editing_transformation if st.session_state.editing_transformation else {}
        
        # Transformation name
        transform_name = st.text_input("Transformation Name", value=edit_values.get("name", ""))
        categories = ["cleaning", "transformation", "standardization","dimensionality reduction","encoding","feature selection"]

        # Transformation type selector
        transform_type = option_menu(
        menu_title=None,
        options=categories,
        icons=["speedometer2", "gear", "speedometer2"],  # Bootstrap icons
        default_index=categories.index(edit_values.get("type", "cleaning")),
        orientation="horizontal",
        styles={
            "container": {  
                "padding": "0!important", 
                "background-color": "#7B99C4",
                "margin-bottom": "1rem"
            },
            "nav-link": {
                "font-size": "14px",
                "text-align": "center",
                "margin": "0px",
                "padding": "8px 12px",
                "border-radius": "4px"
            },
            "nav-link-selected": {
                "background-color": "#0F1533",
                "color": "white"
            }
        }
    )   
        transformations=get_transform_type(transform_type) 
        try:
            subtype_index =transformations.index(edit_values.get("category", ""))
        except Exception as e : subtype_index =0
        transform_subtype = st.selectbox(
            options=get_transform_type(transform_type),
            label="Select Type",
            index=subtype_index
        )

        with st.expander("Transformation Settings", expanded=True):
        # Display appropriate fields based on transformation type
            # try:
            transformation_params = transformation_execution.execute_transformation(transform_type,"config",{"choice":transform_subtype,"edit_values":edit_values})

        
        # Action buttons
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Apply"):
                save_transformation(
                    transform_name,
                    transform_type,
                    transform_subtype,
                    transformation_params,
                    is_edit=bool(st.session_state.editing_transformation))
        with col2:
            if st.button("Cancel"):
                st.session_state.show_modal = False
                st.rerun()


def get_transform_type(action):
    if action == "cleaning":
        trans_list = ['Duplicate Handling', 'Null handling', 'Outlier Handling', 'Rename Columns', 'Binary Encoding', 'Date Validation']
    elif action == "transformation":
        trans_list = ["delete", "computation", "filter", "group"]
    elif action == "standardization":
        trans_list = ['scaling', 'Mean Normalization','MinMax Standardization','Robust Scaler','Z-score Standardization']
    elif action == "dimensionality reduction":
        trans_list = ['PCA', 't-SNE Dimensionality Reduction', 'UMAP Dimensionality Reduction', 'LDA']
    elif action == "encoding": 
        trans_list = ['Ordinal Encoding', 'Label Encoding', 'one-hot encoding', 'Target Encoding','Binary Encoding']
    elif action == "feature selection":
        trans_list = ['Chi-Squared Feature Selection', 'REF Feature Selection','Correlation Feature Selection','Variance Feature Selection','ANOVA']
    return trans_list




def save_transformation(name, type, category, params, is_edit=False):
    """Save the transformation to the pipeline"""
    if not name:
        st.error("Transformation name is required")
        return
    new_step = {"name": name, "type": type,"category": category, **params}

    if is_edit:
        # Find and replace the existing transformation
        for i, step in enumerate(st.session_state.pipeline['transformations']):
            if step["name"] == st.session_state.editing_transformation["name"]:
                st.session_state.pipeline['transformations'][i] = new_step
                if step["name"] in st.session_state.cached_items:
                    del st.session_state.cached_items[step["name"]]
                break

        
    else:
        # Add new transformation
        st.session_state.pipeline['transformations'].append(new_step)
    
    st.session_state.show_modal = False
    st.session_state.editing_transformation = None
    st.rerun()

def load_data(f):
    loaded_data = save.load_encoded_transformations(f)
    return loaded_data
