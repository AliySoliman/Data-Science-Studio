from transformations.cleaning.cleanings import cleanings_config, apply_cleaning_transformation
import streamlit as st

def execute_cleaning(type, action, param_dict):
    """Execute cleaning transformations"""
    
    cleanings = {
        "cleaning": {
            "config": cleanings_config,
            "execution": apply_cleaning_transformation,
            "config_params": ["choice", "edit_values"],
            "execution_params": ["df", "step"]
        }
    }
    
    cleaning_info = cleanings.get(type)
    if not cleaning_info:
        st.error(f"Cleaning type '{type}' not found")
        return None
    
    # Pick function and expected params
    if action not in cleaning_info:
        raise ValueError(f"Action '{action}' not found for cleaning type '{type}'")
    
    func = cleaning_info[action]
    
    if action == "config":
        expected_params = cleaning_info["config_params"]
    elif action == "execution":
        expected_params = cleaning_info["execution_params"]
    else:
        expected_params = {}
    
    # Extract only the needed params
    args = {k: param_dict[k] for k in expected_params if k in param_dict}
    
    # Call the function dynamically
    return func(**args)