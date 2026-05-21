from transformations.transformation.transformations import transformations_config, apply_selected_transformations
from transformations.cleaning.cleaning_execution import execute_cleaning
from stats_lib.statistics_execution import execute_statistics
from transformations.encoding.encoding import encoding_config, apply_selected_encodings
from transformations.dimensionality_reduction.dimensional_reduction import get_dimension_config, get_dimension_execution
from transformations.feature_selection.feature_selection import get_feature_config, get_feature_execution
from transformations.Standardization.standardizations import standardizations_config, apply_standardization_transformation
import streamlit as st


def execute_transformation(type,action,param_dict):



    transformations = {
        "transformation": {
            "config": transformations_config,
            "execution": apply_selected_transformations,
            "config_params":["choice","edit_values"],
            "execution_params":["df","step"]
        },

            "cleaning": {
            "config": lambda choice, edit_values: execute_cleaning("cleaning", "config", {"choice": choice, "edit_values": edit_values}),
            "execution": lambda df, step: execute_cleaning("cleaning", "execution", {"df": df, "step": step}),
            "config_params": ["choice", "edit_values"],
            "execution_params": ["df", "step"]
        },
        "statistics": {
            # Since the statistics functions require a 'df' even for config, we use a lambda to route
            "config": lambda choice, edit_values: execute_statistics("statistics", "config", {"choice": choice, "df": param_dict.get('df'), "edit_values": edit_values}),
            "execution": lambda df, step: execute_statistics("statistics", "execution", {"df": df, "step": step}),
            # The execution params will be used to dynamically call the lambda in transform.py
            "config_params": ["choice", "edit_values","df"], # Note: 'df' must be passed when calling from transform.py
            "execution_params": ["df", "step"]
        },

        "encoding": {
            "config":  encoding_config,
            "execution": apply_selected_encodings,
            "config_params": ["choice", "edit_values"],
            "execution_params": ["df", "step"]
        },
        
        "feature selection": {
            "config":  get_feature_config,
            "execution": get_feature_execution,
            "config_params": ["choice", "edit_values"],
            "execution_params": ["df", "step"]
        },
        "standardization": {
            "config": standardizations_config,
            "execution":    apply_standardization_transformation,
            "config_params": ["choice", "edit_values"],
            "execution_params": ["df", "step"]
        },
        "dimensionality reduction": {
            "config": get_dimension_config,
            "execution": get_dimension_execution,
            "config_params": ["choice", "edit_values"],
            "execution_params": ["df", "step"]
        }
    # *********************************************************************************************************************
        }
    trans_info = transformations[type]
    # Pick function and expected params
    if action not in trans_info:
         raise ValueError(f"Action '{action}' not found for model '{type}'")
    func = trans_info[action]
    if action=="config":
        expected_params = trans_info["config_params"]
    elif action =="execution":
        expected_params = trans_info['execution_params']

    else : expected_params ={}
    # st.write(param_dict)
    # Extract only the needed params for this model
    args = {k: param_dict[k] for k in expected_params if k in param_dict}

    # st.write(args)
    # Call the model function dynamically
    return func(**args)
