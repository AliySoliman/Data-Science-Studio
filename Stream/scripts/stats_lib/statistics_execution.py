import pandas as pd
from typing import Dict, Any
from stats_lib.statistics_setup import statistics_config, apply_selected_statistics

def execute_statistics(type: str, action: str, param_dict: Dict[str, Any]):
    """
    Central router to execute configuration or execution functions for statistical analysis.
    This mirrors the logic in transformation_execution.py
    """

    # We only handle 'statistics' type here
    if type != "statistics":
        raise ValueError(f"Unknown type for execution: {type}")

    stat_info = {
        "config": statistics_config,
        "execution": apply_selected_statistics,
        # We need to explicitly define the parameter names for dynamic calling
        "config_params": ["choice", "df", "edit_values"],
        "execution_params": ["df", "step"]
    }
    
    if action not in stat_info:
         raise ValueError(f"Action '{action}' not found for model '{type}'")
         
    func = stat_info[action]
    
    # Determine the expected parameters for the selected action
    expected_params = stat_info[f"{action}_params"]

    # Extract only the needed params from the provided dictionary
    args = {k: param_dict[k] for k in expected_params if k in param_dict}

    # Call the statistical function dynamically
    return func(**args)