import streamlit as st
# from model_report import ML_REPORT_ASSETS
import uuid
from typing import List, Dict, Any


Models = {
"KNN Classifier" :{
    "create_ml_summary_text": "Model Summary (Text)",
    "create_performance_metrics_plot": "Performance Bar Chart (Plot)",
    "create_confusion_matrix_plot": "Confusion Matrix (Plot)",
    "create_classification_report_table": "Classification Report (Table)"
},
"KNN_Regressor" :{
    "create_ml_summary_text": "Model Summary (Text)",
    "create_performance_metrics_plot": "Performance Bar Chart (Plot)",
    "create_classification_report_table": "Classification Report (Table)"
},
   "Logistic Regression": {
        "create_ml_summary_text": "Model Summary (Text)",
        "create_performance_metrics_plot": "Performance Bar Chart (Plot)",
        "create_confusion_matrix_plot": "Confusion Matrix (Plot)",
        "create_classification_report_table": "Classification Report (Table)",
        "create_roc_curve_plot_wrapper": "ROC Curve (Plot)",
        "create_hyperparameters_table": "Hyperparameters (Text)",
        "create_dataset_info_table": "Dataset Information (Text)", 
        "create_performance_interpretation": "Performance Interpretation (Text)"
    },
    "Gradient Boosting Regressor": {
        "create_ml_summary_text": "Model Summary (Text)",
        "create_performance_metrics_plot": "R² Score Visualization (Plot)",
        "create_classification_report_table": "Regression Metrics (Table)", 
        "create_error_metrics_plot": "Error Metrics Visualization (Plot)"
    },
"DBSCAN": {
        "create_ml_summary_text": "Model Summary (Text)", 
        "create_classification_report_table": "Clustering Metrics (Table)",
        "create_cluster_distribution_plot": "Cluster Distribution (Plot)",
        "create_quality_metrics_plot": "Quality Metrics (Plot)",
        "create_parameter_importance_plot": "Parameter Configuration (Plot)"
    },
"Naive Bayes Classifier" :{
    "create_summary_text": "Model Summary (Text)",
    "create_performance_metrics_plot": "Performance Bar Chart (Plot)",
    "create_confusion_matrix_plot": "Confusion Matrix (Plot)",
    "create_class_distribution_plot": "Class Distribution (Plot)",
    "create_classification_report_table": "Classification Report (Table)"
},
"Support Vector Machine Classifier" :{
    "create_summary_text": "Model Summary (Text)",
    "create_classification_report_table": "Classification Report (Table)",
    "create_performance_metrics_plot": "Performance Bar Chart (Plot)",
    "create_confusion_matrix_plot": "Confusion Matrix (Plot)",
    "create_per_class_plot": "Per-Class Accuracy (Plot)",
    "create_model_config_block": "Model Configuration (Text)",
    "create_performance_interpretation": "Performance Interpretation (Text)"
},
"Random Forest Classifier": {  
        "create_rf_summary_text": "Model Summary (Text)",
        "create_rf_performance_plot": "Performance Bar Chart (Plot)",
        "create_rf_confusion_matrix_plot": "Confusion Matrix (Plot)", 
        "create_rf_feature_importance_plot": "Feature Importance (Plot)",
        "create_rf_classification_report_table": "Classification Report (Table)"
    },
    "Gradient Boosting Classifier": {  
    "create_gbc_summary_text": "Model Summary (Text)",
    "create_gbc_performance_plot": "Performance Bar Chart (Plot)",
    "create_gbc_confusion_matrix_plot": "Confusion Matrix (Plot)", 
    "create_gbc_feature_importance_plot": "Feature Importance (Plot)",
    "create_gbc_classification_report_table": "Classification Report (Table)"
},
    "KMeans Clustering": {
        "create_clustering_summary_text": "Clustering Summary (Text)",
        "create_cluster_distribution_plot": "Cluster Distribution (Plot)",
        "create_metrics_radar_chart": "Metrics Radar Chart (Plot)",
        "create_performance_metrics_table": "Performance Metrics (Table)",
        "create_cluster_analysis_text": "Performance Interpretation (Text)",           # ADD THIS
        "create_configuration_details_text": "Configuration Details (Text)"           # ADD THIS
    },
        "Linear Regression": {
        "create_regression_summary_text": "Regression Summary (Text)",
        "create_coefficients_table": "Feature Coefficients (Table)", 
        "create_residuals_plot": "Residuals Analysis (Plot)",
        "create_performance_comparison": "Performance Metrics (Table)",
        "create_feature_importance_plot": "Feature Importance (Plot)"
    },
    "Hierarchical Clustering":{
    "get_common_metrics":"Common Metrics (Text)",
    "create_dendrogram_plot":"Dendrogram (Plot)",
    "create_cluster_ditribution_plot":"Cluster Distribution (Plot)",
    "create_PCA_Projection_plot":"PCA Projection (Plot)",
    "create_silhouette_plot":"Silhouette Analysis (Plot)"

},
"Random Forest Regression":{
    "create_ml_summary_text":"Model Summary (Text)",
    "create_feature_importance_plot":"Feature Importance (Plot)",
    "create_prediction_analysis_plot":"Prediction Analysis (Plot)",
    "create_residual_analysis_plot":"Residual Analysis (Plot)",
    "model_configuration_insights":"Model Configuration & Insights (Text)",
    "create_model_recommendations":"Model Recommendations (Text)"
},
"XGBoost Classifier": {
    "create_xgb_summary_text": "Model Summary (Text)",
    "create_xgb_performance_plot": "Performance Bar Chart (Plot)",
    "create_xgb_confusion_matrix_plot": "Confusion Matrix (Plot)",
    "create_xgb_feature_importance_plot": "Feature Importance (Plot)",
    "create_xgb_classification_report_table": "Classification Report (Table)"
},
"XGBoost Regressor": {
    "create_xgb_summary_text": "Model Summary (Text)",
    "create_xgb_performance_metrics_plot": "R² Score Visualization (Plot)",
    "create_xgb_regression_report_table": "Regression Metrics (Table)", 
    "create_xgb_error_metrics_plot": "Error Metrics Visualization (Plot)"
},
"Support Vector Regression": {
    "create_ml_summary_text": "Model Summary (Text)",
    "create_performance_metrics_plot": "Performance Visualization (Plot)",
    "create_classification_report_table": "Regression Metrics (Table)", 
    "create_error_metrics_plot": "Error Distribution (Plot)"
},
"Decision Tree Classifier": {
    "create_summary_text": "Model Summary (Text)",
    "create_performance_metrics_plot": "Performance Bar Chart (Plot)",
    "create_confusion_matrix_plot": "Confusion Matrix (Plot)",
    "create_classification_report_table": "Classification Report (Table)",
    "create_feature_importance_plot": "Feature Importance (Plot)"
}

}

def save_report_item_to_pipeline(model_name: str, model_index: int, final_choices: List[Dict[str, str]]):
    """Saves the final, ordered choices to the pipeline['report_items']."""
    # 1. Create the final Report Item Token
    new_report_item = {
        "id": str(uuid.uuid4()), # Unique ID for caching
        "type": "Machine Learning",
        "pipeline_source_key": "ML", # Key in st.session_state.pipeline
        "pipeline_source_index": model_index, # Index of the model in the ML list
        "pipeline_step_name": model_name,
        "choices": final_choices
    }
    
    # 2. Add to the global report items list
    st.session_state.pipeline.setdefault("report_items", [])
    st.session_state.pipeline["report_items"].append(new_report_item)
    
    # Clear temporary dialog state
    if 'selected_ml_choices_temp' in st.session_state:
        del st.session_state.selected_ml_choices_temp
    if 'final_ordered_choices' in st.session_state:
        del st.session_state.final_ordered_choices
    
    # Close the expander by resetting the flag
    st.session_state.show_ml_dialogue = False
    st.success(f"Report item '{model_name}' added to the Report Page!")
    st.rerun()


def display_ml_reporting_ui(model_name: str, model_index: int):
    """
    Displays the Report Configuration UI inside a conditional expander.
    """
    st.write(model_name)
    # Use st.container() for consistent placement of the expander
    ML_REPORT_ASSETS = Models.get(model_name)

    if ML_REPORT_ASSETS is None:
        st.error(f"⚠️ Reporting configuration for **{model_name}** not found. Please check `reporting_lib/model_reporting_setup.py`.")
        return

    with st.container():
        
        # Check the flag to determine if the expander should be expanded
        is_expanded = st.session_state.get('show_ml_dialogue', False)
        
        with st.expander(f"⚙️ Configure Report Item: {model_name}", expanded=is_expanded):
            
            # --- Initialize Temporary State ---
            if 'selected_ml_choices_temp' not in st.session_state:
                # Initialize with all assets selected by default
                st.session_state.selected_ml_choices_temp = list(ML_REPORT_ASSETS.keys())
            
            # --- 1. Available Components (Selection) ---
            st.markdown("#### 1. Select Components to Include")
            
            available_keys = list(ML_REPORT_ASSETS.keys())
            
            selected_keys = []
            
            for key in available_keys:
                default_state = key in st.session_state.selected_ml_choices_temp
                checkbox_key = f"ml_report_choice_{key}_{model_index}"
                
                # Checkbox value change requires a rerun to update the list, but st.expander handles it better than dialog
                if st.checkbox(ML_REPORT_ASSETS[key], value=default_state, key=checkbox_key):
                    selected_keys.append(key)

            st.session_state.selected_ml_choices_temp = selected_keys
            
            # --- 2. Ordering and Customization ---
            st.markdown("#### 2. Order and Rename Components")
            
            # Create a mutable list of choices with default titles
            temp_ordered_choices: List[Dict[str, str]] = []
            for key in selected_keys:
                temp_ordered_choices.append({
                    "function_name": key,
                    # We use a unique key for st.text_input to ensure its value is updated immediately
                    "user_title": st.text_input(
                        f"Title for '{ML_REPORT_ASSETS[key]}'",
                        value=ML_REPORT_ASSETS[key],
                        key=f"ml_title_{key}_{model_index}"
                    )
                })

            # Initialize/Update final_ordered_choices state
            if 'final_ordered_choices' not in st.session_state or st.session_state.get('last_selected_keys') != selected_keys:
                # Reset if selection has changed or on first run
                st.session_state.final_ordered_choices = temp_ordered_choices
                st.session_state.last_selected_keys = selected_keys
            else:
                # Update titles in the state list based on text inputs
                for i in range(len(temp_ordered_choices)):
                    if i < len(st.session_state.final_ordered_choices):
                        st.session_state.final_ordered_choices[i]['user_title'] = temp_ordered_choices[i]['user_title']

            
            # Simple list display and reordering buttons
            st.markdown("**Current Order:**")
            
            reordered = False
            
            # Use the state list for display and reordering logic
            for i, choice in enumerate(st.session_state.final_ordered_choices):
                col1, col2, col3 = st.columns([8, 1, 1])
                
                with col1:
                    st.markdown(f"**{i + 1}.** {choice['user_title']} *({ML_REPORT_ASSETS[choice['function_name']]})*")
                
                with col2:
                    # Up button
                    if i > 0 and st.button("⬆️", key=f"up_{model_index}_{i}"):
                        st.session_state.final_ordered_choices[i], st.session_state.final_ordered_choices[i-1] = st.session_state.final_ordered_choices[i-1], st.session_state.final_ordered_choices[i]
                        reordered = True
                        
                with col3:
                    # Down button
                    if i < len(st.session_state.final_ordered_choices) - 1 and st.button("⬇️", key=f"down_{model_index}_{i}"):
                        st.session_state.final_ordered_choices[i], st.session_state.final_ordered_choices[i+1] = st.session_state.final_ordered_choices[i+1], st.session_state.final_ordered_choices[i]
                        reordered = True
            
            if reordered:
                st.rerun() # Rerun to display the new order immediately

            
            # --- 3. Action Buttons ---
            st.markdown("---")
            col_save, col_cancel = st.columns(2)
            
            with col_save:
                if st.button(f"Generate '{model_name}' Report Item", type="primary", use_container_width=True):
                    final_ordered_choices = st.session_state.final_ordered_choices
                    
                    if not final_ordered_choices:
                        st.error("Please select at least one component to include in the report.")
                        return

                    save_report_item_to_pipeline(model_name, model_index, final_ordered_choices)
            
            with col_cancel:
                if st.button("Cancel Report Setup", use_container_width=True):
                    # Close the expander
                    st.session_state.show_ml_dialogue = False
                    st.rerun()
