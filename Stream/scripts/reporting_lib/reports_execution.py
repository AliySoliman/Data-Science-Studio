import streamlit as st
from typing import Dict, Any, Callable
import streamlit as st
from typing import Dict, Any, Callable
import matplotlib.pyplot as plt
import plotly.io as pio
from io import BytesIO
import base64
from viz_lib import visualization_execution
from visualizations import apply_global_transformations
from viz_lib.visualization_ui import apply_individual_transformations
        
# ADD VISUALIZATION ASSET GENERATOR FUNCTIONS FIRST
def create_visualization_plot(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Generate interactive visualization plot for the report"""
    try:
        # Get visualization data from snapshot
        viz_config = metrics_snapshot.get('viz_config', {})
        viz_comments = metrics_snapshot.get('viz_comments', '')
        
        # Import visualization execution to generate the plot
        
        # Get global data
        global_data = apply_global_transformations()
        
        # Apply individual transformations if specified
        individual_transformations = viz_config.get('transformations', ['original'])
        plot_data = global_data
        
        if individual_transformations and not (len(individual_transformations) == 1 and individual_transformations[0] == 'original'):
            plot_data = apply_individual_transformations(global_data, individual_transformations)
        
        # Generate the plot
        fig = visualization_execution.generate_plot(plot_data, viz_config)
        
        # Return the interactive Plotly figure directly
        return {
            "type": "plotly",  # NEW TYPE FOR INTERACTIVE PLOTS
            "content": fig
        }
        
    except Exception as e:
        # Return error message as text
        error_html = f"""
        <div style="background-color: #ffe6e6; padding: 15px; border-radius: 5px; border-left: 4px solid #ff4444;">
            <h4 style="margin-top: 0; color: #cc0000;">❌ Error Generating Visualization</h4>
            <p><strong>Error:</strong> {str(e)}</p>
        </div>
        """
        
        return {
            "type": "text", 
            "content": error_html
        }

def create_visualization_comments(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Generate visualization comments for the report"""
    viz_comments = metrics_snapshot.get('viz_comments', '')
    viz_config = metrics_snapshot.get('viz_config', {})
    
    # Get transformation info
    global_transformations = st.session_state.get('global_transformations', ['original'])
    individual_transformations = viz_config.get('transformations', ['original'])
    
    comments_html = f"""
    <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 4px solid #007bff;">
        <h4 style="margin-top: 0; color: #333;">📊 {viz_config.get('title', 'Visualization')}</h4>
        <p><strong>Chart Type:</strong> {viz_config.get('chart_type', 'N/A')}</p>
        <p><strong>Global Transformations:</strong> {', '.join(global_transformations) if global_transformations else 'None'}</p>
        <p><strong>Individual Transformations:</strong> {', '.join(individual_transformations) if individual_transformations else 'None'}</p>
        <p><strong>Comments:</strong> {viz_comments if viz_comments else 'No comments provided.'}</p>
    </div>
    """
    
    return {
        "type": "text",
        "content": comments_html
    }
from models.scripts.Classification.Knn_Classifier.model_report   import (
    create_ml_summary_text as knn_create_ml_summary_text,
    create_confusion_matrix_plot as knn_create_confusion_matrix_plot,
    create_classification_report_table as knn_create_classification_report_table,
    create_performance_metrics_plot as knn_create_performance_metrics_plot
)
#********************************************************************************************
from models.scripts.Regression.Knn_Regressor.model_report   import (
    create_ml_summary_text as knn_Regressor_create_ml_summary_text,
    create_classification_report_table as knn_Regressor_create_classification_report_table,
    create_performance_metrics_plot as knn_Regressor_create_performance_metrics_plot,
    create_knn_prediction_analysis_plot as knn_Regressor_create_prediction_analysis_plot,
    create_knn_residual_analysis_plot as knn_Regressor_create_residual_analysis_plot
)
# Import Logistic Regression report functions
from models.scripts.Classification.Logistic_Regression.model_report import (
    create_lr_summary_text as lr_create_ml_summary_text,
    create_confusion_matrix_plot as lr_create_confusion_matrix_plot,
    create_classification_report_table as lr_create_classification_report_table,
    create_performance_metrics_plot as lr_create_performance_metrics_plot,
    create_roc_curve_plot_wrapper as lr_create_roc_curve_plot,
    create_hyperparameters_table as lr_create_hyperparameters_table,
    create_dataset_info_table as lr_create_dataset_info_table,
    create_performance_interpretation as lr_create_performance_interpretation
)

#********************************************************************************************
from models.scripts.Regression.Gradient_Boosting_Regressor.model_report   import (
    create_ml_summary_text as gbr_create_ml_summary_text,
    create_classification_report_table as gbr_create_classification_report_table,
    create_performance_metrics_plot as gbr_create_performance_metrics_plot,
    create_error_metrics_plot as gbr_create_error_metrics_plot,
    create_prediction_analysis_plot as gbr_create_prediction_analysis_plot,
    create_residual_analysis_plot as gbr_create_residual_analysis_plot
)

#********************************************************************************************
from models.scripts.Unsupervised.DBSCAN.model_report   import (
    create_ml_summary_text as dbscan_create_ml_summary_text,
    create_classification_report_table as dbscan_create_classification_report_table,
    create_cluster_distribution_plot as dbscan_create_cluster_distribution_plot,
    create_quality_metrics_plot as dbscan_create_quality_metrics_plot,
    create_parameter_importance_plot as dbscan_create_parameter_importance_plot
)
#********************************************************************************************
from models.scripts.Classification.Naive_Bayes.model_report import (
    create_summary_text as nb_create_summary_text,
    create_classification_report_table as nb_create_classification_report_table,
    create_performance_metrics_plot as nb_create_performance_metrics_plot,
    create_confusion_matrix_plot as nb_create_confusion_matrix_plot,
    create_class_distribution_plot as nb_create_class_distribution_plot
)
#********************************************************************************************
from models.scripts.Classification.SVC.model_report import (
    create_svc_summary_text as svc_create_summary_text,
    create_svc_classification_report_table as svc_create_classification_report_table,
    create_svc_metrics_plot as svc_create_performance_metrics_plot,
    create_svc_confusion_matrix_plot as svc_create_confusion_matrix_plot,
    create_svc_per_class_plot as svc_create_per_class_plot,
    create_svc_model_config_block as svc_create_model_config_block,
    create_svc_performance_interpretation as svc_create_performance_interpretation
)   
#********************************************************************************************
from models.scripts.Classification.Random_Forest_Classifier.model_report import (
    create_rf_summary_text,
    create_rf_performance_plot,
    create_rf_confusion_matrix_plot,
    create_rf_feature_importance_plot,
    create_rf_classification_report_table
)
#********************************************************************************************
from models.scripts.Classification.Gradient_Boosting_Classifier.model_report import (
    create_gbc_summary_text as gbc_create_summary_text,
    create_gbc_performance_plot as gbc_create_performance_plot,
    create_gbc_confusion_matrix_plot as gbc_create_confusion_matrix_plot,
    create_gbc_feature_importance_plot as gbc_create_feature_importance_plot,
    create_gbc_classification_report_table as gbc_create_classification_report_table
)
#********************************************************************************************


from models.scripts.Regression.Simple_Linear_Regression.model_report import (
    create_regression_summary_text as regression_create_summary_text,
    create_coefficients_table as regression_create_coefficients_table,
    create_residuals_plot as regression_create_residuals_plot,
    create_performance_comparison as regression_create_performance_comparison,
    create_feature_importance_plot as regression_create_feature_importance_plot
)

#********************************************************************************************
from models.scripts.Unsupervised.Kmeans.model_report import (
    create_clustering_summary_text as kmeans_create_summary_text,
    create_cluster_distribution_plot as kmeans_create_distribution_plot,
    create_metrics_radar_chart as kmeans_create_radar_chart,
    create_performance_metrics_table as kmeans_create_metrics_table,
    create_cluster_analysis_text as kmeans_create_analysis_text,        # ADD THIS
    create_configuration_details_text as kmeans_create_config_text      # ADD THIS
)

#********************************************************************************************
from models.scripts.Unsupervised.Hierarchical_Clustering.model_report   import (
    get_common_metrics as hierarchical_get_common_metrics,
    create_dendrogram_plot as hierarchical_create_dendrogram_plot,
    create_cluster_ditribution_plot as hierarchical_create_cluster_ditribution_plot,
    create_PCA_Projection_plot as hierarchical_create_PCA_Projection_plot,
    create_silhouette_plot as hierarchical_create_silhouette_plot
)
from models.scripts.Regression.Random_Forest_Regressor.model_report   import (
    create_ml_summary_text as RFR_Regressor_create_ml_summary_text,
    create_feature_importance_plot as RFR_Regressor_create_feature_importance_plot,
    create_prediction_analysis_plot as RFR_Regressor_create_prediction_analysis_plot,
    create_residual_analysis_plot as RFR_Regressor_create_residual_analysis_plot,
    model_configuration_insights as RFR_Regressor_model_configuration_insights,
    create_model_recommendations as RFR_Regressor_create_model_recommendations

)
#********************************************************************************************
from models.scripts.Classification.Xgboost_Classifier.model_report import (
    create_xgb_summary_text as xgbc_create_summary_text,
    create_xgb_performance_plot as xgbc_create_performance_plot,
    create_xgb_confusion_matrix_plot as xgbc_create_confusion_matrix_plot,
    create_xgb_feature_importance_plot as xgbc_create_feature_importance_plot,
    create_xgb_classification_report_table as xgbc_create_classification_report_table
)
#********************************************************************************************
from models.scripts.Regression.Xgboost_Regressor.model_report import (
    create_xgb_summary_text as xgbr_create_summary_text,
    create_xgb_performance_metrics_plot as xgbr_create_performance_metrics_plot,
    create_xgb_regression_report_table as xgbr_create_regression_report_table,
    create_xgb_error_metrics_plot as xgbr_create_error_metrics_plot,
    create_xgb_prediction_analysis_plot as xgbr_create_prediction_analysis_plot,
    create_xgb_residual_analysis_plot as xgbr_create_residual_analysis_plot
)
#********************************************************************************************
from models.scripts.Classification.Decision_Tree_Classifier.model_report import (
    create_dt_summary_text,
    create_dt_performance_plot,
    create_dt_confusion_matrix_plot,
    create_dt_tree_structure_plot,
    create_dt_feature_importance_plot,
    create_dt_classification_report_table
)

# Support Vector Regression
from models.scripts.Regression.SVR.model_report import (
    create_ml_summary_text as svr_create_ml_summary_text,
    create_performance_metrics_plot as svr_create_performance_metrics_plot,
    create_prediction_analysis_plot as svr_create_prediction_analysis_plot,
    create_classification_report_table as svr_create_classification_report_table,
    create_error_metrics_plot as svr_create_error_metrics_plot
)
# NOTE: Add imports for Statistics and Visualization asset generators here later

# Dictionary mapping report types and function names to the actual Callable objects
FUNCTION_MAP: Dict[str, Dict[str, Callable]] = {
    "KNN": {
        "create_ml_summary_text": knn_create_ml_summary_text,
        "create_confusion_matrix_plot": knn_create_confusion_matrix_plot,
        "create_classification_report_table": knn_create_classification_report_table,
        "create_performance_metrics_plot": knn_create_performance_metrics_plot,
    },
    "KNN_Regressor":{
        "create_ml_summary_text":knn_Regressor_create_ml_summary_text,
        "create_classification_report_table":knn_Regressor_create_classification_report_table,
        "create_performance_metrics_plot": knn_Regressor_create_performance_metrics_plot,
        "create_knn_prediction_analysis_plot": knn_Regressor_create_prediction_analysis_plot,
        "create_knn_residual_analysis_plot": knn_Regressor_create_residual_analysis_plot
    },
        "LR": {
        "create_ml_summary_text": lr_create_ml_summary_text,
        "create_confusion_matrix_plot": lr_create_confusion_matrix_plot,
        "create_classification_report_table": lr_create_classification_report_table,
        "create_performance_metrics_plot": lr_create_performance_metrics_plot,
        "create_roc_curve_plot_wrapper": lr_create_roc_curve_plot,
        "create_hyperparameters_table": lr_create_hyperparameters_table,
        "create_dataset_info_table": lr_create_dataset_info_table,
        "create_performance_interpretation": lr_create_performance_interpretation
    },
        "Gradient Boosting Regressor": {
        "create_ml_summary_text": gbr_create_ml_summary_text,
        "create_classification_report_table": gbr_create_classification_report_table,
        "create_performance_metrics_plot": gbr_create_performance_metrics_plot,
        "create_error_metrics_plot": gbr_create_error_metrics_plot,
        "create_prediction_analysis_plot": gbr_create_prediction_analysis_plot,
        "create_residual_analysis_plot": gbr_create_residual_analysis_plot
    },
    "DBSCAN": {
        "create_ml_summary_text": dbscan_create_ml_summary_text,
        "create_classification_report_table": dbscan_create_classification_report_table,
        "create_cluster_distribution_plot": dbscan_create_cluster_distribution_plot,
        "create_quality_metrics_plot": dbscan_create_quality_metrics_plot,
        "create_parameter_importance_plot": dbscan_create_parameter_importance_plot
    },
    "Naive Bayes Classifier": {
        "create_summary_text": nb_create_summary_text,
        "create_classification_report_table": nb_create_classification_report_table,
        "create_performance_metrics_plot": nb_create_performance_metrics_plot,
        "create_confusion_matrix_plot": nb_create_confusion_matrix_plot,
        "create_class_distribution_plot": nb_create_class_distribution_plot,
    },
    "Support Vector Machine Classifier": {
        "create_summary_text": svc_create_summary_text,
        "create_classification_report_table": svc_create_classification_report_table,
        "create_performance_metrics_plot": svc_create_performance_metrics_plot,
        "create_confusion_matrix_plot": svc_create_confusion_matrix_plot,
        "create_per_class_plot": svc_create_per_class_plot,
        "create_model_config_block": svc_create_model_config_block,
        "create_performance_interpretation": svc_create_performance_interpretation,
    },
        "Random Forest Classifier": {
        "create_rf_summary_text": create_rf_summary_text,
        "create_rf_performance_plot": create_rf_performance_plot,
        "create_rf_confusion_matrix_plot": create_rf_confusion_matrix_plot,
        "create_rf_feature_importance_plot": create_rf_feature_importance_plot,
        "create_rf_classification_report_table": create_rf_classification_report_table
    },
    "Gradient Boosting Classifier": {
        "create_gbc_summary_text": gbc_create_summary_text,
        "create_gbc_performance_plot": gbc_create_performance_plot,
        "create_gbc_confusion_matrix_plot": gbc_create_confusion_matrix_plot,
        "create_gbc_feature_importance_plot": gbc_create_feature_importance_plot,
        "create_gbc_classification_report_table": gbc_create_classification_report_table
    },

    "KMeans Clustering": {
        "create_clustering_summary_text": kmeans_create_summary_text,
        "create_cluster_distribution_plot": kmeans_create_distribution_plot,
        "create_metrics_radar_chart": kmeans_create_radar_chart,
        "create_performance_metrics_table": kmeans_create_metrics_table,
        "create_cluster_analysis_text": kmeans_create_analysis_text,        # ADD THIS
        "create_configuration_details_text": kmeans_create_config_text      # ADD THIS
    },
        "Linear Regression": {
        "create_regression_summary_text": regression_create_summary_text,
        "create_coefficients_table": regression_create_coefficients_table,
        "create_residuals_plot": regression_create_residuals_plot,
        "create_performance_comparison": regression_create_performance_comparison,
        "create_feature_importance_plot": regression_create_feature_importance_plot,
    },
    "Hierarchical Clustering":{
        "get_common_metrics":hierarchical_get_common_metrics,
        "create_dendrogram_plot":hierarchical_create_dendrogram_plot,
        "create_cluster_ditribution_plot":hierarchical_create_cluster_ditribution_plot,
        "create_PCA_Projection_plot":hierarchical_create_PCA_Projection_plot,
        "create_silhouette_plot":hierarchical_create_silhouette_plot,
    },
    "Random Forest Regression":{
        "create_ml_summary_text":RFR_Regressor_create_ml_summary_text, 
        "create_feature_importance_plot":RFR_Regressor_create_feature_importance_plot,
        "create_prediction_analysis_plot":RFR_Regressor_create_prediction_analysis_plot,
        "create_residual_analysis_plot":RFR_Regressor_create_residual_analysis_plot,
        "model_configuration_insights":RFR_Regressor_model_configuration_insights,
        "create_model_recommendations":RFR_Regressor_create_model_recommendations
    },
    "XGBoost Classifier": {
        "create_xgb_summary_text": xgbc_create_summary_text,
        "create_xgb_performance_plot": xgbc_create_performance_plot,
        "create_xgb_confusion_matrix_plot": xgbc_create_confusion_matrix_plot,
        "create_xgb_feature_importance_plot": xgbc_create_feature_importance_plot,
        "create_xgb_classification_report_table": xgbc_create_classification_report_table,
    },
    "XGBoost Regressor": {
        "create_xgb_summary_text": xgbr_create_summary_text,
        "create_xgb_performance_metrics_plot": xgbr_create_performance_metrics_plot,
        "create_xgb_regression_report_table": xgbr_create_regression_report_table,
        "create_xgb_error_metrics_plot": xgbr_create_error_metrics_plot,
        "create_xgb_prediction_analysis_plot": xgbr_create_prediction_analysis_plot,
        "create_xgb_residual_analysis_plot": xgbr_create_residual_analysis_plot
    },
    "Decision Tree Classifier": {
        "create_dt_summary_text": create_dt_summary_text,
        "create_dt_performance_plot": create_dt_performance_plot,
        "create_dt_confusion_matrix_plot": create_dt_confusion_matrix_plot,
        "create_dt_tree_structure_plot": create_dt_tree_structure_plot,
        "create_dt_feature_importance_plot": create_dt_feature_importance_plot,
        "create_dt_classification_report_table": create_dt_classification_report_table,
        # Aliases to support existing session items before mapping update
        "create_rf_summary_text": create_dt_summary_text,
        "create_rf_performance_plot": create_dt_performance_plot,
        "create_rf_confusion_matrix_plot": create_dt_confusion_matrix_plot,
        "create_rf_classification_report_table": create_dt_classification_report_table
    },
    
    "Support Vector Regression": {
        "create_ml_summary_text": svr_create_ml_summary_text,
        "create_performance_metrics_plot": svr_create_performance_metrics_plot,
        "create_prediction_analysis_plot": svr_create_prediction_analysis_plot,
        "create_classification_report_table": svr_create_classification_report_table,
        "create_error_metrics_plot": svr_create_error_metrics_plot
    },
    # ADD VISUALIZATION FUNCTIONS
    "Visualization": {
        "create_visualization_plot": create_visualization_plot,
        "create_visualization_comments": create_visualization_comments,
    }
    
}



def execute_report_asset(item_type: str, function_name: str, source_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Retrieves the asset generator function and executes it with the source data.
    
    Args:
        item_type: The broad category (e.g., 'Machine Learning').
        function_name: The string name of the granular function to call.
        source_data: The JSON-safe snapshot from the pipeline (e.g., metrics_snapshot).
        
    Returns:
        A structured dictionary containing the generated asset (plot object, dataframe object, or text).
    """
    
    if item_type not in FUNCTION_MAP:
        raise ValueError(f"Report type '{item_type}' not supported in the execution router.")
        
    if function_name not in FUNCTION_MAP[item_type]:
        raise ValueError(f"Function '{function_name}' not found for type '{item_type}'.")
        
    func = FUNCTION_MAP[item_type][function_name]
    
    # All granular functions for ML currently expect 'metrics_snapshot' as the argument
    # We pass the source_data dictionary which contains that snapshot
    return func(metrics_snapshot=source_data)
