# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# def model_report():
#     if 'model_results' not in st.session_state:
#         st.error("No model results found. Please create a model first.")
#         return
    
#     results = st.session_state.model_results
    
#     # Check if results contain Random Forest-specific metrics
#     if 'Accuracy' not in results['metrics']:
#         st.error("Invalid model results format for Random Forest classifier")
#         return
    
#     # Extract metrics from classification report
#     class_report = results['metrics']['Classification Report']
    
#     # Calculate macro averages for Precision, Recall, F1-Score
#     precision_avg = class_report.get('macro avg', {}).get('precision', 0)
#     recall_avg = class_report.get('macro avg', {}).get('recall', 0)
#     f1_avg = class_report.get('macro avg', {}).get('f1-score', 0)
#     accuracy = results['metrics']['Accuracy']
    
#     st.markdown("""
#     <div class="report-container">
#         <h3>Random Forest Classifier Performance Report</h3>
#         <div class="metric-card">
#             <strong>Features:</strong> {features}<br>
#             <strong>Target:</strong> {target}<br>
#             <strong>Grid Search Used:</strong> {grid_search}
#         </div>
#         <div class="metric-card">
#             <strong>Accuracy:</strong> {accuracy:.4f}<br>
#             <strong>Precision (Macro Avg):</strong> {precision:.4f}<br>
#             <strong>Recall (Macro Avg):</strong> {recall:.4f}<br>
#             <strong>F1-Score (Macro Avg):</strong> {f1:.4f}
#         </div>
#         <div class="metric-card">
#             <strong>Best Parameters:</strong> {best_params}
#         </div>
#     </div>
#     """.format(
#         features=", ".join(results['features']),
#         target=results['target'],
#         grid_search="Yes" if results['use_grid_search'] else "No",
#         accuracy=accuracy,
#         precision=precision_avg,
#         recall=recall_avg,
#         f1=f1_avg,
#         best_params=results['metrics']['Best Parameters']
#     ), unsafe_allow_html=True)
    
#     # Feature Importance Visualization
#     if 'feature_importances' in results['metrics']:
#         st.subheader("🔍 Feature Importance")
        
#         feature_importances = results['metrics']['feature_importances']
#         features = results['features']
        
#         # Create feature importance DataFrame
#         importance_df = pd.DataFrame({
#             'Feature': features,
#             'Importance': feature_importances
#         }).sort_values('Importance', ascending=True)
        
#         # Plot feature importance
#         fig, ax = plt.subplots(figsize=(10, 6))
#         y_pos = np.arange(len(importance_df))
#         ax.barh(y_pos, importance_df['Importance'])
#         ax.set_yticks(y_pos)
#         ax.set_yticklabels(importance_df['Feature'])
#         ax.set_xlabel('Importance')
#         ax.set_title('Feature Importance Ranking')
#         plt.tight_layout()
#         st.pyplot(fig)
    
#     # Detailed Classification Report Section
#     st.subheader("📊 Detailed Classification Report")
    
#     # Convert classification report to DataFrame for better display
#     class_df = pd.DataFrame(class_report).transpose().round(4)
    
#     # Display the full classification report as a table
#     st.dataframe(class_df, use_container_width=True)
    
#     # Metrics Visualization
#     st.subheader("📈 Metrics Visualization")
    
#     # Create a bar chart for main metrics
#     fig, ax = plt.subplots(figsize=(10, 6))
    
#     metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
#     values = [accuracy, precision_avg, recall_avg, f1_avg]
    
#     bars = ax.bar(metrics, values, color=['#4CAF50', '#2196F3', '#FF9800', '#9C27B0'])
#     ax.set_ylim(0, 1)
#     ax.set_ylabel('Score')
#     ax.set_title('Model Performance Metrics')
    
#     # Add value labels on bars
#     for bar, value in zip(bars, values):
#         height = bar.get_height()
#         ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
#                 f'{value:.4f}', ha='center', va='bottom')
    
#     st.pyplot(fig)
    
#     # Confusion Matrix Visualization
#     st.subheader("🎯 Confusion Matrix")
#     fig2, ax2 = plt.subplots(figsize=(8, 6))
    
#     # Create heatmap for confusion matrix
#     sns.heatmap(results['metrics']['Confusion Matrix'], 
#                 annot=True, fmt='d', cmap='Blues', ax=ax2,
#                 xticklabels=True, yticklabels=True)
    
#     ax2.set_xlabel('Predicted Labels')
#     ax2.set_ylabel('True Labels')
#     ax2.set_title('Confusion Matrix')
#     st.pyplot(fig2)
    
#     # Per-Class Metrics Breakdown
#     st.subheader("📋 Per-Class Metrics Breakdown")
    
#     # Filter out average rows for per-class display
#     per_class_df = class_df[~class_df.index.isin(['accuracy', 'macro avg', 'weighted avg'])]
    
#     if not per_class_df.empty:
#         # Display per-class metrics
#         col1, col2 = st.columns(2)
        
#         with col1:
#             st.write("**Precision by Class:**")
#             for class_name, row in per_class_df.iterrows():
#                 st.write(f"- Class {class_name}: {row.get('precision', 0):.4f}")
        
#         with col2:
#             st.write("**Recall by Class:**")
#             for class_name, row in per_class_df.iterrows():
#                 st.write(f"- Class {class_name}: {row.get('recall', 0):.4f}")
        
#         # Per-class metrics visualization
#         fig3, ax3 = plt.subplots(figsize=(12, 6))
        
#         if 'precision' in per_class_df.columns and 'recall' in per_class_df.columns and 'f1-score' in per_class_df.columns:
#             per_class_df[['precision', 'recall', 'f1-score']].plot(kind='bar', ax=ax3)
#             ax3.set_title('Precision, Recall, and F1-Score by Class')
#             ax3.set_ylabel('Score')
#             ax3.set_xlabel('Class')
#             ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#             plt.xticks(rotation=45)
#             st.pyplot(fig3)
    
#     # Model Configuration Details
#     st.subheader("⚙️ Model Configuration")
    
#     col3, col4 = st.columns(2)
    
#     with col3:
#         st.write("**Hyperparameters:**")
#         best_params = results['metrics']['Best Parameters']
#         if isinstance(best_params, dict):
#             for param, value in best_params.items():
#                 st.write(f"- **{param}:** {value}")
#         else:
#             st.write(f"- {best_params}")
        
#         st.write("**Training Method:**")
#         st.write(f"- Grid Search: {'✅ Yes' if results['use_grid_search'] else '❌ No'}")
#         if results['use_grid_search']:
#             st.write(f"- CV Folds: **{results.get('cv_folds', 'N/A')}**")
    
#     with col4:
#         st.write("**Dataset Information:**")
#         st.write(f"- Number of features: **{len(results['features'])}**")
#         st.write(f"- Target variable: **{results['target']}**")
        
#         # Safely get number of classes
#         num_classes = "N/A"
#         if 'target_encoder' in results and results['target_encoder'] is not None:
#             num_classes = len(results['target_encoder'].classes_)
#         elif 'Confusion Matrix' in results['metrics']:
#             num_classes = results['metrics']['Confusion Matrix'].shape[0]
        
#         st.write(f"- Classes: **{num_classes}**")
#         st.write(f"- Number of trees: **{best_params.get('n_estimators', 'N/A')}**")
    
#     # Performance Interpretation
#     st.subheader("📊 Performance Interpretation")
    
#     if accuracy >= 0.9:
#         st.success("**Excellent Performance** - The model shows outstanding classification ability!")
#     elif accuracy >= 0.8:
#         st.info("**Good Performance** - The model performs well on the classification task.")
#     elif accuracy >= 0.7:
#         st.warning("**Fair Performance** - The model has acceptable performance but could be improved.")
#     else:
#         st.error("**Poor Performance** - Consider feature engineering, parameter tuning, or trying a different algorithm.")
    
#     # Key Metrics Summary
#     st.subheader("🔑 Key Metrics Summary")
    
#     metrics_summary = {
#         "Accuracy": f"{accuracy:.4f}",
#         "Precision (Macro)": f"{precision_avg:.4f}",
#         "Recall (Macro)": f"{recall_avg:.4f}",
#         "F1-Score (Macro)": f"{f1_avg:.4f}"
#     }
    
#     summary_df = pd.DataFrame(list(metrics_summary.items()), columns=['Metric', 'Value'])
#     st.table(summary_df)

# def display_compact_metrics():
#     """Alternative compact metrics display"""
#     if 'model_results' not in st.session_state:
#         return
    
#     results = st.session_state.model_results
#     class_report = results['metrics']['Classification Report']
    
#     precision_avg = class_report.get('macro avg', {}).get('precision', 0)
#     recall_avg = class_report.get('macro avg', {}).get('recall', 0)
#     f1_avg = class_report.get('macro avg', {}).get('f1-score', 0)
#     accuracy = results['metrics']['Accuracy']
    
#     # Create a compact metrics card
#     col1, col2, col3, col4 = st.columns(4)
    
#     with col1:
#         st.metric("Accuracy", f"{accuracy:.4f}")
#     with col2:
#         st.metric("Precision", f"{precision_avg:.4f}")
#     with col3:
#         st.metric("Recall", f"{recall_avg:.4f}")
#     with col4:
#         st.metric("F1-Score", f"{f1_avg:.4f}")


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any

# --- Helper to safely retrieve metrics ---
def _get_common_metrics(metrics_snapshot: Dict[str, Any]):
    """Extracts common metrics from the snapshot."""
    try:
        # Get basic metrics
        accuracy = metrics_snapshot.get('Accuracy', 0)
        class_report = metrics_snapshot.get('Classification Report', {})
        
        # Handle classification report - check different possible structures
        precision_avg = recall_avg = f1_avg = 0
        if isinstance(class_report, dict):
            # Try macro avg first
            if 'macro avg' in class_report and isinstance(class_report['macro avg'], dict):
                macro_avg = class_report['macro avg']
                precision_avg = macro_avg.get('precision', 0)
                recall_avg = macro_avg.get('recall', 0)
                f1_avg = macro_avg.get('f1-score', 0)
            # Try weighted avg
            elif 'weighted avg' in class_report and isinstance(class_report['weighted avg'], dict):
                weighted_avg = class_report['weighted avg']
                precision_avg = weighted_avg.get('precision', 0)
                recall_avg = weighted_avg.get('recall', 0)
                f1_avg = weighted_avg.get('f1-score', 0)
            # Try direct keys for binary classification
            elif '1' in class_report and isinstance(class_report['1'], dict):
                class_1 = class_report['1']
                precision_avg = class_1.get('precision', 0)
                recall_avg = class_1.get('recall', 0)
                f1_avg = class_1.get('f1-score', 0)
        
        # Handle confusion matrix - ensure it's properly converted from list
        conf_matrix_data = metrics_snapshot.get('Confusion Matrix', [])
        if isinstance(conf_matrix_data, list) and len(conf_matrix_data) > 0:
            conf_matrix_array = np.array(conf_matrix_data)
        else:
            conf_matrix_array = np.array([])
        
        # Handle feature importances
        feature_importances = metrics_snapshot.get('feature_importances', [])
        features = metrics_snapshot.get('features', [])
        
        # Create classification report dataframe
        class_df = pd.DataFrame()
        if isinstance(class_report, dict) and class_report:
            # Filter out non-dict items and create dataframe
            report_data = {}
            for key, value in class_report.items():
                if isinstance(value, dict):
                    report_data[key] = value
            if report_data:
                class_df = pd.DataFrame(report_data).transpose().round(4)
        
        return accuracy, precision_avg, recall_avg, f1_avg, class_df, conf_matrix_array, feature_importances, features
    
    except Exception:
        return 0, 0, 0, 0, pd.DataFrame(), np.array([]), [], []

# --- Report Asset Generation Functions ---
def create_rf_summary_text(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Generates the main summary text block."""
    accuracy, precision_avg, recall_avg, f1_avg, _, _, _, features = _get_common_metrics(metrics_snapshot)

    # Format best parameters for display
    best_params = metrics_snapshot.get('Best Parameters', {})
    if isinstance(best_params, dict):
        params_str = ", ".join([f"{k}: {v}" for k, v in best_params.items()])
    else:
        params_str = str(best_params)

    text = f"""
    ### Random Forest Classifier Performance Summary
    - **Features:** {', '.join(features) if features else 'None'}
    - **Target:** {metrics_snapshot.get('target', 'Unknown')}
    - **Accuracy:** **{accuracy:.4f}**
    - **Precision (Macro):** {precision_avg:.4f}
    - **Recall (Macro):** {recall_avg:.4f}
    - **F1-Score (Macro):** {f1_avg:.4f}
    - **Best Parameters:** {params_str}
    - **Training Method:** {"Grid Search" if metrics_snapshot.get('use_grid_search', False) else "Manual"}
    """
    
    return {"type": "text", "content": text}

def create_rf_confusion_matrix_plot(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Generates the Confusion Matrix plot."""
    _, _, _, _, _, conf_matrix_array, _, _ = _get_common_metrics(metrics_snapshot)

    fig, ax = plt.subplots(figsize=(8, 6))
    
    if conf_matrix_array.size > 0 and conf_matrix_array.ndim == 2:
        sns.heatmap(conf_matrix_array, 
                    annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=True, yticklabels=True)
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        ax.set_title('Confusion Matrix - Random Forest')
    else:
        ax.text(0.5, 0.5, 'No confusion matrix data available', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Confusion Matrix - Random Forest')
    
    plt.tight_layout()
    plt.close(fig)
    
    return {"type": "plot", "content": fig}

def create_rf_classification_report_table(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Generates the Classification Report table."""
    _, _, _, _, class_df, _, _, _ = _get_common_metrics(metrics_snapshot)
    return {"type": "dataframe", "content": class_df}

def create_rf_performance_plot(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Generates the main metrics bar chart."""
    accuracy, precision_avg, recall_avg, f1_avg, _, _, _, _ = _get_common_metrics(metrics_snapshot)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [accuracy, precision_avg, recall_avg, f1_avg]
    
    bars = ax.bar(metrics, values, color=['#4CAF50', '#2196F3', '#FF9800', '#9C27B0'])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score')
    ax.set_title('Random Forest Performance Metrics')
    
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.4f}', ha='center', va='bottom')
                
    plt.tight_layout()
    plt.close(fig)
    
    return {"type": "plot", "content": fig}

def create_rf_feature_importance_plot(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Generates the Feature Importance visualization."""
    _, _, _, _, _, _, feature_importances, features = _get_common_metrics(metrics_snapshot)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if (feature_importances and features and 
        len(feature_importances) == len(features)):
        
        # Create feature importance DataFrame
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': feature_importances
        }).sort_values('Importance', ascending=True)
        
        # Plot feature importance
        y_pos = np.arange(len(importance_df))
        bars = ax.barh(y_pos, importance_df['Importance'], color='skyblue', edgecolor='navy')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(importance_df['Feature'])
        ax.set_xlabel('Importance Score')
        ax.set_title('Random Forest - Feature Importance Ranking')
        
        # Add value labels
        for i, (bar, importance) in enumerate(zip(bars, importance_df['Importance'])):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2., 
                   f'{importance:.4f}', ha='left', va='center', fontsize=8)
    else:
        ax.text(0.5, 0.5, 'No feature importance data available', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Random Forest - Feature Importance')
    
    plt.tight_layout()
    plt.close(fig)
    
    return {"type": "plot", "content": fig}

# --- Main Model Report Function ---
def model_report():
    """Main function to find and display Random Forest model results."""
    
    # Search for Random Forest model
    found = False
    model_results = None
    
    if hasattr(st.session_state, 'pipeline') and "ML" in st.session_state.pipeline:
        for model_info in st.session_state.pipeline["ML"]:
            model_name = (model_info.get('model name') or 
                         model_info.get('name') or 
                         model_info.get('model_type') or 
                         model_info.get('type', ''))
            
            # Match Random Forest model
            if model_name == 'Random Forest Classifier':
                if 'metrics_snapshot' in model_info:
                    model_results = model_info['metrics_snapshot']
                    found = True
                    break
    
    if not found:
        st.error("No Random Forest model results found. Please train a model first.")
        return
    
    # Display the complete ML report
    st.markdown("---")
    st.markdown("## Random Forest Model Analysis")
    
    try:
        # Get all the report assets
        summary_asset = create_rf_summary_text(model_results)
        table_asset = create_rf_classification_report_table(model_results)
        metrics_plot_asset = create_rf_performance_plot(model_results)
        conf_matrix_asset = create_rf_confusion_matrix_plot(model_results)
        feature_importance_asset = create_rf_feature_importance_plot(model_results)
        
        # Display Summary
        st.markdown(summary_asset['content'])
        
        # Display Metrics Plot
        st.subheader("📈 Key Metrics Visualization")
        st.pyplot(metrics_plot_asset['content'])
        
        # Display Feature Importance
        st.subheader("🔍 Feature Importance")
        st.pyplot(feature_importance_asset['content'])
        
        # Display Confusion Matrix
        st.subheader("🎯 Confusion Matrix")
        st.pyplot(conf_matrix_asset['content'])
        
        # Display Classification Table
        st.subheader("📊 Detailed Classification Report")
        if not table_asset['content'].empty:
            st.dataframe(table_asset['content'], use_container_width=True)
        else:
            st.info("Classification report not available")
            
    except Exception as e:
        st.error(f"Error displaying Random Forest model report: {e}")