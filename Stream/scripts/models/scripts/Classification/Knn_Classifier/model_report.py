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
    
#     # Check if results contain KNN-specific metrics
#     if 'Accuracy' not in results['metrics']:
#         st.error("Invalid model results format for KNN classifier")
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
#         <h3>KNN Classifier Performance Report</h3>
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
    
#     with col4:
#         st.write("**Dataset Information:**")
#         st.write(f"- Number of features: **{len(results['features'])}**")
#         st.write(f"- Target variable: **{results['target']}**")
        
#         # Safely get number of classes without accessing 'df'
#         num_classes = "N/A"
#         if 'target_encoder' in results and results['target_encoder'] is not None:
#             num_classes = len(results['target_encoder'].classes_)
#         elif 'Confusion Matrix' in results['metrics']:
#             num_classes = results['metrics']['Confusion Matrix'].shape[0]
        
#         st.write(f"- Classes: **{num_classes}**")
        
#         if results['use_grid_search']:
#             cv_folds = results.get('cv_folds', 'N/A')
#             st.write(f"- CV Folds: **{cv_folds}**")
    
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

# # Optional: Add a function to display metrics in a more compact way
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

# --- Helper to safely retrieve metrics and convert JSON-safe lists back to arrays ---

def _get_common_metrics(metrics_snapshot: Dict[str, Any]):
    """Extracts common metrics and converts the confusion matrix list back to array."""
    class_report = metrics_snapshot['Classification Report']
    accuracy = metrics_snapshot['Accuracy']
    
    precision_avg = class_report.get('macro avg', {}).get('precision', 0)
    recall_avg = class_report.get('macro avg', {}).get('recall', 0)
    f1_avg = class_report.get('macro avg', {}).get('f1-score', 0)
    
    # CRITICAL: Convert the saved Python list back to a NumPy array for plotting
    conf_matrix_array = np.array(metrics_snapshot['Confusion Matrix'])
    
    class_df = pd.DataFrame(class_report).transpose().round(4)
    
    return accuracy, precision_avg, recall_avg, f1_avg, class_df, conf_matrix_array

# --- Granular Report Asset Generation Functions (Input: metrics_snapshot) ---

def create_ml_summary_text(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Generates the main summary text block (Report Asset Type: text)."""
    accuracy, precision_avg, recall_avg, f1_avg, _, _ = _get_common_metrics(metrics_snapshot)

    text = """
    ### KNN Classifier Performance Summary
    - **Features:** {features}
    - **Target:** {target}
    - **Accuracy:** **{accuracy:.4f}**
    - **Precision (Macro):** {precision:.4f}
    - **F1-Score (Macro):** {f1:.4f}
    - **Best Parameters:** {best_params}
    - **Training Method:** {grid_search}
    """.format(
        features=", ".join(metrics_snapshot['features']),
        target=metrics_snapshot['target'],
        accuracy=accuracy,
        precision=precision_avg,
        f1=f1_avg,
        best_params=metrics_snapshot['Best Parameters'],
        grid_search="Grid Search" if metrics_snapshot['use_grid_search'] else "Manual"
    )
    # The returned dictionary is the Report Asset Token (must be returned, not st.written)
    return {"type": "text", "content": text}


def create_confusion_matrix_plot(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Generates the Confusion Matrix plot (Report Asset Type: plot)."""
    _, _, _, _, _, conf_matrix_array = _get_common_metrics(metrics_snapshot)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(conf_matrix_array, 
                annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=True, yticklabels=True)
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.set_title('Confusion Matrix')
    plt.close(fig)
    
    # The returned dictionary is the Report Asset Token
    return {"type": "plot", "content": fig}


def create_classification_report_table(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Generates the Classification Report table (Report Asset Type: dataframe)."""
    _, _, _, _, class_df, _ = _get_common_metrics(metrics_snapshot)

    # The returned dictionary is the Report Asset Token
    return {"type": "dataframe", "content": class_df}


def create_performance_metrics_plot(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Generates the main Accuracy/Precision/Recall/F1 bar chart (Report Asset Type: plot)."""
    accuracy, precision_avg, recall_avg, f1_avg, _, _ = _get_common_metrics(metrics_snapshot)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [accuracy, precision_avg, recall_avg, f1_avg]
    
    bars = ax.bar(metrics, values, color=['#4CAF50', '#2196F3', '#FF9800', '#9C27B0'])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Metrics')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.4f}', ha='center', va='bottom')
                
    plt.close(fig)
    
    # The returned dictionary is the Report Asset Token
    return {"type": "plot", "content": fig}


# --- ML Page Display Function (Original model_report logic, now decoupled) ---

def display_ml_report(metrics_snapshot: Dict[str, Any]):
    """
    Function to be called on the main ML page. It uses the granular functions
    to generate and display all components using Streamlit commands.
    """
    if not metrics_snapshot:
        st.error("No model data provided for display.")
        return

    st.markdown("---")
    st.markdown("## KNN Model Analysis (Live View)")
    
    # Get all the report assets
    summary_asset = create_ml_summary_text(metrics_snapshot)
    table_asset = create_classification_report_table(metrics_snapshot)
    metrics_plot_asset = create_performance_metrics_plot(metrics_snapshot)
    conf_matrix_asset = create_confusion_matrix_plot(metrics_snapshot)
    
    # Display Summary
    st.markdown(summary_asset['content'], unsafe_allow_html=True)
    
    # Display Metrics Plot
    st.subheader("📈 Key Metrics Visualization")
    st.pyplot(metrics_plot_asset['content'])
    
    # Display Confusion Matrix
    st.subheader("🎯 Confusion Matrix")
    st.pyplot(conf_matrix_asset['content'])
    
    # Display Classification Table
    st.subheader("📊 Detailed Classification Report")
    st.dataframe(table_asset['content'], use_container_width=True)
    
    # NOTE: You can now port the rest of your original display logic here 
    # (e.g., Performance Interpretation, Hyperparameter details) using helper functions 
    # or by accessing values from _get_common_metrics(metrics_snapshot).

# For compatibility with your old system, we keep the original function name, 
# but it now just calls the decoupled display function.
def model_report():
    found = False 
    for model_info in st.session_state.pipeline.get("ML", []):
        if model_info.get("model name") == 'KNN':
            if 'metrics_snapshot' in model_info:
                found = True
                model_results = model_info.get("metrics_snapshot", {})
                break
    # if 'model_results' not in st.session_state or 'metrics' not in st.session_state.model_results:
    if not found:
        st.error("No model results found. Please create a model first.")
        return
    
    # Pass the JSON-safe metrics snapshot (which is stored under the 'metrics' key)
    display_ml_report(model_results)
