# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import roc_curve, auc

# def model_report():
#     if 'model_results' not in st.session_state:
#         st.error("No model results found. Please create a model first.")
#         return
    
#     results = st.session_state.model_results
    
#     # Check if results contain classification metrics
#     if 'Accuracy' not in results['metrics']:
#         st.error("Invalid model results format for Logistic Regression classifier")
#         return
    
#     # Extract metrics from classification report
#     class_report = results['metrics']['Classification Report']
    
#     # Calculate macro averages for Precision, Recall, F1-Score
#     precision_avg = class_report.get('macro avg', {}).get('precision', 0)
#     recall_avg = class_report.get('macro avg', {}).get('recall', 0)
#     f1_avg = class_report.get('macro avg', {}).get('f1-score', 0)
#     accuracy = results['metrics']['Accuracy']
#     auc_score = results['metrics'].get('AUC-ROC', 'N/A')
    
#     st.markdown("""
#     <div class="report-container">
#         <h3>Logistic Regression Classifier Performance Report</h3>
#         <div class="metric-card">
#             <strong>Features:</strong> {features}<br>
#             <strong>Target:</strong> {target}<br>
#             <strong>Grid Search Used:</strong> {grid_search}
#         </div>
#         <div class="metric-card">
#             <strong>Accuracy:</strong> {accuracy:.4f}<br>
#             <strong>Precision (Macro Avg):</strong> {precision:.4f}<br>
#             <strong>Recall (Macro Avg):</strong> {recall:.4f}<br>
#             <strong>F1-Score (Macro Avg):</strong> {f1:.4f}<br>
#             <strong>AUC-ROC:</strong> {auc}
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
#         auc=auc_score if auc_score != 'N/A' else f"{auc_score:.4f}",
#         best_params=results['metrics']['Best Parameters']
#     ), unsafe_allow_html=True)
    
#     # Detailed Classification Report Section
#     st.subheader("📊 Detailed Classification Report")
    
#     # Convert classification report to DataFrame for better display
#     class_df = pd.DataFrame(class_report).transpose().round(4)
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
    
#     # ROC Curve for binary classification
#     if auc_score != 'N/A' and results['model'] and hasattr(results['model'], 'predict_proba'):
#         st.subheader("📊 ROC Curve")
#         y_test = results.get('y_test')
#         if y_test is not None:
#             y_pred_proba = results['model'].predict_proba(results['X_test'])[:, 1]
#             fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
#             roc_auc = auc(fpr, tpr)
            
#             fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
#             ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
#             ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#             ax_roc.set_xlim([0.0, 1.0])
#             ax_roc.set_ylim([0.0, 1.05])
#             ax_roc.set_xlabel('False Positive Rate')
#             ax_roc.set_ylabel('True Positive Rate')
#             ax_roc.set_title('Receiver Operating Characteristic (ROC) Curve')
#             ax_roc.legend(loc="lower right")
#             st.pyplot(fig_roc)
    
#     # Confusion Matrix Visualization
#     st.subheader("🎯 Confusion Matrix")
#     fig2, ax2 = plt.subplots(figsize=(8, 6))
    
#     sns.heatmap(results['metrics']['Confusion Matrix'], 
#                 annot=True, fmt='d', cmap='Blues', ax=ax2,
#                 xticklabels=True, yticklabels=True)
    
#     ax2.set_xlabel('Predicted Labels')
#     ax2.set_ylabel('True Labels')
#     ax2.set_title('Confusion Matrix')
#     st.pyplot(fig2)
    
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
        
#         # Safely get number of classes
#         num_classes = "N/A"
#         if 'target_encoder' in results and results['target_encoder'] is not None:
#             num_classes = len(results['target_encoder'].classes_)
#         elif 'Confusion Matrix' in results['metrics']:
#             num_classes = results['metrics']['Confusion Matrix'].shape[0]
        
#         st.write(f"- Classes: **{num_classes}**")
#         st.write(f"- AUC-ROC: **{auc_score if auc_score != 'N/A' else f'{auc_score:.4f}'}**")
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from typing import Dict, Any

# --- Helper to safely retrieve metrics and convert JSON-safe lists back to arrays ---

def _get_common_metrics(metrics_snapshot: Dict[str, Any]):
    """Extracts common metrics and converts the confusion matrix list back to array."""
    # Add null safety
    if not metrics_snapshot:
        return 0, 0, 0, 0, 'N/A', pd.DataFrame(), np.array([])
    
    class_report = metrics_snapshot.get('Classification Report', {})
    accuracy = metrics_snapshot.get('Accuracy', 0)
    auc_score = metrics_snapshot.get('AUC-ROC', 'N/A')
    
    precision_avg = class_report.get('macro avg', {}).get('precision', 0)
    recall_avg = class_report.get('macro avg', {}).get('recall', 0)
    f1_avg = class_report.get('macro avg', {}).get('f1-score', 0)
    
    # Convert the saved Python list back to a NumPy array for plotting
    conf_matrix_list = metrics_snapshot.get('Confusion Matrix', [])
    conf_matrix_array = np.array(conf_matrix_list) if conf_matrix_list else np.array([])
    
    class_df = pd.DataFrame(class_report).transpose().round(4) if class_report else pd.DataFrame()
    
    return accuracy, precision_avg, recall_avg, f1_avg, auc_score, class_df, conf_matrix_array

# --- Granular Report Asset Generation Functions ---

def create_lr_summary_text(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Generates the main summary text block (Report Asset Type: text)."""
    accuracy, precision_avg, recall_avg, f1_avg, auc_score, _, _ = _get_common_metrics(metrics_snapshot)

    # Safe AUC score handling
    try:
        auc_display = f"{float(auc_score):.4f}" if auc_score != 'N/A' else "N/A"
    except (ValueError, TypeError):
        auc_display = "N/A"
    
    text = """
    ### Logistic Regression Classifier Performance Summary
    - **Features:** {features}
    - **Target:** {target}
    - **Accuracy:** **{accuracy:.4f}**
    - **Precision (Macro):** {precision:.4f}
    - **Recall (Macro):** {recall:.4f}
    - **F1-Score (Macro):** {f1:.4f}
    - **AUC-ROC:** {auc}
    - **Best Parameters:** {best_params}
    - **Training Method:** {grid_search}
    """.format(
        features=", ".join(metrics_snapshot.get('features', [])),
        target=metrics_snapshot.get('target', 'Unknown'),
        accuracy=accuracy,
        precision=precision_avg,
        recall=recall_avg,
        f1=f1_avg,
        auc=auc_display,
        best_params=metrics_snapshot.get('Best Parameters', 'N/A'),
        grid_search="Grid Search" if metrics_snapshot.get('use_grid_search', False) else "Manual"
    )
    return {"type": "text", "content": text}


def create_confusion_matrix_plot(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Generates the Confusion Matrix plot (Report Asset Type: plot)."""
    _, _, _, _, _, _, conf_matrix_array = _get_common_metrics(metrics_snapshot)

    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Check if confusion matrix is empty
    if conf_matrix_array.size == 0:
        ax.text(0.5, 0.5, 'No confusion matrix data available', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Confusion Matrix (No Data)')
    else:
        sns.heatmap(conf_matrix_array, 
                    annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=True, yticklabels=True)
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        ax.set_title('Confusion Matrix')
    
    plt.tight_layout()
    plt.close(fig)
    
    return {"type": "plot", "content": fig}


def create_classification_report_table(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Generates the Classification Report table (Report Asset Type: dataframe)."""
    _, _, _, _, _, class_df, _ = _get_common_metrics(metrics_snapshot)
    return {"type": "dataframe", "content": class_df}


def create_performance_metrics_plot(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Generates the main Accuracy/Precision/Recall/F1 bar chart (Report Asset Type: plot)."""
    accuracy, precision_avg, recall_avg, f1_avg, _, _, _ = _get_common_metrics(metrics_snapshot)
    
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
    
    plt.tight_layout()
    plt.close(fig)
    
    return {"type": "plot", "content": fig}


def create_roc_curve_plot(metrics_snapshot: Dict[str, Any], model_results: Dict[str, Any] = None) -> Dict[str, Any]:
    """Generates ROC Curve plot for binary classification (Report Asset Type: plot)."""
    if model_results is None:
        model_results = {}
    
    auc_score = metrics_snapshot.get('AUC-ROC', 'N/A')
    fpr_snapshot = metrics_snapshot.get('fpr', [])
    tpr_snapshot = metrics_snapshot.get('tpr', [])

    # Check if we can generate ROC curve (Method 1: From Snapshot - Preferred)
    if auc_score != 'N/A' and fpr_snapshot and tpr_snapshot:
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(fpr_snapshot, tpr_snapshot, color='darkorange', lw=2, label=f'ROC curve (AUC = {float(auc_score):.2f})')
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('Receiver Operating Characteristic (ROC) Curve')
            ax.legend(loc="lower right")
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.close(fig)
            return {"type": "plot", "content": fig}
        except Exception as e:
            # Fall back to Method 2 if plot fail
            pass

    # Method 2: From live model (Legacy/Manual support)
    if (auc_score != 'N/A' and model_results.get('model') is not None and 
        hasattr(model_results['model'], 'predict_proba') and
        model_results.get('y_test') is not None and model_results.get('X_test') is not None):
        
        try:
            y_test = model_results['y_test']
            X_test = model_results['X_test']
            model = model_results['model']
            
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('Receiver Operating Characteristic (ROC) Curve')
            ax.legend(loc="lower right")
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.close(fig)
            
            return {"type": "plot", "content": fig}
            
        except Exception as e:
            # Return empty plot if ROC generation fails
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, f'ROC Curve not available\nError: {str(e)}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curve')
            plt.close(fig)
            return {"type": "plot", "content": fig}
    else:
        # Return empty plot if conditions not met
        fig, ax = plt.subplots(figsize=(8, 6))
        missing_conditions = []
        if auc_score == 'N/A':
            missing_conditions.append("AUC score not available")
        
        # If we reach here, it means we don't have snapshot data AND we don't have live model data
        if not fpr_snapshot or not tpr_snapshot:
            if model_results.get('model') is None:
                missing_conditions.append("Model not available")
            elif not hasattr(model_results.get('model'), 'predict_proba'):
                missing_conditions.append("Model doesn't support probabilities")
            if model_results.get('y_test') is None:
                missing_conditions.append("Test labels not available")
            if model_results.get('X_test') is None:
                missing_conditions.append("Test features not available")
        
        message = 'ROC Curve not available'
        if missing_conditions:
            message += f'\nMissing: {", ".join(missing_conditions)}'
            
        ax.text(0.5, 0.5, message, 
               ha='center', va='center', transform=ax.transAxes, fontsize=10)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        plt.close(fig)
        return {"type": "plot", "content": fig}

def create_roc_curve_plot_wrapper(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Wrapper function for reporting system that doesn't pass model_results."""
    return create_roc_curve_plot(metrics_snapshot, {})

def create_hyperparameters_table(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Generates hyperparameters display (Report Asset Type: text)."""
    best_params = metrics_snapshot.get('Best Parameters', {})
    
    if isinstance(best_params, dict) and best_params:
        params_text = "**Hyperparameters:**\n"
        for param, value in best_params.items():
            params_text += f"- **{param}:** {value}\n"
    else:
        params_text = f"**Best Parameters:** {best_params}\n"
    
    params_text += f"\n**Training Method:** {'Grid Search' if metrics_snapshot.get('use_grid_search', False) else 'Manual'}"
    
    return {"type": "text", "content": params_text}


def create_dataset_info_table(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Generates dataset information display (Report Asset Type: text)."""
    num_features = len(metrics_snapshot.get('features', []))
    target = metrics_snapshot.get('target', 'Unknown')
    auc_score = metrics_snapshot.get('AUC-ROC', 'N/A')
    
    # Determine number of classes from confusion matrix
    conf_matrix = np.array(metrics_snapshot.get('Confusion Matrix', []))
    num_classes = conf_matrix.shape[0] if conf_matrix.size > 0 else "Unknown"
    
    info_text = f"""
    **Dataset Information:**
    - Number of features: **{num_features}**
    - Target variable: **{target}**
    - Classes: **{num_classes}**
    - AUC-ROC: **{auc_score if auc_score != 'N/A' else 'N/A'}**
    """
    
    return {"type": "text", "content": info_text}


def create_performance_interpretation(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Generates performance interpretation text (Report Asset Type: text)."""
    accuracy, _, _, _, auc_score, _, _ = _get_common_metrics(metrics_snapshot)
    
    if accuracy >= 0.9:
        interpretation = "**🎉 Excellent Performance** - The model shows outstanding classification ability!"
        color = "green"
    elif accuracy >= 0.8:
        interpretation = "**👍 Good Performance** - The model performs well on the classification task."
        color = "blue"
    elif accuracy >= 0.7:
        interpretation = "**⚠️ Fair Performance** - The model has acceptable performance but could be improved."
        color = "orange"
    else:
        interpretation = "**❌ Poor Performance** - Consider feature engineering, parameter tuning, or trying a different algorithm."
        color = "red"
    
    if auc_score != 'N/A':
        try:
            auc_value = float(auc_score)
            if auc_value >= 0.9:
                auc_interpretation = " Outstanding discriminative power!"
            elif auc_value >= 0.8:
                auc_interpretation = " Good discriminative power."
            elif auc_value >= 0.7:
                auc_interpretation = " Fair discriminative power."
            else:
                auc_interpretation = " Poor discriminative power."
            interpretation += auc_interpretation
        except (ValueError, TypeError):
            pass  # Skip AUC interpretation if conversion fails
    
    return {"type": "text", "content": f'<span style="color:{color}">{interpretation}</span>'}


# --- ML Page Display Function ---

def display_lr_report(metrics_snapshot: Dict[str, Any], model_results: Dict[str, Any] = None):
    """
    Function to be called on the main ML page. It uses the granular functions
    to generate and display all components using Streamlit commands.
    """
    if not metrics_snapshot:
        st.error("No model data provided for display.")
        return

    # Ensure model_results is not None
    if model_results is None:
        model_results = {}

    st.markdown("---")
    st.markdown("## Logistic Regression Model Analysis")
    
    # Get all the report assets with error handling
    try:
        summary_asset = create_lr_summary_text(metrics_snapshot)
        table_asset = create_classification_report_table(metrics_snapshot)
        metrics_plot_asset = create_performance_metrics_plot(metrics_snapshot)
        conf_matrix_asset = create_confusion_matrix_plot(metrics_snapshot)
        roc_asset = create_roc_curve_plot_wrapper(metrics_snapshot)  # Use wrapper function
        params_asset = create_hyperparameters_table(metrics_snapshot)
        dataset_asset = create_dataset_info_table(metrics_snapshot)
        interpretation_asset = create_performance_interpretation(metrics_snapshot)
    except Exception as e:
        st.error(f"Error generating report assets: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return
        
    # Display Summary
    st.markdown(summary_asset['content'], unsafe_allow_html=True)
    
    # Performance Interpretation
    st.markdown("### 📊 Performance Interpretation")
    st.markdown(interpretation_asset['content'], unsafe_allow_html=True)
    
    # Display Metrics Plot
    st.markdown("### 📈 Key Metrics Visualization")
    try:
        st.pyplot(metrics_plot_asset['content'])
    except Exception as e:
        st.error(f"Error displaying metrics plot: {str(e)}")
    
    # Display ROC Curve
    st.markdown("### 📊 ROC Curve")
    try:
        st.pyplot(roc_asset['content'])
    except Exception as e:
        st.error(f"Error displaying ROC curve: {str(e)}")
    
    # Display Confusion Matrix
    st.markdown("### 🎯 Confusion Matrix")
    try:
        st.pyplot(conf_matrix_asset['content'])
    except Exception as e:
        st.error(f"Error displaying confusion matrix: {str(e)}")
    
    # Display Classification Table
    st.markdown("### 📋 Detailed Classification Report")
    try:
        if not table_asset['content'].empty:
            st.dataframe(table_asset['content'], use_container_width=True)
        else:
            st.info("No classification report data available.")
    except Exception as e:
        st.error(f"Error displaying classification report: {str(e)}")
    
    # Configuration Details
    st.markdown("### ⚙️ Model Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(params_asset['content'], unsafe_allow_html=True)
    
    with col2:
        st.markdown(dataset_asset['content'], unsafe_allow_html=True)


# --- Compatibility function for your existing system ---

def model_report():
    """Main function to maintain compatibility with your existing system."""
    # Try multiple ways to find the model data
    metrics_snapshot = None
    model_results = None
    model_found = False
    
    # Method 1: Look for Logistic Regression in pipeline ML
    for model_info in st.session_state.pipeline.get("ML", []):
        model_type = model_info.get("model type", "")
        # Check for various Logistic Regression identifiers
        if any(keyword in str(model_type) for keyword in ["Logistic Regression", "LR", "Logistic"]):
            model_found = True
            metrics_snapshot = model_info.get("metrics_snapshot", {})
            # Try to get the actual model if available
            if 'model' in model_info:
                try:
                    import joblib
                    from io import BytesIO
                    import base64
                    model_b64 = model_info['model']
                    model_bytes = base64.b64decode(model_b64)
                    buffer = BytesIO(model_bytes)
                    model_obj = joblib.load(buffer)
                    if model_results is None:
                        model_results = {}
                    model_results['model'] = model_obj
                except Exception as e:
                    st.warning(f"Could not load saved model: {e}")
            break
    
    # Method 2: Check session state for model_results
    if not model_found and 'model_results' in st.session_state:
        model_results_data = st.session_state.model_results
        if model_results_data and 'metrics' in model_results_data:
            model_found = True
            metrics_snapshot = model_results_data.get('metrics', {})
            model_results = model_results_data
    
    if not model_found:
        st.error("No Logistic Regression model results found. Please create a model first.")
        st.info("Available models in pipeline:")
        for i, model_info in enumerate(st.session_state.pipeline.get("ML", [])):
            st.write(f"- {i}: {model_info.get('model type', 'Unknown')} - {model_info.get('model name', 'No name')}")
        return
    
    if not metrics_snapshot:
        st.error("Found model but no metrics data available.")
        return
    
    # Ensure model_results has necessary structure for ROC curve
    if model_results is None:
        model_results = {
            'model': None,
            'X_test': None, 
            'y_test': None
        }
    else:
        if 'model' not in model_results:
            model_results['model'] = None
        if 'X_test' not in model_results:
            model_results['X_test'] = None
        if 'y_test' not in model_results:
            model_results['y_test'] = None
    
    # Validate metrics_snapshot has required fields
    required_fields = ['Classification Report', 'Accuracy', 'Confusion Matrix', 'Best Parameters']
    missing_fields = [field for field in required_fields if field not in metrics_snapshot]
    
    if missing_fields:
        st.warning(f"Missing some metrics fields: {missing_fields}")
        st.write("Available fields in metrics_snapshot:", list(metrics_snapshot.keys()))
        # Continue anyway, as some fields might be optional
    
    # Pass the metrics snapshot to the display function
    try:
        display_lr_report(metrics_snapshot, model_results)  # Ensure model_results is passed
    except Exception as e:
        st.error(f"Error displaying report: {str(e)}")
        import traceback
        st.code(traceback.format_exc())