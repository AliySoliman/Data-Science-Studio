import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree
from typing import Dict, Any

# --- Helper to safely retrieve metrics ---
def _get_common_metrics(metrics_snapshot: Dict[str, Any]):
    """Extracts common metrics from the snapshot."""
    try:
        accuracy = metrics_snapshot.get('Accuracy', 0)
        precision = metrics_snapshot.get('Precision', 0)
        recall = metrics_snapshot.get('Recall', 0)
        f1 = metrics_snapshot.get('F1 Score', 0)
        
        # Handle confusion matrix
        conf_matrix_data = metrics_snapshot.get('Confusion Matrix', [])
        if isinstance(conf_matrix_data, list) and len(conf_matrix_data) > 0:
            conf_matrix_array = np.array(conf_matrix_data)
        else:
            conf_matrix_array = np.array([])
            
        features = metrics_snapshot.get('features', [])
        target = metrics_snapshot.get('target', 'Unknown')
        
        return accuracy, precision, recall, f1, conf_matrix_array, features, target
    except Exception:
        return 0, 0, 0, 0, np.array([]), [], 'Unknown'

# --- Report Asset Generation Functions ---

def create_dt_summary_text(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Generates the main summary text block."""
    accuracy, precision, recall, f1, _, features, target = _get_common_metrics(metrics_snapshot)
    
    text = f"""
    ### Decision Tree Classifier Performance Summary
    - **Features:** {', '.join(features) if features else 'None'}
    - **Target:** {target}
    - **Accuracy:** **{accuracy:.4f}**
    - **Precision:** {precision:.4f}
    - **Recall:** {recall:.4f}
    - **F1-Score:** {f1:.4f}
    - **Training Method:** {"Grid Search" if metrics_snapshot.get('use_grid_search', False) else "Manual"}
    """
    return {"type": "text", "content": text}

def create_dt_performance_plot(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Generates the main metrics bar chart."""
    accuracy, precision, recall, f1, _, _, _ = _get_common_metrics(metrics_snapshot)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [accuracy, precision, recall, f1]
    
    bars = ax.bar(metrics, values, color=['#4CAF50', '#2196F3', '#FF9800', '#9C27B0'])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score')
    ax.set_title('Decision Tree Performance Metrics')
    
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.4f}', ha='center', va='bottom')
                
    plt.tight_layout()
    plt.close(fig)
    return {"type": "plot", "content": fig}

def create_dt_confusion_matrix_plot(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Generates the Confusion Matrix plot."""
    _, _, _, _, conf_matrix_array, _, _ = _get_common_metrics(metrics_snapshot)

    fig, ax = plt.subplots(figsize=(8, 6))
    if conf_matrix_array.size > 0 and conf_matrix_array.ndim == 2:
        num_classes = conf_matrix_array.shape[0]
        classes = [str(i) for i in range(num_classes)]
        sns.heatmap(conf_matrix_array, 
                    annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=classes, yticklabels=classes)
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        ax.set_title('Confusion Matrix - Decision Tree')
    else:
        ax.text(0.5, 0.5, 'No confusion matrix data available', 
                ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    plt.close(fig)
    return {"type": "plot", "content": fig}

def create_dt_tree_structure_plot(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Generates the Decision Tree structure visualization (plot_tree)."""
    # NOTE: This requires the actual model object, but we only have the snapshot.
    # However, for automated reporting, we might want to store more in the snapshot if needed.
    # For now, if we can't get the model, we display a message.
    
    # Check if a model exists in session_state for this snapshot
    # This is a bit of a hack since snapshots are meant to be JSON serializable
    model = None
    if 'model_results' in st.session_state and st.session_state.model_results.get('metrics') == metrics_snapshot:
        model = st.session_state.model_results.get('model')
    
    if model is None:
        return {"type": "text", "content": "*Tree structure plot is only available immediately after training.*"}

    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(
        model, 
        feature_names=metrics_snapshot.get('features', []),
        filled=True,
        rounded=True,
        ax=ax,
        fontsize=8
    )
    ax.set_title("Decision Tree Visualization")
    plt.tight_layout()
    plt.close(fig)
    
    return {"type": "plot", "content": fig}

def create_dt_feature_importance_plot(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Generates the Feature Importance visualization if available."""
    # We need feature importances in the snapshot. Let's make sure they are there.
    # If not, we might need to update model_script to include them.
    importances = metrics_snapshot.get('feature_importances', [])
    features = metrics_snapshot.get('features', [])
    
    if not importances or not features or len(importances) != len(features):
        return {"type": "text", "content": "Feature importance data not available in snapshot."}

    fig, ax = plt.subplots(figsize=(10, 6))
    indices = np.argsort(importances)[::-1]
    ax.barh(range(len(features)), np.array(importances)[indices])
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels([features[i] for i in indices])
    ax.set_xlabel('Importance Score')
    ax.set_title('Decision Tree Feature Importance')
    plt.tight_layout()
    plt.close(fig)
    
    return {"type": "plot", "content": fig}

def create_dt_classification_report_table(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Generates a simple classification metrics table."""
    accuracy, precision, recall, f1, _, _, _ = _get_common_metrics(metrics_snapshot)
    
    df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
        'Value': [accuracy, precision, recall, f1]
    })
    return {"type": "dataframe", "content": df}

def model_report():
    """Main function for manual report display (Legacy support)"""
    results = st.session_state.get('model_results')
    if not results:
        st.error("No model results found.")
        return
        
    metrics = results.get('metrics', {})
    
    # Use the asset functions to render
    summary = create_dt_summary_text(metrics)
    st.markdown(summary['content'])
    
    col1, col2 = st.columns(2)
    with col1:
        perf = create_dt_performance_plot(metrics)
        st.pyplot(perf['content'])
    with col2:
        cm = create_dt_confusion_matrix_plot(metrics)
        st.pyplot(cm['content'])
        
    st.subheader("Tree Structure")
    tree = create_dt_tree_structure_plot(metrics)
    if tree['type'] == 'plot':
        st.pyplot(tree['content'])
    else:
        st.info(tree['content'])