import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any

# ------------------------------------------------------
# Helper Function
# ------------------------------------------------------

def _get_common_metrics(metrics_snapshot: Dict[str, Any]):
    """Extract all metrics from the snapshot and rebuild the confusion matrix."""
    
    class_report = metrics_snapshot['Classification Report']
    accuracy = metrics_snapshot['Accuracy']
    
    precision_avg = class_report.get('macro avg', {}).get('precision', 0)
    recall_avg = class_report.get('macro avg', {}).get('recall', 0)
    f1_avg = class_report.get('macro avg', {}).get('f1-score', 0)
    
    conf_matrix_array = np.array(metrics_snapshot['Confusion Matrix'])
    
    class_df = pd.DataFrame(class_report).transpose().round(4)
    
    return accuracy, precision_avg, recall_avg, f1_avg, class_df, conf_matrix_array



# ------------------------------------------------------
# Asset Generators (Return dict tokens)
# ------------------------------------------------------

def create_summary_text(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Summary block text."""

    accuracy, precision_avg, recall_avg, f1_avg, _, _ = _get_common_metrics(metrics_snapshot)

    text = f"""
    ### KNN Classifier Performance Summary
    - **Features:** {", ".join(metrics_snapshot["features"])}
    - **Target:** {metrics_snapshot["target"]}
    - **Accuracy:** **{accuracy:.4f}**
    - **Precision (Macro):** {precision_avg:.4f}
    - **Recall (Macro):** {recall_avg:.4f}
    - **F1-Score (Macro):** {f1_avg:.4f}
    - **Best Parameters:** {metrics_snapshot["Best Parameters"]}
    - **Training Method:** {"Grid Search" if metrics_snapshot["use_grid_search"] else "Manual"}
    """

    return {
        "type": "text",
        "content": text
    }



def create_classification_report_table(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Classification report table."""
    
    _, _, _, _, class_df, _ = _get_common_metrics(metrics_snapshot)

    return {
        "type": "dataframe",
        "content": class_df
    }



def create_performance_metrics_plot(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Performance metrics bar chart."""
    
    accuracy, precision_avg, recall_avg, f1_avg, _, _ = _get_common_metrics(metrics_snapshot)

    fig, ax = plt.subplots(figsize=(10, 6))

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [accuracy, precision_avg, recall_avg, f1_avg]

    bars = ax.bar(metrics, values, color=['#4CAF50', '#2196F3', '#FF9800', '#9C27B0'])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Metrics')

    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height + 0.01,
            f'{value:.4f}',
            ha='center', va='bottom'
        )

    plt.close(fig)

    return {
        "type": "plot",
        "content": fig
    }



def create_confusion_matrix_plot(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Confusion matrix heatmap."""
    
    _, _, _, _, _, conf_matrix_array = _get_common_metrics(metrics_snapshot)

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        conf_matrix_array,
        annot=True,
        fmt='d',
        cmap='Blues',
        ax=ax,
        xticklabels=True,
        yticklabels=True
    )

    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.set_title('Confusion Matrix')

    plt.close(fig)

    return {
        "type": "plot",
        "content": fig
    }

def create_class_distribution_plot(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    _, _, _, _, _, conf_matrix_array = _get_common_metrics(metrics_snapshot)

    if 'target_encoder' in metrics_snapshot and metrics_snapshot['target_encoder']:
        class_names = metrics_snapshot['target_encoder'].classes_
    else:
        class_names = [f'Class {i}' for i in range(conf_matrix_array.shape[0])]

    actual_dist = conf_matrix_array.sum(axis=1)
    predicted_dist = conf_matrix_array.sum(axis=0)
    per_class_accuracy = np.diag(conf_matrix_array) / conf_matrix_array.sum(axis=1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    x_pos = np.arange(len(class_names))
    ax1.bar(x_pos - 0.2, actual_dist, 0.4, label='Actual', alpha=0.7)
    ax1.bar(x_pos + 0.2, predicted_dist, 0.4, label='Predicted', alpha=0.7)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(class_names, rotation=45)
    ax1.set_xlabel('Classes')
    ax1.set_ylabel('Count')
    ax1.set_title('Actual vs Predicted Class Distribution')
    ax1.legend()

    bars = ax2.bar(class_names, per_class_accuracy, color='skyblue', alpha=0.7)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Per-Class Accuracy')
    ax2.set_xticklabels(class_names, rotation=45)
    for bar, acc in zip(bars, per_class_accuracy):
        ax2.text(bar.get_x() + bar.get_width() / 2., acc + 0.01, f'{acc:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.close(fig)
    return {"type": "plot", "content": fig}

# ------------------------------------------------------
# Main Display Function
# ------------------------------------------------------

def display_ml_report(metrics_snapshot: Dict[str, Any]):
    """Builds the full ML report in Streamlit."""

    if not metrics_snapshot:
        st.error("No model data provided.")
        return

    st.markdown("---")
    st.markdown("## KNN Model Analysis (Live View)")

    summary_asset = create_summary_text(metrics_snapshot)
    table_asset = create_classification_report_table(metrics_snapshot)
    metrics_plot_asset = create_performance_metrics_plot(metrics_snapshot)
    conf_matrix_asset = create_confusion_matrix_plot(metrics_snapshot)
    class_dist_asset = create_class_distribution_plot(metrics_snapshot)
    st.markdown(summary_asset["content"], unsafe_allow_html=True)

    st.subheader("📈 Key Metrics Visualization")
    st.pyplot(metrics_plot_asset["content"])

    st.subheader("🎯 Confusion Matrix")
    st.pyplot(conf_matrix_asset["content"])

    st.subheader("📊 Detailed Classification Report")
    st.dataframe(table_asset["content"], use_container_width=True)
    
    st.subheader("📉 Class Distribution Analysis")
    st.pyplot(class_dist_asset['content'][0])
    st.pyplot(class_dist_asset['content'][1])

    # Hyperparameters and Algorithm Details
    st.subheader("🔧 Model-Specific Information")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"- Model Type: **{metrics_snapshot['model_type'].upper()} Naive Bayes**")
        st.write(f"- Number of features: **{len(metrics_snapshot['features'])}**")
        st.write(f"- Target variable: **{metrics_snapshot['target']}**")
        st.write(f"- Grid Search: **{'Yes' if metrics_snapshot['use_grid_search'] else 'No'}**")
        if metrics_snapshot['use_grid_search']:
            st.write(f"- CV Folds: **{metrics_snapshot['cv_folds']}**")

        if 'target_encoder' in metrics_snapshot and metrics_snapshot['target_encoder']:
            num_classes = len(metrics_snapshot['target_encoder'].classes_)
        else:
            num_classes = metrics_snapshot['metrics']['Confusion Matrix'].shape[0]
        st.write(f"- Number of classes: **{num_classes}**")

    with col2:
        st.write("**Hyperparameters:**")
        best_params = metrics_snapshot['metrics']['Best Parameters']
        if isinstance(best_params, dict):
            for param, value in best_params.items():
                st.write(f"- **{param}:** {value}")
        else:
            st.write(f"- {best_params}")

        st.write("**Algorithm Characteristics:**")
        model_type = metrics_snapshot['model_type']
        if model_type == 'gaussian':
            st.write("- Assumes features follow Gaussian distribution")
        elif model_type == 'multinomial':
            st.write("- Suitable for discrete count data")
        else:
            st.write("- Suitable for binary/boolean features")

# ------------------------------------------------------
# Wrapper Function (same name as original)
# ------------------------------------------------------

def model_report():
    """Load metrics_snapshot from session_state and display Naive Bayes report."""
    found = False
    for model_info in st.session_state.pipeline.get("ML", []):
        if model_info.get("model name") == 'Naive Bayes':
            if "metrics_snapshot" in model_info:
                found = True
                metrics_snapshot = model_info["metrics_snapshot"]
                break

    if not found:
        st.error("No model results found. Please create a model first.")
        return

    display_ml_report(metrics_snapshot)
