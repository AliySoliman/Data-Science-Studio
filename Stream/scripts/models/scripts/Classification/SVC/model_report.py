import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any


# -------------------------
# Helper to extract common SVC metrics
# -------------------------
def _get_svc_common_metrics(metrics_snapshot: Dict[str, Any]):
    """
    Extracts key SVM metrics from the snapshot and returns them.
    Preserves original SVM logic (classification report macro averages, accuracy, confusion matrix).
    """
    # Handle nested metrics dictionary if present
    data = metrics_snapshot.get('metrics', metrics_snapshot)
    
    # classification report in the original snippet is stored under metrics -> 'Classification Report'
    class_report = data['Classification Report']
    accuracy = data['Accuracy']

    precision_avg = class_report.get('macro avg', {}).get('precision', 0)
    recall_avg = class_report.get('macro avg', {}).get('recall', 0)
    f1_avg = class_report.get('macro avg', {}).get('f1-score', 0)

    # confusion matrix (keep as numpy array for plotting)
    conf_matrix_array = np.array(data['Confusion Matrix'])
    # DataFrame for the classification report
    class_df = pd.DataFrame(class_report).transpose().round(4)

    return {
        "class_report": class_report,
        "class_df": class_df,
        "accuracy": accuracy,
        "precision_avg": precision_avg,
        "recall_avg": recall_avg,
        "f1_avg": f1_avg,
        "conf_matrix": conf_matrix_array
    }

# -------------------------
# Asset generator functions (return dict tokens)
# -------------------------
def create_svc_summary_text(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Generates the main summary HTML/text block for SVM."""
    m = _get_svc_common_metrics(metrics_snapshot)

    text = f"""
    ### Support Vector Classifier Performance Summary
    - **Features:** {", ".join(metrics_snapshot["features"])}
    - **Target:** {metrics_snapshot["target"]}
    - **Accuracy:** **{m['accuracy']:.4f}**
    - **Precision (Macro):** {m['precision_avg']:.4f}
    - **Recall (Macro):** {m['recall_avg']:.4f}
    - **F1-Score (Macro):** {m['f1_avg']:.4f}
    - **Best Parameters:** {metrics_snapshot["Best Parameters"]}
    - **Training Method:** {"Grid Search" if metrics_snapshot["use_grid_search"] else "Manual"}
    """

    return {"type": "text","content": text}


def create_svc_classification_report_table(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Creates the classification report dataframe token."""
    m = _get_svc_common_metrics(metrics_snapshot)
    
    return {"type": "dataframe", "content": m['class_df']}


def create_svc_metrics_plot(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Creates bar chart for Accuracy/Precision/Recall/F1 (macro averages)."""
    m = _get_svc_common_metrics(metrics_snapshot)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [m['accuracy'], m['precision_avg'], m['recall_avg'], m['f1_avg']]
    bars = ax.bar(metrics, values, color=['#4CAF50', '#2196F3', '#FF9800', '#9C27B0'])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Metrics')

    for bar, value in zip(bars, values):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., h + 0.01, f'{value:.4f}', ha='center', va='bottom')

    plt.close(fig)
    return {"type": "plot", "content": fig}


def create_svc_confusion_matrix_plot(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Generates the confusion matrix heatmap token."""
    m = _get_svc_common_metrics(metrics_snapshot)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(m['conf_matrix'],
                annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=True, yticklabels=True)
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.set_title('Confusion Matrix')
    plt.close(fig)
    return {"type": "plot", "content": fig}


def create_svc_per_class_plot(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """
    Creates per-class bar charts for precision/recall/f1 if available in classification report.
    Returns a composite token that contains the figure and the per-class dataframe.
    """
    metrics = _get_svc_common_metrics(metrics_snapshot)
    class_df = metrics["class_df"]

    # Filter out average rows for per-class display like original code
    per_class_df = class_df[~class_df.index.isin(['accuracy', 'macro avg', 'weighted avg'])]

    # Create composite: if per-class rows exist, create bar plot of precision/recall/f1
    if not per_class_df.empty and {'precision', 'recall', 'f1-score'}.issubset(set(per_class_df.columns)):
        fig, ax = plt.subplots(figsize=(12, 6))
        per_class_df[['precision', 'recall', 'f1-score']].plot(kind='bar', ax=ax)
        ax.set_title('Precision, Recall, and F1-Score by Class')
        ax.set_ylabel('Score')
        ax.set_xlabel('Class')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        plt.close(fig)
        return {"type": "composite", "content": {"fig": fig, "per_class_df": per_class_df}}
    else:
        # Empty placeholder if per-class details are not provided
        empty_df = pd.DataFrame()
        return {"type": "composite", "content": {"fig": None, "per_class_df": empty_df}}


def create_svc_model_config_block(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Creates a text block with model configuration & dataset info."""
    # Best params may include kernel; follow original snippet
    best_params = metrics_snapshot.get('Best Parameters', {})

    # Determine number of classes safely
    num_classes = "N/A"
    if 'target_encoder' in metrics_snapshot and metrics_snapshot['target_encoder'] is not None:
        num_classes = len(metrics_snapshot['target_encoder'].classes_)
    elif 'metrics' in metrics_snapshot and 'Confusion Matrix' in metrics_snapshot['metrics']:
        num_classes = metrics_snapshot['metrics']['Confusion Matrix'].shape[0]

    text = (
        f"**Hyperparameters:**\n\n"
        f"- Best Parameters: **{best_params}**\n\n"
        f"**Training Method:**\n\n"
        f"- Grid Search: {'✅ Yes' if metrics_snapshot.get('use_grid_search') else '❌ No'}\n"
    )
    if metrics_snapshot.get('use_grid_search'):
        text += f"- CV Folds: **{metrics_snapshot.get('cv_folds', 'N/A')}**\n"

    text += (
        f"\n**Dataset Information:**\n\n"
        f"- Number of features: **{len(metrics_snapshot.get('features', []))}**\n"
        f"- Target variable: **{metrics_snapshot.get('target', '')}**\n"
        f"- Classes: **{num_classes}**\n"
        f"- Kernel: **{best_params.get('kernel', 'N/A')}**"
    )

    return {"type": "text", "content": text}


def create_svc_performance_interpretation(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Creates performance interpretation text based on accuracy thresholds (same messages as original)."""
    metrics = _get_svc_common_metrics(metrics_snapshot)
    accuracy = metrics["accuracy"]

    if accuracy >= 0.9:
        content = "**Excellent Performance** - The model shows outstanding classification ability!"
    elif accuracy >= 0.8:
        content = "**Good Performance** - The model performs well on the classification task."
    elif accuracy >= 0.7:
        content = "**Fair Performance** - The model has acceptable performance but could be improved."
    else:
        content = "**Poor Performance** - Consider feature engineering, parameter tuning, or trying a different algorithm."

    return {"type": "text", "content": content}


# -------------------------
# Main display function (composes assets and renders)
# -------------------------
def display_svc_report(metrics_snapshot: Dict[str, Any]):
    """
    Compose SVC report by generating asset tokens and rendering them with Streamlit.
    """
    if not metrics_snapshot:
        st.error("No model data provided for display.")
        return

    st.markdown("---")
    st.markdown("## Support Vector Machine Model Analysis (Live View)")

    # Generate assets
    summary_asset = create_svc_summary_text(metrics_snapshot)
    class_report_asset = create_svc_classification_report_table(metrics_snapshot)
    metrics_plot_asset = create_svc_metrics_plot(metrics_snapshot)
    conf_matrix_asset = create_svc_confusion_matrix_plot(metrics_snapshot)
    per_class_asset = create_svc_per_class_plot(metrics_snapshot)
    model_config_asset = create_svc_model_config_block(metrics_snapshot)
    interpretation_asset = create_svc_performance_interpretation(metrics_snapshot)
    
    # Display Summary (HTML)
    st.markdown(summary_asset['content'], unsafe_allow_html=True)

    # Detailed classification report table
    st.subheader("📊 Detailed Classification Report")
    st.dataframe(class_report_asset['content'], use_container_width=True)

    # Metrics Plot
    st.subheader("📈 Metrics Visualization")
    st.pyplot(metrics_plot_asset['content'])
    
    # Confusion Matrix
    st.subheader("🎯 Confusion Matrix")
    st.pyplot(conf_matrix_asset['content'])

    # Per-Class Breakdown (text + plot if available)
    st.subheader("📋 Per-Class Metrics Breakdown")
    comp = per_class_asset['content']
    per_class_df = comp.get('per_class_df', pd.DataFrame())
    if not per_class_df.empty:
        # display lists of precision & recall by class (left/right columns) similar to original
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Precision by Class:**")
            for class_name, row in per_class_df.iterrows():
                st.write(f"- Class {class_name}: {row.get('precision', 0):.4f}")
        with col2:
            st.write("**Recall by Class:**")
            for class_name, row in per_class_df.iterrows():
                st.write(f"- Class {class_name}: {row.get('recall', 0):.4f}")
        # plot the bar chart
        if comp.get('fig') is not None:
            st.pyplot(comp['fig'])

    # Model Configuration Block
    st.subheader("⚙️ Model Configuration")
    st.markdown(model_config_asset['content'])

    # Performance Interpretation
    st.subheader("📊 Performance Interpretation")
    st.markdown(interpretation_asset['content'])

# -------------------------
# Wrapper function (keeps name model_report)
# -------------------------
def model_report():
    """
    Wrapper that loads SVC metrics_snapshot from session_state and calls display_svc_report.
    Compatible with both:
    - st.session_state.model_results (original SVM snippet)
    - st.session_state.pipeline["ML"] (KNN/Naive Bayes pipeline style)
    """
    # Priority 1: direct model_results (original snippet)
    if 'model_results' in st.session_state:
        results = st.session_state.model_results
        if 'metrics' not in results or 'Accuracy' not in results['metrics']:
            st.error("Invalid model results format for Support Vector Machine classifier")
            return
        display_svc_report(results)
        return

    # Priority 2: pipeline-style storage (compat with other modular reports)
    found = False
    pipeline = st.session_state.get("pipeline", {})
    for model_info in pipeline.get("ML", []):
        # match common names for SVM
        if model_info.get("model name")== 'Support Vector Classifier':
            if 'metrics_snapshot' in model_info:
                found = True
                metrics_snapshot = model_info.get("metrics_snapshot", {})
                break

    if found:
        display_svc_report(metrics_snapshot)
        return

    st.error("No model results found. Please create a model first.")
