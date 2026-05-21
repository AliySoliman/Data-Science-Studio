import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any


def _fmt(value, decimals=4):
    """Format a metric value safely — returns 'N/A' if None or not numeric."""
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return "N/A"


def _get_common_metrics(metrics_snapshot: Dict[str, Any]):
    try:
        accuracy = metrics_snapshot.get('Accuracy', 0) or 0
        class_report = metrics_snapshot.get('Classification Report', {})

        precision_avg = recall_avg = f1_avg = 0
        if isinstance(class_report, dict):
            if 'macro avg' in class_report and isinstance(class_report['macro avg'], dict):
                macro_avg = class_report['macro avg']
                precision_avg = macro_avg.get('precision', 0) or 0
                recall_avg = macro_avg.get('recall', 0) or 0
                f1_avg = macro_avg.get('f1-score', 0) or 0
            elif 'weighted avg' in class_report and isinstance(class_report['weighted avg'], dict):
                weighted_avg = class_report['weighted avg']
                precision_avg = weighted_avg.get('precision', 0) or 0
                recall_avg = weighted_avg.get('recall', 0) or 0
                f1_avg = weighted_avg.get('f1-score', 0) or 0
            elif '1' in class_report and isinstance(class_report['1'], dict):
                class_1 = class_report['1']
                precision_avg = class_1.get('precision', 0) or 0
                recall_avg = class_1.get('recall', 0) or 0
                f1_avg = class_1.get('f1-score', 0) or 0

        conf_matrix_data = metrics_snapshot.get('Confusion Matrix', [])
        if isinstance(conf_matrix_data, list) and len(conf_matrix_data) > 0:
            conf_matrix_array = np.array(conf_matrix_data)
        else:
            conf_matrix_array = np.array([])

        feature_importances = metrics_snapshot.get('feature_importances', [])
        features = metrics_snapshot.get('features', [])

        class_df = pd.DataFrame()
        if isinstance(class_report, dict) and class_report:
            report_data = {}
            for key, value in class_report.items():
                if isinstance(value, dict):
                    report_data[key] = value
            if report_data:
                class_df = pd.DataFrame(report_data).transpose().round(4)

        return accuracy, precision_avg, recall_avg, f1_avg, class_df, conf_matrix_array, feature_importances, features

    except Exception:
        return 0, 0, 0, 0, pd.DataFrame(), np.array([]), [], []


def create_gbc_summary_text(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    accuracy, precision_avg, recall_avg, f1_avg, _, _, _, features = _get_common_metrics(metrics_snapshot)

    best_params = metrics_snapshot.get('Best Parameters', {})
    if isinstance(best_params, dict):
        params_str = ", ".join([f"{k}: {v}" for k, v in best_params.items()])
    else:
        params_str = str(best_params)

    text = f"""
    ### Gradient Boosting Classifier Performance Summary
    - **Features:** {', '.join(features) if features else 'None'}
    - **Target:** {metrics_snapshot.get('target', 'Unknown')}
    - **Accuracy:** **{_fmt(accuracy)}**
    - **Precision (Macro):** {_fmt(precision_avg)}
    - **Recall (Macro):** {_fmt(recall_avg)}
    - **F1-Score (Macro):** {_fmt(f1_avg)}
    - **Best Parameters:** {params_str}
    - **Training Method:** {"Grid Search" if metrics_snapshot.get('use_grid_search', False) else "Manual"}
    """

    return {"type": "text", "content": text}


def create_gbc_confusion_matrix_plot(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    _, _, _, _, _, conf_matrix_array, _, _ = _get_common_metrics(metrics_snapshot)

    fig, ax = plt.subplots(figsize=(8, 6))

    if conf_matrix_array.size > 0 and conf_matrix_array.ndim == 2:
        sns.heatmap(conf_matrix_array,
                    annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=True, yticklabels=True)
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        ax.set_title('Confusion Matrix - Gradient Boosting Classifier')
    else:
        ax.text(0.5, 0.5, 'No confusion matrix data available',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Confusion Matrix - Gradient Boosting Classifier')

    plt.tight_layout()
    plt.close(fig)

    return {"type": "plot", "content": fig}


def create_gbc_classification_report_table(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    _, _, _, _, class_df, _, _, _ = _get_common_metrics(metrics_snapshot)
    return {"type": "dataframe", "content": class_df}


def create_gbc_performance_plot(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    accuracy, precision_avg, recall_avg, f1_avg, _, _, _, _ = _get_common_metrics(metrics_snapshot)

    fig, ax = plt.subplots(figsize=(10, 6))
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [accuracy, precision_avg, recall_avg, f1_avg]

    bars = ax.bar(metrics, values, color=['#4CAF50', '#2196F3', '#FF9800', '#9C27B0'])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score')
    ax.set_title('Gradient Boosting Classifier Performance Metrics')

    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                _fmt(value), ha='center', va='bottom')

    plt.tight_layout()
    plt.close(fig)

    return {"type": "plot", "content": fig}


def create_gbc_feature_importance_plot(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    _, _, _, _, _, _, feature_importances, features = _get_common_metrics(metrics_snapshot)

    fig, ax = plt.subplots(figsize=(10, 6))

    if (feature_importances and features and
            len(feature_importances) == len(features)):

        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': feature_importances
        }).sort_values('Importance', ascending=True)

        y_pos = np.arange(len(importance_df))
        bars = ax.barh(y_pos, importance_df['Importance'], color='lightgreen', edgecolor='darkgreen')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(importance_df['Feature'])
        ax.set_xlabel('Importance Score')
        ax.set_title('Gradient Boosting Classifier - Feature Importance Ranking')

        for i, (bar, importance) in enumerate(zip(bars, importance_df['Importance'])):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height() / 2.,
                    _fmt(importance), ha='left', va='center', fontsize=8)
    else:
        ax.text(0.5, 0.5, 'No feature importance data available',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Gradient Boosting Classifier - Feature Importance')

    plt.tight_layout()
    plt.close(fig)

    return {"type": "plot", "content": fig}


def model_report():
    found = False
    model_results = None

    if hasattr(st.session_state, 'pipeline') and "ML" in st.session_state.pipeline:
        for model_info in st.session_state.pipeline["ML"]:
            model_name = (model_info.get('model name') or
                          model_info.get('name') or
                          model_info.get('model_type') or
                          model_info.get('type', ''))

            if model_name == 'Gradient Boosting Classifier':
                if 'metrics_snapshot' in model_info:
                    model_results = model_info['metrics_snapshot']
                    found = True
                    break

    if not found:
        st.error("No Gradient Boosting Classifier model results found. Please train a model first.")
        if hasattr(st.session_state, 'pipeline') and "ML" in st.session_state.pipeline:
            saved_models = [(m.get('model name') or m.get('name') or m.get('model_type')) for m in st.session_state.pipeline["ML"]]
            st.info(f"Available models in pipeline: {saved_models}")
        return

    st.markdown("---")
    st.markdown("## Gradient Boosting Classifier Model Analysis")

    try:
        summary_asset = create_gbc_summary_text(model_results)
        table_asset = create_gbc_classification_report_table(model_results)
        metrics_plot_asset = create_gbc_performance_plot(model_results)
        conf_matrix_asset = create_gbc_confusion_matrix_plot(model_results)
        feature_importance_asset = create_gbc_feature_importance_plot(model_results)

        st.markdown(summary_asset['content'])

        st.subheader("📈 Key Metrics Visualization")
        st.pyplot(metrics_plot_asset['content'])

        st.subheader("🔍 Feature Importance")
        st.pyplot(feature_importance_asset['content'])

        st.subheader("🎯 Confusion Matrix")
        st.pyplot(conf_matrix_asset['content'])

        st.subheader("📊 Detailed Classification Report")
        if not table_asset['content'].empty:
            st.dataframe(table_asset['content'], use_container_width=True)
        else:
            st.info("Classification report not available")

    except Exception as e:
        st.error(f"Error displaying Gradient Boosting Classifier model report: {e}")


def display_compact_metrics():
    """Compact metrics card for pipeline summary view — mirrors GBR compact display."""
    found = False
    model_results = {}

    if hasattr(st.session_state, 'pipeline') and "ML" in st.session_state.pipeline:
        for model_info in st.session_state.pipeline["ML"]:
            model_name = (model_info.get('model name') or
                          model_info.get('name') or
                          model_info.get('model_type') or
                          model_info.get('type', ''))
            if model_name == 'Gradient Boosting Classifier':
                if 'metrics_snapshot' in model_info:
                    model_results = model_info['metrics_snapshot']
                    found = True
                    break

    if not found:
        return

    accuracy = model_results.get('Accuracy', 0) or 0
    class_report = model_results.get('Classification Report', {})
    macro = class_report.get('macro avg', {}) if isinstance(class_report, dict) else {}

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", _fmt(accuracy))
    with col2:
        st.metric("Precision", _fmt(macro.get('precision', 0)))
    with col3:
        st.metric("Recall", _fmt(macro.get('recall', 0)))
    with col4:
        st.metric("F1 Score", _fmt(macro.get('f1-score', 0)))
