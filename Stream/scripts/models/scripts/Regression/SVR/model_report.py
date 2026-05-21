import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any

def _fmt(value, decimals=4):
    if value is None: return "N/A"
    try: return f"{float(value):.{decimals}f}"
    except: return "N/A"

def create_ml_summary_text(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    text = """
    ### Support Vector Regression Performance Summary
    - **R2 Score:** **{r2}**
    - **MAE:** {mae}
    - **MSE:** {mse}
    - **RMSE:** {rmse}
    - **Best Parameters:** {best_params}
    """.format(
        r2=_fmt(metrics_snapshot.get('R2 Score')),
        mae=_fmt(metrics_snapshot.get('MAE')),
        mse=_fmt(metrics_snapshot.get('MSE')),
        rmse=_fmt(metrics_snapshot.get('RMSE')),
        best_params=metrics_snapshot.get('Best Parameters', {})
    )
    return {"type": "text", "content": text}

def create_performance_metrics_plot(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    fig, ax = plt.subplots(figsize=(8, 5))
    r2 = metrics_snapshot.get('R2 Score', 0)
    ax.bar(['R2 Score'], [r2], color='#2196F3')
    ax.set_ylim(0, 1.1)
    ax.set_title('Model Performance: R2 Score')
    ax.text(0, r2 + 0.02, _fmt(r2), ha='center')
    plt.close(fig)
    return {"type": "plot", "content": fig}

def create_prediction_analysis_plot(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    try:
        y_test = np.array(metrics_snapshot.get('y_test', []))
        y_pred = np.array(metrics_snapshot.get('y_pred', []))
        if len(y_test) == 0: return {"type": "text", "content": "No prediction data available."}
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_test, y_pred, alpha=0.5, color='purple')
        lims = [min(min(y_test), min(y_pred)), max(max(y_test), max(y_pred))]
        ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0)
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title('Predicted vs Actual (SVR)')
        plt.close(fig)
        return {"type": "plot", "content": fig}
    except:
        return {"type": "text", "content": "Error generating prediction plot."}

def create_classification_report_table(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    return {"type": "text", "content": "Classification report is not applicable for Regression models."}

def create_error_metrics_plot(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    return {"type": "text", "content": "Error distribution plot not available for SVR currently."}

def model_report():
    for model_info in st.session_state.pipeline.get("ML", []):
        if model_info.get("model type") == "Support Vector Regression" or model_info.get("model name") == "Support Vector Regression":
            snapshot = model_info.get("metrics_snapshot", {})
            st.markdown(create_ml_summary_text(snapshot)['content'])
            st.pyplot(create_performance_metrics_plot(snapshot)['content'])
            st.pyplot(create_prediction_analysis_plot(snapshot)['content'])
            return
    st.error("No Support Vector Regression results found.")
