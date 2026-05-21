import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from typing import Dict, Any

def display_ml_report(metrics_snapshot: Dict[str, Any]):
    ml_summary_asset = create_ml_summary_text(metrics_snapshot)
    feature_importance_asset = create_feature_importance_plot(metrics_snapshot)
    prediction_analysis_asset = create_prediction_analysis_plot(metrics_snapshot)
    residual_analysis_asset = create_residual_analysis_plot(metrics_snapshot)
    get_model_configuration_insights = model_configuration_insights(metrics_snapshot)
    model_recommendations = create_model_recommendations(metrics_snapshot)
    
    
    st.markdown(ml_summary_asset["content"], unsafe_allow_html=True)
    

    
    st.pyplot(feature_importance_asset['content'])

    

    st.pyplot(prediction_analysis_asset['content'])
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("R² Score", f"{metrics_snapshot['R2 Score']:.4f}")
    with col2:
        st.metric("Mean Squared Error", f"{metrics_snapshot['MSE']:.4f}")
    with col3:
        st.metric("Mean Absolute Error", f"{metrics_snapshot['MAE']:.4f}")
    st.pyplot(residual_analysis_asset['content'])

    
    
 # Placeholder return


    


def model_report():
    found = False 
    # Check for the updated model name
    for model_info in st.session_state.pipeline.get("ML", []):
        if model_info.get("model name") == 'Random Forest Regression':
            if 'metrics_snapshot' in model_info:
                found = True
                model_results = model_info.get("metrics_snapshot", {})
                break
    


    if not found:
        st.error("No Random Forest Regressor model results found. Please create a model first.")
        # return
    
    display_ml_report(model_results)


def get_common_params_metrics(metrics_snapshot: Dict[str, Any]):
    """
    Extracts and returns common regression metrics from the provided metrics snapshot.
    """
    if not metrics_snapshot:
        return {}

    features = metrics_snapshot.get('features', [])
    target = metrics_snapshot.get('target', '')
    n_trees = metrics_snapshot.get('Number of Trees', 0)
    tuning_method = "Grid Search" if metrics_snapshot.get('use_grid_search', False) else "Manual Parameters"
    params = metrics_snapshot.get('Best Parameters', {})
    r2 = metrics_snapshot.get('R2 Score', 0)
    mae = metrics_snapshot.get('MAE', 0)    
    mse = metrics_snapshot.get('MSE', 0)
    class_df = pd.DataFrame() 
    conf_matrix_array = np.array([]) 
    
    return r2, mae, mse, class_df, conf_matrix_array, features, target, n_trees, tuning_method, params

def create_ml_summary_text(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Generates the main summary text block (Report Asset Type: text)."""
    # Use R2, MAE, MSE
    r2, mae, mse, _, _, features, target, n_trees, tuning_method, params = get_common_params_metrics(metrics_snapshot)

    text = """
    ### Random Forest Regressor Performance Summary
    - **Features:** {features}
    - **Target:** {target}
    - **Number of Trees:** {n_trees}
    - **R2 Score:** **{r2:.4f}**
    - **Mean Absolute Error (MAE):** {mae:.4f}
    - **Mean Squared Error (MSE):** {mse:.4f}
    - **Best Parameters:** {best_params}
    - **Training Method:** {grid_search}
    """.format(
        features=", ".join(features),
        target=target,
        n_trees=n_trees,
        r2=r2,
        mae=mae,
        mse=mse,
        best_params=params,
        grid_search=tuning_method
    )
    # The returned dictionary is the Report Asset Token (must be returned, not st.written)
    return {"type": "text", "content": text}



def create_feature_importance_plot(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Render feature importance bar chart from stored snapshot."""
    feature_importance = metrics_snapshot.get('Feature Importance') or metrics_snapshot.get('feature_importance')

    if not feature_importance:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, 'Feature importance data not available', ha='center', va='center', transform=ax.transAxes)
        plt.close(fig)
        return {"type": "plot", "content": fig}

    features = list(feature_importance.keys())
    importance_values = list(feature_importance.values())

    sorted_idx = np.argsort(importance_values)[::-1]
    sorted_features = [features[i] for i in sorted_idx]
    sorted_importance = [importance_values[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(sorted_features))
    bars = ax.barh(y_pos, sorted_importance, align='center', color='skyblue', alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_features)
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance Score')
    ax.set_title('Random Forest Feature Importance')
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.001, bar.get_y() + bar.get_height()/2,
               f'{width:.3f}', ha='left', va='center', fontsize=9)
    plt.tight_layout()
    plt.close(fig)
    return {"type": "plot", "content": fig}


def create_prediction_analysis_plot(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Generates Prediction vs Actual plots using stored y_test/y_pred arrays."""
    try:
        y_test = metrics_snapshot.get('y_test')
        y_pred = metrics_snapshot.get('y_pred')

        if y_test is None or y_pred is None:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.text(0.5, 0.5, 'Prediction data not available in snapshot', ha='center', va='center', transform=ax.transAxes)
            plt.close(fig)
            return {"type": "plot", "content": fig}

        y_test = np.array(y_test)
        y_pred = np.array(y_pred)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        ax1.scatter(y_test, y_pred, alpha=0.6, color='blue')
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        ax1.set_xlabel('Actual Values')
        ax1.set_ylabel('Predicted Values')
        ax1.set_title('Predicted vs Actual Values')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2.hist(y_test, alpha=0.7, label='Actual', bins=20, color='blue')
        ax2.hist(y_pred, alpha=0.7, label='Predicted', bins=20, color='red')
        ax2.set_xlabel('Value')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution: Actual vs Predicted')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.close(fig)
        return {"type": "plot", "content": fig}

    except Exception as e:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, f'Prediction plot error: {str(e)}', ha='center', va='center', transform=ax.transAxes, wrap=True)
        plt.close(fig)
        return {"type": "plot", "content": fig}


def create_residual_analysis_plot(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Generates Residual Analysis plots using stored y_test/y_pred arrays."""
    try:
        y_test = metrics_snapshot.get('y_test')
        y_pred = metrics_snapshot.get('y_pred')

        if y_test is None or y_pred is None:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.text(0.5, 0.5, 'Residual data not available in snapshot', ha='center', va='center', transform=ax.transAxes)
            plt.close(fig)
            return {"type": "plot", "content": fig}

        y_test = np.array(y_test)
        y_pred = np.array(y_pred)
        residuals = y_test - y_pred

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        ax1.scatter(y_pred, residuals, alpha=0.6, color='green')
        ax1.axhline(y=0, color='red', linestyle='--')
        ax1.set_xlabel('Predicted Values')
        ax1.set_ylabel('Residuals')
        ax1.set_title('Residuals vs Predicted Values')
        ax1.grid(True, alpha=0.3)

        ax2.hist(residuals, bins=20, alpha=0.7, color='green', edgecolor='black')
        ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax2.set_xlabel('Residual Value')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Residuals Distribution')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.close(fig)
        return {"type": "plot", "content": fig}

    except Exception as e:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, f'Residual plot error: {str(e)}', ha='center', va='center', transform=ax.transAxes, wrap=True)
        plt.close(fig)
        return {"type": "plot", "content": fig}


def model_configuration_insights(metrics_snapshot: Dict[str, Any]):
    """Returns model config as a text asset dict (does not call st directly)."""
    results = metrics_snapshot
    best_params = results.get('Best Parameters', {})
    r2 = results.get('R2 Score', 0)
    mse = results.get('MSE', 0)
    mae = results.get('MAE', 0)

    params_rows = "\n".join([f"| {k} | {v} |" for k, v in best_params.items()]) if isinstance(best_params, dict) else ""
    interp = 'Variance explained' if r2 > 0.7 else ('Moderate fit' if r2 > 0.5 else 'Poor fit')

    text = (
        f"**Model Configuration & Insights**\n\n"
        f"**Model Parameters:**\n\n"
        f"| Parameter | Value |\n|---|---|\n{params_rows}\n\n"
        f"**Performance Summary:**\n\n"
        f"| Metric | Value | Interpretation |\n|---|---|---|\n"
        f"| R² Score | {r2:.4f} | {interp} |\n"
        f"| Mean Squared Error | {mse:.4f} | Lower is better |\n"
        f"| Mean Absolute Error | {mae:.4f} | Lower is better |\n"
    )
    return {"type": "text", "content": text}


def create_model_recommendations(metrics_snapshot: Dict[str, Any]):
    """Returns model recommendations as a text asset dict."""
    r2_score = metrics_snapshot.get('R2 Score', 0)
    if r2_score < 0.5:
        content = (
            "**Model Performance Needs Improvement (R² < 0.5)**\n\nConsider:\n"
            "- Adding more relevant features\n"
            "- Trying different model parameters\n"
            "- Checking for data quality issues\n"
            "- Using feature engineering\n"
            "- Trying different algorithms"
        )
    elif r2_score < 0.8:
        content = (
            "**Model Performance is Good (R² 0.5-0.8)**\n\nSuggestions:\n"
            "- Fine-tune hyperparameters further\n"
            "- Try feature selection to remove noise\n"
            "- Consider ensemble methods\n"
            "- Check for non-linear relationships"
        )
    else:
        content = (
            "**Model Performance is Excellent (R² > 0.8)**\n\n"
            "The model explains most of the variance in the target variable.\n\nConsider:\n"
            "- Testing on new data to validate performance\n"
            "- Monitoring for concept drift over time\n"
            "- Exploring model interpretability techniques"
        )
    return {"type": "text", "content": content}
