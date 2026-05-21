import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any

def _get_common_metrics(metrics_snapshot: Dict[str, Any]):
    """Extracts common regression metrics."""
    r2 = metrics_snapshot.get('R2 Score', 0)
    mae = metrics_snapshot.get('MAE', 0)
    mse = metrics_snapshot.get('MSE', 0)
    rmse = metrics_snapshot.get('RMSE', 0)
    
    class_df = pd.DataFrame() 
    conf_matrix_array = np.array([]) 
    
    return r2, mae, mse, rmse, class_df, conf_matrix_array

def create_xgb_summary_text(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Generates the main summary text block (Report Asset Type: text)."""
    r2, mae, mse, rmse, _, _ = _get_common_metrics(metrics_snapshot)

    text = """
    ### XGBoost Regressor Performance Summary
    - **Features:** {features}
    - **Target:** {target}
    - **R2 Score:** **{r2:.4f}**
    - **Mean Absolute Error (MAE):** {mae:.4f}
    - **Mean Squared Error (MSE):** {mse:.4f}
    - **Root Mean Squared Error (RMSE):** {rmse:.4f}
    - **Best Parameters:** {best_params}
    - **Training Method:** {grid_search}
    """.format(
        features=", ".join(metrics_snapshot.get('features', [])),
        target=metrics_snapshot.get('target', 'Unknown'),
        r2=r2,
        mae=mae,
        mse=mse,
        rmse=rmse,
        best_params=metrics_snapshot.get('Best Parameters', {}),
        grid_search="Grid Search" if metrics_snapshot.get('use_grid_search', False) else "Manual"
    )
    return {"type": "text", "content": text}

def create_xgb_regression_report_table(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Creates a summary table of regression metrics."""
    r2 = metrics_snapshot.get('R2 Score', 0)
    mae = metrics_snapshot.get('MAE', 0)
    mse = metrics_snapshot.get('MSE', 0)
    rmse = metrics_snapshot.get('RMSE', 0)

    regression_summary = {
        'Metric': ['R2 Score', 'MAE', 'MSE', 'RMSE'],
        'Value': [f"{r2:.4f}", f"{mae:.4f}", f"{mse:.4f}", f"{rmse:.4f}"]
    }
    df = pd.DataFrame(regression_summary)
    return {"type": "dataframe", "content": df}

def create_xgb_performance_metrics_plot(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Generates a bar chart for main regression metrics (Report Asset Type: plot)."""
    r2, mae, mse, rmse, _, _ = _get_common_metrics(metrics_snapshot)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = ['$R^2$ Score']
    values = [r2]
    
    bars = ax.bar(metrics, values, color=['#4CAF50'])
    ax.set_ylim(0, 1.05)    
    ax.set_ylabel('Score')
    ax.set_title('Model Performance - $R^2$ Score')
    
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.4f}', ha='center', va='bottom')
                
    plt.close(fig)
    return {"type": "plot", "content": fig}

def create_xgb_error_metrics_plot(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Generates a bar chart for error metrics (Report Asset Type: plot)."""
    r2, mae, mse, rmse, _, _ = _get_common_metrics(metrics_snapshot)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    error_metrics = ['RMSE', 'MSE', 'MAE']
    error_values = [rmse, mse, mae]
    error_colors = ['#FF9800', '#F44336', '#2196F3']
    
    bars_err = ax.bar(error_metrics, error_values, color=error_colors)
    ax.set_ylabel('Error Value')
    ax.set_title('Model Performance - Error Metrics')
    
    for bar, value in zip(bars_err, error_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.close(fig)
    return {"type": "plot", "content": fig}

def create_xgb_prediction_analysis_plot(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
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

def create_xgb_residual_analysis_plot(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
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

def display_ml_report(metrics_snapshot: Dict[str, Any]):
    """
    Function to be called on the main ML page. It uses the granular functions
    to generate and display all components using Streamlit commands.
    """
    if not metrics_snapshot:
        st.error("No model data provided for display.")
        return

    st.markdown("---")
    st.markdown("## XGBoost Regressor Analysis (Live View)")
    
    summary_asset = create_xgb_summary_text(metrics_snapshot)
    table_asset = create_xgb_regression_report_table(metrics_snapshot)
    metrics_plot_asset = create_xgb_performance_metrics_plot(metrics_snapshot)
    error_plot_asset = create_xgb_error_metrics_plot(metrics_snapshot)
    
    st.markdown(summary_asset['content'], unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📈 $R^2$ Score")
        st.pyplot(metrics_plot_asset['content'])
    with col2:
        st.subheader("📊 Error Metrics")
        st.pyplot(error_plot_asset['content'])
    
    st.subheader("📋 Key Regression Metrics")
    st.dataframe(table_asset['content'], use_container_width=True)
    
    r2 = metrics_snapshot.get('R2 Score', 0)
    st.subheader("📊 Performance Interpretation")
    
    if r2 >= 0.9:
        st.success("**Excellent Performance** - The model explains a very high variance in the target variable!")
    elif r2 >= 0.7:
        st.info("**Good Performance** - The model provides a strong fit to the data.")
    elif r2 >= 0.5:
        st.warning("**Fair Performance** - The model has an acceptable predictive power but could be improved.")
    else:
        st.error("**Poor Performance** - The model explains less than 50% of the variance. Consider feature engineering, parameter tuning, or trying a different algorithm.")
        
    st.subheader("⚙️ Model Configuration")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.write("**Hyperparameters:**")
        best_params = metrics_snapshot.get('Best Parameters', {})
        if isinstance(best_params, dict):
            for param, value in best_params.items():
                st.write(f"- **{param}:** {value}")
        else:
            st.write(f"- {best_params}")
        
        st.write("**Training Method:**")
        st.write(f"- Grid Search: {'✅ Yes' if metrics_snapshot.get('use_grid_search', False) else '❌ No'}")
    
    with col4:
        st.write("**Dataset Information:**")
        features = metrics_snapshot.get('features', [])
        st.write(f"- Number of features: **{len(features)}**")
        st.write(f"- Target variable: **{metrics_snapshot.get('target', 'Unknown')}**")
        
        if metrics_snapshot.get('use_grid_search', False):
            cv_folds = metrics_snapshot.get('cv_folds', 'N/A')
            st.write(f"- CV Folds: **{cv_folds}**")
            
    if 'feature_importances' in metrics_snapshot and len(metrics_snapshot['feature_importances']) > 0:
        st.subheader("🎯 Feature Importances")
        importances = metrics_snapshot['feature_importances']
        if len(importances) == len(features):
            fig, ax = plt.subplots(figsize=(10, 6))
            importance_df = pd.DataFrame({
                'Feature': features,
                'Importance': importances
            }).sort_values('Importance', ascending=True)
            
            y_pos = np.arange(len(importance_df))
            ax.barh(y_pos, importance_df['Importance'], color='#FF6B6B')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(importance_df['Feature'])
            ax.set_xlabel('Importance Score')
            ax.set_title('XGBoost Regressor Feature Importances')
            plt.tight_layout()
            st.pyplot(fig)


def model_report():
    found = False 
    model_results = {}
    
    if hasattr(st.session_state, 'pipeline') and "ML" in st.session_state.pipeline:
        for model_info in st.session_state.pipeline.get("ML", []):
            model_name = (model_info.get('model name') or 
                          model_info.get('name') or 
                          model_info.get('model_type') or 
                          model_info.get('type', ''))
            
            if model_name == 'XGBoost Regressor':
                if 'metrics_snapshot' in model_info:
                    found = True
                    model_results = model_info.get("metrics_snapshot", {})
                    break

    if not found:
        st.error("No XGBoost Regressor model results found. Please create a model first.")
        return
    
    display_ml_report(model_results)

def display_compact_metrics():
    found = False
    model_results = {}
    
    for model_info in st.session_state.pipeline.get("ML", []):
        model_name = (model_info.get('model name') or 
                      model_info.get('name') or 
                      model_info.get('model_type') or 
                      model_info.get('type', ''))
            
        if model_name == 'XGBoost Regressor':
            if 'metrics_snapshot' in model_info:
                found = True
                model_results = model_info.get("metrics_snapshot", {})
                break

    if not found:
        return
    
    r2 = model_results.get('R2 Score', 0)
    mse = model_results.get('MSE', 0)
    rmse = model_results.get('RMSE', 0)
    mae = model_results.get('MAE', 0)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("R² Score", f"{r2:.4f}")
    with col2:
        st.metric("RMSE", f"{rmse:.4f}")
    with col3:
        st.metric("MSE", f"{mse:.4f}")
    with col4:
        st.metric("MAE", f"{mae:.4f}")
