import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any

def _get_common_metrics(metrics_snapshot: Dict[str, Any]):
    """Extracts common metrics from the snapshot with safe defaults."""
    # Safe access with defaults for ALL required keys
    r2 = metrics_snapshot.get('R² Score', 0.0)
    adjusted_r2 = metrics_snapshot.get('Adjusted R²', 0.0)
    mae = metrics_snapshot.get('MAE', 0.0)
    mse = metrics_snapshot.get('MSE', 0.0)
    rmse = metrics_snapshot.get('RMSE', 0.0)
    mape = metrics_snapshot.get('MAPE', 0.0)
    coefficients = metrics_snapshot.get('Coefficients', {})
    intercept = metrics_snapshot.get('Intercept', 0.0)
    model_type = metrics_snapshot.get('model_type', 'linear')
    features = metrics_snapshot.get('features', [])
    target = metrics_snapshot.get('target', 'unknown')
    alpha = metrics_snapshot.get('alpha', 1.0)
    
    return (r2, adjusted_r2, mae, mse, rmse, mape, coefficients, 
            intercept, model_type, features, target, alpha)

# INDEPENDENT REPORT ITEM FUNCTIONS
def create_regression_summary_text(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Create a text summary of regression results"""
    (r2, adjusted_r2, mae, mse, rmse, mape, coefficients, 
     intercept, model_type, features, target, alpha) = _get_common_metrics(metrics_snapshot)
    
    model_names = {
        'linear': 'Linear Regression',
        'ridge': 'Ridge Regression', 
        'lasso': 'Lasso Regression'
    }
    
    # Check if we have valid data
    if not features:
        return {"type": "text", "content": "❌ No regression model data available. Please train a model first."}

    summary_text = """
    ### {model_name} Summary

    **R² Score:** {r2:.4f}  
    **Adjusted R²:** {adj_r2:.4f}  
    **Mean Absolute Error:** {mae:.4f}  
    **Root Mean Squared Error:** {rmse:.4f}  
    **Mean Absolute Percentage Error:** {mape:.2f}%

    **Features:** {features}
    **Target:** {target}
    {alpha_text}
    **Intercept:** {intercept:.4f}

    ### Model Performance:
    - **R² Score** indicates {r2_pct:.1%} of variance explained by the model
    - **Lower MSE/RMSE/MAE** values indicate better prediction accuracy
    - **MAPE** shows average percentage error in predictions
    """.format(
        model_name=model_names.get(model_type, 'Regression'),
        r2=r2,
        adj_r2=adjusted_r2,
        mae=mae,
        rmse=rmse,
        mape=mape,
        features=", ".join(features) if features else "No features",
        target=target,
        alpha_text=f"**Alpha:** {alpha}" if model_type in ['ridge', 'lasso'] else "",
        intercept=intercept,
        r2_pct=r2
    )
    
    return {
        "type": "text",
        "content": summary_text
    }

def create_coefficients_table(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Create a table of regression coefficients"""
    (_, _, _, _, _, _, coefficients, _, _, features, _, _) = _get_common_metrics(metrics_snapshot)
    
    if not coefficients:
        return {
            "type": "dataframe",
            "content": pd.DataFrame({
                'Feature': ['No coefficients available'],
                'Coefficient': [0],
                'Impact': ['N/A']
            })
        }
    
    coefficients_data = {
        'Feature': list(coefficients.keys()),
        'Coefficient': [float(coef) for coef in coefficients.values()],
        'Impact': ['Positive' if coef >= 0 else 'Negative' for coef in coefficients.values()]
    }
    
    df = pd.DataFrame(coefficients_data)
    df = df.sort_values('Coefficient', key=abs, ascending=False)
    
    return {
        "type": "dataframe",
        "content": df
    }

def create_residuals_plot(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Create residuals plot for regression diagnostics using real data from snapshot"""
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
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_pred, residuals, alpha=0.6, color='green')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Residuals')
    ax.set_title('Residuals Plot (Real Data)\n(Ideal: random scatter around zero)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.close(fig)
    
    return {
        "type": "plot",
        "content": fig
    }

def create_performance_comparison(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Create performance metrics comparison table"""
    (r2, adjusted_r2, mae, mse, rmse, mape, _, _, _, _, _, _) = _get_common_metrics(metrics_snapshot)
    
    performance_data = {
        'Metric': ['R² Score', 'Adjusted R²', 'MSE', 'RMSE', 'MAE', 'MAPE'],
        'Value': [
            f"{r2:.4f}",
            f"{adjusted_r2:.4f}",
            f"{mse:.4f}",
            f"{rmse:.4f}",
            f"{mae:.4f}",
            f"{mape:.2f}%"
        ],
        'Interpretation': [
            'Higher is better (0-1)',
            'Higher is better (0-1)',
            'Lower is better', 
            'Lower is better',
            'Lower is better',
            'Lower is better (%)'
        ]
    }
    
    df = pd.DataFrame(performance_data)
    
    return {
        "type": "dataframe", 
        "content": df
    }

def create_feature_importance_plot(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Create feature importance plot based on coefficients"""
    (_, _, _, _, _, _, coefficients, _, _, features, _, _) = _get_common_metrics(metrics_snapshot)
    
    if not coefficients:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No coefficients available', 
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title('Feature Importance - No Data')
        plt.tight_layout()
        plt.close(fig)
        return {"type": "plot", "content": fig}
    
    features_list = list(coefficients.keys())
    coef_values = [float(coef) for coef in coefficients.values()]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    y_pos = np.arange(len(features_list))
    
    colors = ['green' if x >= 0 else 'red' for x in coef_values]
    bars = ax.barh(y_pos, coef_values, color=colors, alpha=0.7)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features_list)
    ax.set_xlabel('Coefficient Value')
    ax.set_title('Feature Coefficients (Impact on Target)')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        label_x = width if width >= 0 else width
        ax.text(label_x, bar.get_y() + bar.get_height()/2, 
                f'{width:.4f}', ha='left' if width >= 0 else 'right', 
                va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.close(fig)
    
    return {
        "type": "plot",
        "content": fig
    }

# Legacy function for backward compatibility
def model_report():
    """Legacy function - use individual report functions instead"""
    found = False 
    model_results = None
    for model_info in st.session_state.pipeline.get("ML", []):
        if model_info.get("model type") == 'Linear Regression':
            if 'metrics_snapshot' in model_info:
                found = True
                model_results = model_info.get("metrics_snapshot", {})
                break
    
    if not found:
        st.error("No Linear Regression model results found. Please create a model first.")
        return
    
    # Display all components like the original report
    st.markdown("---")
    st.markdown("## Linear Regression Model Analysis (Live View)")
    
    # Get all the report assets
    summary_asset = create_regression_summary_text(model_results)
    coefficients_asset = create_coefficients_table(model_results)
    residuals_asset = create_residuals_plot(model_results)
    performance_asset = create_performance_comparison(model_results)
    importance_asset = create_feature_importance_plot(model_results)
    
    # Display Summary
    st.markdown(summary_asset['content'], unsafe_allow_html=True)
    
    # Display Performance Metrics Table
    st.subheader("📋 Performance Metrics Summary")
    st.dataframe(performance_asset['content'], use_container_width=True)
    
    # Display Feature Importance Plot
    st.subheader("🔍 Feature Coefficients Analysis")
    st.pyplot(importance_asset['content'])
    
    # Display Coefficients Table
    st.subheader("📊 Detailed Coefficients Table")
    st.dataframe(coefficients_asset['content'], use_container_width=True)
    
    # Display Residuals Plot
    st.subheader("📈 Residuals Analysis")
    st.pyplot(residuals_asset['content'])
    
    # Performance Interpretation
    st.subheader("📊 Performance Interpretation")
    r2_score_val = model_results.get('R² Score', 0)
    mape_val = model_results.get('MAPE', 0)
    
    if r2_score_val >= 0.8:
        st.success("**Excellent Fit** - Model explains most of the variance in the target variable!")
    elif r2_score_val >= 0.6:
        st.info("**Good Fit** - Model provides reasonable predictions.")
    elif r2_score_val >= 0.4:
        st.warning("**Fair Fit** - Model has moderate predictive power.")
    else:
        st.error("**Poor Fit** - Consider feature engineering or different model.")
    
    if mape_val <= 10:
        st.success("**Excellent Accuracy** - Very low percentage error!")
    elif mape_val <= 20:
        st.info("**Good Accuracy** - Reasonable prediction accuracy.")
    elif mape_val <= 30:
        st.warning("**Fair Accuracy** - Moderate prediction error.")
    else:
        st.error("**Poor Accuracy** - High percentage error in predictions.")
    
    # Model Configuration Details
    st.subheader("⚙️ Model Configuration")
    
    model_type = model_results.get('model_type', 'linear')
    features = model_results.get('features', [])
    target = model_results.get('target', 'unknown')
    alpha = model_results.get('alpha', 1.0)
    
    model_names = {
        'linear': 'Linear Regression',
        'ridge': 'Ridge Regression', 
        'lasso': 'Lasso Regression'
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Model Parameters:**")
        st.write(f"- Model Type: **{model_names.get(model_type, model_type)}**")
        if model_type in ['ridge', 'lasso']:
            st.write(f"- Alpha: **{alpha}**")
        st.write(f"- Features: **{len(features)}**")
        st.write(f"- Target: **{target}**")
    
    with col2:
        st.write("**Key Metrics:**")
        st.write(f"- R² Score: **{r2_score_val:.4f}**")
        st.write(f"- Adjusted R²: **{model_results.get('Adjusted R²', 0):.4f}**")
        st.write(f"- RMSE: **{model_results.get('RMSE', 0):.4f}**")
        st.write(f"- MAE: **{model_results.get('MAE', 0):.4f}**")
    
    # Compact Metrics Display
    st.subheader("🔢 Compact Metrics View")
    display_compact_metrics_legacy(model_results)

def display_compact_metrics_legacy(metrics_snapshot):
    """Alternative compact metrics display for legacy function"""
    if not metrics_snapshot:
        return
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric("R² Score", f"{metrics_snapshot.get('R² Score', 0):.4f}")
    with col2:
        st.metric("Adj R²", f"{metrics_snapshot.get('Adjusted R²', 0):.4f}")
    with col3:
        st.metric("RMSE", f"{metrics_snapshot.get('RMSE', 0):.4f}")
    with col4:
        st.metric("MAE", f"{metrics_snapshot.get('MAE', 0):.4f}")
    with col5:
        st.metric("MAPE", f"{metrics_snapshot.get('MAPE', 0):.2f}%")
    with col6:
        st.metric("MSE", f"{metrics_snapshot.get('MSE', 0):.4f}")