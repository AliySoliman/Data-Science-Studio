import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any

# --- Helper to safely retrieve metrics and convert JSON-safe lists back to arrays ---

def _get_common_metrics(metrics_snapshot: Dict[str, Any]):
    """Extracts common regression metrics."""
    # --- Regression Metrics ---
    r2 = metrics_snapshot.get('R2 Score', 0)
    mae = metrics_snapshot.get('MAE', 0)
    mse = metrics_snapshot.get('MSE', 0)
    
    # Placeholder/Dummy for compatibility, not used in regressor
    class_df = pd.DataFrame() 
    conf_matrix_array = np.array([]) 
    
    return r2, mae, mse, class_df, conf_matrix_array

# --- Granular Report Asset Generation Functions (Input: metrics_snapshot) ---

def create_ml_summary_text(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Generates the main summary text block (Report Asset Type: text)."""
    # Use R2, MAE, MSE
    r2, mae, mse, _, _ = _get_common_metrics(metrics_snapshot)

    text = """
    ### KNN Regressor Performance Summary
    - **Features:** {features}
    - **Target:** {target}
    - **R2 Score:** **{r2:.4f}**
    - **Mean Absolute Error (MAE):** {mae:.4f}
    - **Mean Squared Error (MSE):** {mse:.4f}
    - **Best Parameters:** {best_params}
    - **Training Method:** {grid_search}
    """.format(
        features=", ".join(metrics_snapshot['features']),
        target=metrics_snapshot['target'],
        r2=r2,
        mae=mae,
        mse=mse,
        best_params=metrics_snapshot['Best Parameters'],
        grid_search="Grid Search" if metrics_snapshot['use_grid_search'] else "Manual"
    )
    # The returned dictionary is the Report Asset Token (must be returned, not st.written)
    return {"type": "text", "content": text}


# def create_confusion_matrix_plot(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
#     """Placeholder - Confusion Matrix is NOT applicable to Regression."""
#     return {"type": "text", "content": "Confusion Matrix not applicable for KNN Regressor."}


def create_classification_report_table(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Placeholder - Classification Report is NOT applicable to Regression."""
    
    regression_summary = {
        'Metric': ['R2 Score', 'MAE', 'MSE'],
        'Value': [
            f"{metrics_snapshot.get('R2 Score', 0):.4f}", 
            f"{metrics_snapshot.get('MAE', 0):.4f}", 
            f"{metrics_snapshot.get('MSE', 0):.4f}"
        ]
    }
    df = pd.DataFrame(regression_summary)
    
    # Returning a dataframe that summarizes the metrics as a table
    return {"type": "dataframe", "content": df}


def create_performance_metrics_plot(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Generates a bar chart for main regression metrics (Report Asset Type: plot)."""
    # Use R2, MAE, MSE
    r2, mae, mse, _, _ = _get_common_metrics(metrics_snapshot)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # We plot R2 score separately as it's typically between 0 and 1.
    # MAE/MSE can be any positive value, so we'll only plot R2 for the fixed range plot.
    metrics = ['$R^2$ Score']
    values = [r2]
    
    bars = ax.bar(metrics, values, color=['#4CAF50'])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score')
    ax.set_title('Model Performance - $R^2$ Score')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.4f}', ha='center', va='bottom')
                
    plt.close(fig)
    
    # The returned dictionary is the Report Asset Token
    return {"type": "plot", "content": fig}

def create_knn_prediction_analysis_plot(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
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

def create_knn_residual_analysis_plot(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
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
    st.markdown("## KNN Regressor Analysis (Live View)") # Updated Title
    
    # Get all the report assets
    summary_asset = create_ml_summary_text(metrics_snapshot)
    table_asset = create_classification_report_table(metrics_snapshot)
    metrics_plot_asset = create_performance_metrics_plot(metrics_snapshot)
    
    # Display Summary
    st.markdown(summary_asset['content'], unsafe_allow_html=True)
    
    # Display Metrics Plot
    st.subheader("📈 $R^2$ Score Visualization")
    st.pyplot(metrics_plot_asset['content'])
    
    # Display Regression Metrics Table
    st.subheader("📋 Key Regression Metrics")
    st.dataframe(table_asset['content'], use_container_width=True)
    
    # Performance Interpretation based on R2 score
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
        
    # Model Configuration Details (Reusing original logic structure)
    st.subheader("⚙️ Model Configuration")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.write("**Hyperparameters:**")
        best_params = metrics_snapshot['Best Parameters']
        if isinstance(best_params, dict):
            for param, value in best_params.items():
                st.write(f"- **{param}:** {value}")
        else:
            st.write(f"- {best_params}")
        
        st.write("**Training Method:**")
        st.write(f"- Grid Search: {'✅ Yes' if metrics_snapshot['use_grid_search'] else '❌ No'}")
    
    with col4:
        st.write("**Dataset Information:**")
        st.write(f"- Number of features: **{len(metrics_snapshot['features'])}**")
        st.write(f"- Target variable: **{metrics_snapshot['target']}**")
        
        if metrics_snapshot['use_grid_search']:
            cv_folds = metrics_snapshot.get('cv_folds', 'N/A')
            st.write(f"- CV Folds: **{cv_folds}**")
            
    # NOTE: Confusion Matrix removed as it's not relevant for regression.

# For compatibility with your old system, we keep the original function name, 
# but it now just calls the decoupled display function.
def model_report():
    found = False 
    # Check for the updated model name
    for model_info in st.session_state.pipeline.get("ML", []):
        if model_info.get("model name") == 'KNN_Regressor':
            if 'metrics_snapshot' in model_info:
                found = True
                model_results = model_info.get("metrics_snapshot", {})
                break
    
    # If the regressor wasn't found, try the old name for robustness
    if not found:
        for model_info in st.session_state.pipeline.get("ML", []):
            if model_info.get("model name") == 'KNN': # Fallback check
                if 'metrics_snapshot' in model_info:
                    found = True
                    model_results = model_info.get("metrics_snapshot", {})
                    break

    if not found:
        st.error("No KNN Regressor model results found. Please create a model first.")
        return
    
    # Pass the JSON-safe metrics snapshot 
    display_ml_report(model_results)