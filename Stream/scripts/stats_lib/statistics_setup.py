import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List

# --- Configuration Builders (UI) ---

def statistics_config(choice: str, df: pd.DataFrame, edit_values: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handles the UI configuration for different statistical analyses based on user choice.
    """
    st.subheader(f"Configure {choice.replace('_', ' ').title()}")
    transformation_params = {"category": choice}
    
    if choice == "descriptive_stats":
        transformation_params.update(build_descriptive_config(df, edit_values))
    elif choice == "t_test_two_sample":
        transformation_params.update(build_t_test_two_sample_config(df, edit_values))
    elif choice == "confidence_interval":
        transformation_params.update(build_confidence_interval_config(df, edit_values))
    elif choice == "correlation_regression":
        transformation_params.update(build_correlation_regression_config(df, edit_values))
    
    return transformation_params

def build_descriptive_config(df: pd.DataFrame, edit_values: Dict[str, Any]) -> Dict[str, Any]:
    """Builds UI for Descriptive Statistics."""
    
    # Get all numeric columns for selection
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    # Allow selection of columns
    default_cols = edit_values.get("columns", numeric_cols)
    selected_cols = st.multiselect(
        "Select columns for analysis",
        options=numeric_cols,
        default=default_cols
    )
    
    return {"columns": selected_cols}

def build_t_test_two_sample_config(df: pd.DataFrame, edit_values: Dict[str, Any]) -> Dict[str, Any]:
    """Builds UI for Two-Sample T-Test."""
    
    st.markdown("⚠️ **Two-Sample T-Test** requires one **grouping column** (categorical) and one **value column** (numeric).")
    
    # Grouping column (Categorical)
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    default_group_col = edit_values.get("group_col", categorical_cols[0] if categorical_cols else None)
    group_col = st.selectbox(
        "Grouping Column (Categorical)",
        options=categorical_cols,
        index=categorical_cols.index(default_group_col) if default_group_col in categorical_cols else 0
    )
    
    # Value column (Numeric)
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    default_value_col = edit_values.get("value_col", numeric_cols[0] if numeric_cols else None)
    value_col = st.selectbox(
        "Value Column (Numeric)",
        options=numeric_cols,
        index=numeric_cols.index(default_value_col) if default_value_col in numeric_cols else 0
    )

    if group_col and value_col:
        unique_groups = df[group_col].nunique()
        if unique_groups != 2:
            st.warning(f"The grouping column '{group_col}' has {unique_groups} unique values. A two-sample t-test requires exactly two groups.")
    
    return {"group_col": group_col, "value_col": value_col}


def build_confidence_interval_config(df: pd.DataFrame, edit_values: Dict[str, Any]) -> Dict[str, Any]:
    """Builds UI for Confidence Interval Estimation."""
    
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    default_col = edit_values.get("column", numeric_cols[0] if numeric_cols else None)
    selected_col = st.selectbox(
        "Select Numeric Column",
        options=numeric_cols,
        index=numeric_cols.index(default_col) if default_col in numeric_cols else 0
    )
    
    confidence_levels = [90, 95, 99]
    default_level = edit_values.get("conf_level", 95)
    conf_level = st.selectbox(
        "Confidence Level (%)",
        options=confidence_levels,
        index=confidence_levels.index(default_level) if default_level in confidence_levels else 1
    )
    
    return {"column": selected_col, "conf_level": conf_level}

def build_correlation_regression_config(df: pd.DataFrame, edit_values: Dict[str, Any]) -> Dict[str, Any]:
    """Builds UI for Correlation & Regression Preview."""
    
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    default_x = edit_values.get("x_col", numeric_cols[0] if numeric_cols else None)
    default_y = edit_values.get("y_col", numeric_cols[1] if len(numeric_cols) > 1 else (numeric_cols[0] if numeric_cols else None))
    
    x_col = st.selectbox(
        "X Variable (Predictor)",
        options=numeric_cols,
        index=numeric_cols.index(default_x) if default_x in numeric_cols else 0,
        key="corr_x"
    )
    
    y_col = st.selectbox(
        "Y Variable (Response)",
        options=numeric_cols,
        index=numeric_cols.index(default_y) if default_y in numeric_cols else (1 if len(numeric_cols) > 1 else 0),
        key="corr_y"
    )
    
    if x_col == y_col and x_col is not None:
        st.error("X and Y variables must be different.")
        return {"x_col": None, "y_col": None}

    return {"x_col": x_col, "y_col": y_col}


# --- Execution Functions ---

def apply_selected_statistics(df: pd.DataFrame, step: Dict[str, Any]) -> None:
    """
    Executes the selected statistical analysis and displays the results.
    Note: Statistical steps don't modify the dataframe, they only output results.
    """
    category = step["category"]
    st.markdown(f"### 📈 Results for: {step['name']}")
    
    if category == "descriptive_stats":
        execute_descriptive_stats(df, step)
    elif category == "t_test_two_sample":
        execute_t_test_two_sample(df, step)
    elif category == "confidence_interval":
        execute_confidence_interval(df, step)
    elif category == "correlation_regression":
        execute_correlation_regression(df, step)

def execute_descriptive_stats(df: pd.DataFrame, step: Dict[str, Any]):
    """Calculates and displays Descriptive Statistics and plots."""
    cols = step["columns"]
    if not cols:
        st.warning("No columns selected for descriptive statistics.")
        return
        
    df_desc = df[cols]
    
    # Calculate main descriptive stats
    stats_df = df_desc.agg(['mean', 'median', 'std', 'var', 'skew', 'kurt']).T
    stats_df.columns = stats_df.columns.str.title()
    stats_df['Mode'] = df_desc.mode().iloc[0]

    # Display results in a clean format
    st.dataframe(stats_df.style.format(precision=3))

    # Correlation Matrix with Heatmap
    st.markdown("#### 🌡️ Correlation Matrix")
    corr_matrix = df_desc.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5, ax=ax)
    st.pyplot(fig)
    plt.close(fig) # Prevent plot from showing twice
    
    # Plots (Histograms & Boxplots)
    st.markdown("#### 📊 Distributions & Outliers")
    for col in cols:
        st.markdown(f"##### {col}")
        
        col1, col2 = st.columns(2)
        
        # Histogram
        with col1:
            fig, ax = plt.subplots()
            sns.histplot(df_desc[col].dropna(), kde=True, ax=ax)
            ax.set_title(f"Histogram of {col}")
            st.pyplot(fig)
            plt.close(fig)

        # Boxplot
        with col2:
            fig, ax = plt.subplots()
            sns.boxplot(y=df_desc[col].dropna(), ax=ax)
            ax.set_title(f"Boxplot of {col}")
            st.pyplot(fig)
            plt.close(fig)


def execute_t_test_two_sample(df: pd.DataFrame, step: Dict[str, Any]):
    """Performs and displays results for a Two-Sample T-Test."""
    group_col = step["group_col"]
    value_col = step["value_col"]

    if not group_col or not value_col:
        st.warning("Grouping and value columns must be selected to run the T-Test.")
        return

    unique_groups = df[group_col].dropna().unique()
    if len(unique_groups) != 2:
        st.error(f"Cannot run Two-Sample T-Test: The grouping column '{group_col}' must have exactly two unique values. Found: {len(unique_groups)}")
        return

    group1 = df[df[group_col] == unique_groups[0]][value_col].dropna()
    group2 = df[df[group_col] == unique_groups[1]][value_col].dropna()

    if group1.empty or group2.empty:
        st.error("One or both groups are empty after dropping NaNs. Cannot perform T-Test.")
        return
        
    # Perform independent two-sample t-test (assuming unequal variance for simplicity - Welch's t-test)
    t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)
    
    # Determine conclusion
    alpha = 0.05
    conclusion = f"**{unique_groups[0]}** mean ($\mu_1$) is **not significantly different** from **{unique_groups[1]}** mean ($\mu_2$)."
    if p_value < alpha:
        conclusion = f"**Reject the null hypothesis (H₀)**. There is **significant evidence** that the mean of **{unique_groups[0]}** and **{unique_groups[1]}** are different at the {100-alpha*100}% confidence level."

    # Display numerical results
    st.markdown(f"| Metric | Value |")
    st.markdown(f"| :--- | :--- |")
    st.markdown(f"| **T-Statistic** | {t_stat:.4f} |")
    st.markdown(f"| **P-Value** | {p_value:.4f} |")
    st.markdown(f"| **Confidence Level** | {100-alpha*100}% |")

    # Conclusion
    st.markdown(f"#### 🧠 What it means: {conclusion}")
    
    # Simple Visual Explanation
    st.markdown("#### 📊 Distribution Comparison")
    fig, ax = plt.subplots()
    sns.kdeplot(group1, label=unique_groups[0], fill=True, ax=ax)
    sns.kdeplot(group2, label=unique_groups[1], fill=True, ax=ax)
    ax.set_title(f"Distribution of {value_col} by {group_col}")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)


def execute_confidence_interval(df: pd.DataFrame, step: Dict[str, Any]):
    """Calculates and displays the Confidence Interval for the mean."""
    col = step["column"]
    conf_level = step["conf_level"] / 100.0
    
    if not col:
        st.warning("A column must be selected to calculate the Confidence Interval.")
        return

    data = df[col].dropna()
    if data.empty:
        st.error(f"Column '{col}' is empty or contains only NaNs. Cannot calculate CI.")
        return
        
    mean = data.mean()
    std_err = data.std() / np.sqrt(len(data))
    
    # Calculate the confidence interval using the t-distribution (more robust for sample means)
    ci_low, ci_high = stats.t.interval(conf_level, len(data) - 1, loc=mean, scale=std_err)

    # Display results
    st.markdown(f"The **{conf_level*100:.0f}% Confidence Interval** for the mean of **{col}** is:")
    st.markdown(f"$$CI = [{ci_low:.3f}, {ci_high:.3f}]$$")
    st.markdown(f"**Mean ($\mu$):** {mean:.3f}")

    # Visualize with a Confidence Interval Plot
    st.markdown("#### 🎯 Confidence Interval Plot")
    fig, ax = plt.subplots()
    ax.errorbar(mean, 0, xerr=[[mean - ci_low], [ci_high - mean]], fmt='o', capsize=5, label=f'{conf_level*100:.0f}% CI')
    ax.axvline(mean, color='red', linestyle='--', alpha=0.6, label='Mean')
    ax.set_yticks([]) # Hide Y-axis for a cleaner look
    ax.set_title(f'{conf_level*100:.0f}% CI for Mean of {col}')
    ax.legend(loc='upper right')
    st.pyplot(fig)
    plt.close(fig)


def execute_correlation_regression(df: pd.DataFrame, step: Dict[str, Any]):
    """Calculates correlation and displays a regression preview."""
    x_col = step["x_col"]
    y_col = step["y_col"]
    
    if not x_col or not y_col:
        st.warning("Both X and Y columns must be selected for Correlation/Regression Preview.")
        return
        
    # Drop NaNs for the two columns
    df_clean = df[[x_col, y_col]].dropna()
    
    if df_clean.empty:
        st.error("No valid data points after dropping NaNs in selected columns.")
        return

    # Calculate Pearson's Correlation Coefficient
    corr_coeff = df_clean[x_col].corr(df_clean[y_col])
    
    # Interpret correlation
    abs_corr = abs(corr_coeff)
    strength = "Very Strong" if abs_corr >= 0.8 else ("Strong" if abs_corr >= 0.6 else ("Moderate" if abs_corr >= 0.3 else "Weak"))
    direction = "Positive" if corr_coeff > 0 else ("Negative" if corr_coeff < 0 else "No")
    
    if abs_corr < 0.05:
         summary = "No significant linear relationship found between X and Y."
    else:
        summary = f"There is a **{strength.lower()} {direction.lower()} correlation** ($\mathbf{{r={corr_coeff:.3f}}}$) between **{x_col}** and **{y_col}**."

    st.markdown(f"**Correlation Coefficient ($r$):** `{corr_coeff:.3f}`")
    st.markdown(f"#### 🧠 Summary: {summary}")

    # Scatter Plot with Line of Best Fit
    st.markdown("#### 📉 Regression Preview")
    fig, ax = plt.subplots()
    sns.regplot(x=x_col, y=y_col, data=df_clean, ax=ax, scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
    ax.set_title(f'Regression of {y_col} on {x_col}')
    st.pyplot(fig)
    plt.close(fig)