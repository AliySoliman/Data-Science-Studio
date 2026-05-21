import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import re
from typing import Dict, Any, Optional

# ___________________________________________________________________________________________________________________________________________________________

def create_data_quality_visual(df: pd.DataFrame) -> go.Figure:
    """
    Create a bar chart showing the missing value percentage per column.
    
    This visual helps users identify which features require the most imputation 
    or might need to be dropped due to low data quality.
    """
    missing_pct = (df.isnull().sum() / len(df)) * 100
    missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=True)
    
    if missing_pct.empty:
        # Show a "Perfect Quality" placeholder if no missing values
        fig = go.Figure()
        fig.add_annotation(text="✅ 100% Data Completeness<br>No missing values detected!", 
                          showarrow=False, font=dict(size=20, color="#2ca02c"))
        fig.update_layout(title="Data Quality Summary", height=400)
        return fig

    fig = px.bar(
        x=missing_pct.values,
        y=missing_pct.index,
        orientation='h',
        title="Data Quality: Missing Values (%) 🔍",
        labels={'x': 'Missing Percentage (%)', 'y': 'Column Name'},
        color=missing_pct.values,
        color_continuous_scale='Reds'
    )
    
    fig.update_layout(height=400, showlegend=False, coloraxis_showscale=False)
    return fig

# ___________________________________________________________________________________________________________________________________________________________

def create_target_distribution_visual(df: pd.DataFrame, target: str, task_type: str) -> go.Figure:
    """
    Create a visualization for the target column's distribution.
    
    Uses a Pie chart for Classification (to show imbalance) and a 
    Histogram for Regression (to show skewness and range).
    """
    if target not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="No target column selected", showarrow=False)
        return fig

    if task_type == "Classification":
        counts = df[target].value_counts()
        fig = px.pie(
            values=counts.values,
            names=counts.index,
            title=f"Target Distribution: {target} 🎯",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
    else:
        fig = px.histogram(
            df, 
            x=target,
            title=f"Target Range & Skew: {target} 📈",
            marginal="box", 
            color_discrete_sequence=['#636EFA']
        )
        fig.update_layout(xaxis_title=target, yaxis_title="Frequency")

    fig.update_layout(height=400)
    return fig

# ___________________________________________________________________________________________________________________________________________________________

def create_correlation_visual(df: pd.DataFrame, target: Optional[str] = None) -> go.Figure:
    """
    Create a bar chart showing top feature correlations.
    
    If a target is provided, it shows features most correlated with the target.
    Otherwise, it shows the most correlated pairs in the dataset.
    """
    # Select only numeric columns for correlation
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty or len(numeric_df.columns) < 2:
        fig = go.Figure()
        fig.add_annotation(text="Insufficient numeric data for correlation analysis", showarrow=False)
        return fig

    if target and target in numeric_df.columns:
        corrs = numeric_df.corr()[target].abs().sort_values(ascending=False)
        # Drop the target itself
        corrs = corrs.drop(labels=[target])
        top_corrs = corrs.head(10).sort_values(ascending=True)
        
        fig = px.bar(
            x=top_corrs.values,
            y=top_corrs.index,
            orientation='h',
            title=f"Top Features Correlated with {target} 🔗",
            labels={'x': 'Absolute Correlation', 'y': 'Feature'},
            color=top_corrs.values,
            color_continuous_scale='Viridis'
        )
    else:
        # Show general correlation heatmap or top pairs
        corr_matrix = numeric_df.corr().abs()
        # Keep only upper triangle
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        top_pairs = upper.unstack().dropna().sort_values(ascending=False).head(10)
        top_pairs.index = [f"{a} & {b}" for a, b in top_pairs.index]
        top_pairs = top_pairs.sort_values(ascending=True)

        fig = px.bar(
            x=top_pairs.values,
            y=top_pairs.index,
            orientation='h',
            title="Top Feature Correlations 🔗",
            labels={'x': 'Absolute Correlation', 'y': 'Feature Pair'},
            color=top_pairs.values,
            color_continuous_scale='Plasma'
        )

    fig.update_layout(height=400, showlegend=False, coloraxis_showscale=False)
    return fig

# ___________________________________________________________________________________________________________________________________________________________
# ── Unsupervised-specific visuals ────────────────────────────────────────────

def create_feature_variance_visual(df: pd.DataFrame) -> go.Figure:
    """
    Bar chart of normalised feature variance (std / mean where mean != 0).

    Useful for unsupervised analysis to spot low-variance (uninformative)
    features that can distort clustering algorithms like KMeans or DBSCAN.
    The chart ranks features by coefficient of variation in descending order
    so the most 'spread-out' features appear at the top.
    """
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    if numeric_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No numeric features found.", showarrow=False)
        fig.update_layout(title="Feature Variance", height=350)
        return fig

    # Coefficient of variation (normalised std). Fall back to raw std if mean ≈ 0.
    means = numeric_df.mean().abs()
    stds  = numeric_df.std()
    cv    = stds.where(means < 1e-8, stds / means.replace(0, np.nan)).fillna(stds)
    cv    = cv.sort_values(ascending=True).tail(20)   # top 20 most variable

    fig = px.bar(
        x=cv.values,
        y=cv.index,
        orientation='h',
        title="Feature Spread (Coefficient of Variation) 📐",
        labels={'x': 'Coefficient of Variation', 'y': 'Feature'},
        color=cv.values,
        color_continuous_scale='Teal'
    )
    fig.update_layout(height=400, showlegend=False, coloraxis_showscale=False)
    return fig


def create_pairplot_visual(df: pd.DataFrame, max_features: int = 5) -> go.Figure:
    """
    Scatter-matrix (pair plot) of the top numeric features by variance.

    Shows pairwise relationships between the most variable numeric features —
    essential for spotting natural cluster structure before applying KMeans,
    DBSCAN, or Hierarchical Clustering.

    Args:
        df:           The full dataset.
        max_features: Maximum number of features to include (cap at 5 for readability).
    """
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    if numeric_df.shape[1] < 2:
        fig = go.Figure()
        fig.add_annotation(text="At least 2 numeric features required for a pair plot.", showarrow=False)
        fig.update_layout(title="Feature Pair Plot", height=400)
        return fig

    # Select top-N most variable features
    top_cols = numeric_df.std().sort_values(ascending=False).head(max_features).index.tolist()
    plot_df  = numeric_df[top_cols].dropna()

    fig = px.scatter_matrix(
        plot_df,
        dimensions=top_cols,
        title=f"Pair Plot — Top {len(top_cols)} Variable Features (Unsupervised View) 🔭",
        color_discrete_sequence=['#7B2D8B'],
    )
    fig.update_traces(diagonal_visible=True, showupperhalf=False, marker=dict(size=3, opacity=0.5))
    fig.update_layout(height=520)
    return fig


def create_categorical_frequency_visual(df: pd.DataFrame, max_cols: int = 3) -> go.Figure:
    """
    Stacked bar chart of value-count distributions for the top categorical columns.

    Helps inspect cardinality and dominant values in object/bool columns without
    needing a target variable — useful for understanding group structure.

    Args:
        df:       The full dataset.
        max_cols: Maximum number of categorical columns to show (default 3).
    """
    cat_df = df.select_dtypes(include=["object", "category", "bool"])
    # Pick columns with reasonable cardinality (2–30 unique values)
    suitable = [
        c for c in cat_df.columns
        if 1 < cat_df[c].nunique() <= 30
    ][:max_cols]

    if not suitable:
        fig = go.Figure()
        fig.add_annotation(
            text="No categorical columns with suitable cardinality (2–30 values) found.",
            showarrow=False
        )
        fig.update_layout(title="Categorical Distributions", height=350)
        return fig

    fig = go.Figure()
    for col in suitable:
        counts = cat_df[col].value_counts().head(15)
        fig.add_trace(go.Bar(name=col, x=counts.index.astype(str), y=counts.values))

    fig.update_layout(
        title="Categorical Feature Distributions 📊",
        barmode="group",
        xaxis_title="Category Value",
        yaxis_title="Count",
        height=400,
        legend_title="Column",
    )
    return fig

# ___________________________________________________________________________________________________________________________________________________________

def parse_and_enhance_recommendations(
    recommendations: str, 
    task_type: str, 
    df: pd.DataFrame, 
    target_column: Optional[str] = None
) -> Dict[str, Any]:
    """
    Parse LLM output and bundle it with real, data-driven visualizations.
    
    For supervised tasks (Classification / Regression) the standard three-chart
    set is returned: data quality, target distribution, and feature correlations.

    For unsupervised tasks (no target column), three cluster-focused charts are
    returned instead: feature variance, a pair plot, and categorical frequencies.

    Args:
        recommendations (str): The raw markdown text output from the LLM.
        task_type (str): "Classification", "Regression", or "Unsupervised".
        df (pd.DataFrame): The actual dataset for generating visuals.
        target_column (Optional[str]): The designated target variable.

    Returns:
        Dict[str, Any]: Enhanced recommendation payload for the frontend.
    """
    sections = re.split(r'(?=^##\s*\d+️⃣)', recommendations, flags=re.MULTILINE)

    is_unsupervised = (task_type == "Unsupervised") or (not target_column)

    if is_unsupervised:
        # ── Unsupervised chart suite ──────────────────────────────────────────
        visuals = {
            "data_quality":      create_data_quality_visual(df),
            "feature_variance":  create_feature_variance_visual(df),
            "pairplot":          create_pairplot_visual(df),
            "cat_frequency":     create_categorical_frequency_visual(df),
            # Keep None so the caller can branch on this key
            "target_dist":       None,
            "correlation":       create_correlation_visual(df, target=None),
        }
    else:
        # ── Supervised chart suite (Classification / Regression) ─────────────
        visuals = {
            "data_quality": create_data_quality_visual(df),
            "target_dist":  create_target_distribution_visual(df, target_column, task_type),
            "correlation":  create_correlation_visual(df, target_column),
            # These keys are absent in supervised mode — callers can check with .get()
            "feature_variance": create_feature_variance_visual(df),
            "pairplot":         create_pairplot_visual(df),
            "cat_frequency":    create_categorical_frequency_visual(df),
        }
        
    return {
        "full_text":     recommendations,
        "sections":      sections,
        "visuals":       visuals,
        "is_unsupervised": is_unsupervised,
    }
