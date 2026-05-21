import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def build_pca_config(edit_values=None):
    """Build PCA configuration UI"""
    st.markdown("""
    **Principal Component Analysis (PCA)**
    - Linear dimensionality reduction using Singular Value Decomposition
    - Projects data to lower dimensions while preserving variance
    - Useful for visualization and noise reduction
    """)
    
    df = st.session_state.df_to_show if not st.session_state.df_to_show.empty else st.session_state.df_original
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if not numeric_cols:
        st.error("No numeric columns found for PCA")
        return {"columns": [], "n_components": 2}
    
    # Column selection
    default_cols = edit_values.get("columns", []) if edit_values else numeric_cols
    selected_cols = st.multiselect(
        "Select numeric columns for PCA:",
        options=numeric_cols,
        default=default_cols,
        help="Select numeric columns to include in PCA analysis"
    )
    
    # Number of components
    max_components = min(len(selected_cols), 10) if selected_cols else 2
    default_components = edit_values.get("n_components", 2) if edit_values else 2
    n_components = st.slider(
        "Number of principal components:",
        min_value=1,
        max_value=max_components,
        value=min(default_components, max_components),
        help="Number of components to keep"
    )
    
    # Advanced options
    # with st.expander("Advanced Options"):
    col1, col2 = st.columns(2)
    with col1:
        whiten = st.checkbox(
            "Whiten components",
            value=edit_values.get("whiten", False) if edit_values else False,
            help="Whiten the components to remove correlation"
        )
    with col2:
        svd_solver = st.selectbox(
            "SVD solver",
            options=['auto', 'full', 'arpack', 'randomized'],
            index=['auto', 'full', 'arpack', 'randomized'].index(
                edit_values.get("svd_solver", "auto") if edit_values else "auto"
            ),
            help="SVD solver to use"
        )
    
    # Preview selected data
    if selected_cols:
        st.info(f"Selected {len(selected_cols)} features for PCA")
        st.write("**Selected Features Statistics:**")
        st.dataframe(df[selected_cols].describe().round(4))
    
    return {
        "columns": selected_cols,
        "n_components": n_components,
        "whiten": whiten,
        "svd_solver": svd_solver
    }

def apply_pca(df, step):
    """Apply PCA to selected columns"""
    columns = step.get("columns", [])
    n_components = step.get("n_components", 2)
    whiten = step.get("whiten", False)
    svd_solver = step.get("svd_solver", "auto")
    
    if not columns:
        st.warning("No columns selected for PCA")
        return df
    
    if len(columns) < n_components:
        st.error(f"Number of components ({n_components}) cannot be greater than number of features ({len(columns)})")
        return df
    
    # Create a copy to avoid modifying original
    df_transformed = df.copy()
    
    # Extract selected features
    X = df_transformed[columns].values
    
    # Handle missing values
    if np.isnan(X).any():
        st.warning("Missing values found. Filling with column means...")
        X = np.nan_to_num(X, nan=np.nanmean(X, axis=0))
    
    # Initialize and fit PCA
    pca = PCA(n_components=n_components, whiten=whiten, svd_solver=svd_solver)
    X_pca = pca.fit_transform(X)
    
    # Create new column names for principal components
    pc_columns = [f'PC{i+1}' for i in range(n_components)]
    
    # Add principal components to dataframe
    for i, pc_col in enumerate(pc_columns):
        df_transformed[pc_col] = X_pca[:, i]
    
    # Show PCA results
    st.success(f"✅ PCA applied: {len(columns)} features → {n_components} principal components")
    
    # Display explained variance
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Explained Variance Ratio:**")
        variance_df = pd.DataFrame({
            'Component': pc_columns,
            'Explained Variance': pca.explained_variance_ratio_,
            'Cumulative Variance': np.cumsum(pca.explained_variance_ratio_)
        })
        st.dataframe(variance_df.round(4))
    
    with col2:
        # Plot explained variance
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(range(1, n_components + 1), pca.explained_variance_ratio_, alpha=0.7)
        ax.plot(range(1, n_components + 1), np.cumsum(pca.explained_variance_ratio_), 'ro-')
        ax.set_xlabel('Principal Components')
        ax.set_ylabel('Explained Variance Ratio')
        ax.set_title('PCA Explained Variance')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    # Show component loadings
    st.write("**Component Loadings (First 5 features):**")
    loadings_df = pd.DataFrame(
        pca.components_[:5].T if len(pca.components_) > 5 else pca.components_.T,
        columns=[f'PC{i+1}' for i in range(min(5, n_components))],
        index=columns
    )
    st.dataframe(loadings_df.round(4))
    
    return df_transformed