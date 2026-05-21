from transformations.dimensionality_reduction.tsne import build_tsne, apply_tsne
from transformations.dimensionality_reduction.umap import build_umap, apply_umap
from transformations.dimensionality_reduction.LDA import build_lda_reduction, apply_lda_reduction
from transformations.dimensionality_reduction.PCA import build_pca_config, apply_pca
import streamlit as st
def get_dimension_config(choice, edit_values):
    """Get the appropriate config function for dimension extraction"""

    if choice == "t-SNE Dimensionality Reduction":
        return build_tsne(st.session_state.df_to_show if not st.session_state.df_to_show.empty else st.session_state.df_original, edit_values)
    elif choice == "UMAP Dimensionality Reduction":
        return build_umap(st.session_state.df_to_show if not st.session_state.df_to_show.empty else st.session_state.df_original, edit_values)
    elif choice == "LDA":   
        return build_lda_reduction(st.session_state.df_to_show if not st.session_state.df_to_show.empty else st.session_state.df_original, edit_values)
    elif choice == "PCA":
        return build_pca_config(edit_values)
    else:
        return {}

def get_dimension_execution(df, step):
    """Get the appropriate execution function for dimension extraction"""
    if step["category"] == "t-SNE Dimensionality Reduction":
        return apply_tsne(st.session_state.df_to_show if not st.session_state.df_to_show.empty else st.session_state.df_original, step)
    elif step["category"] == "UMAP Dimensionality Reduction":
        return apply_umap(st.session_state.df_to_show if not st.session_state.df_to_show.empty else st.session_state.df_original, step)
    elif step["category"] == "LDA":   
        return apply_lda_reduction(st.session_state.df_to_show if not st.session_state.df_to_show.empty else st.session_state.df_original, step)
    elif step["category"] == "PCA":
        return apply_pca(st.session_state.df_to_show if not st.session_state.df_to_show.empty else st.session_state.df_original, step)
    else:
        return df
