from transformations.feature_selection.filter_method.chi_squared import build_chi_squared, apply_chi_squared_selection
from transformations.feature_selection.wrapper_method.ref import build_ref_selection, apply_ref_selection
import streamlit as st
from transformations.feature_selection.correlation import build_corr_reduction_transf, reduce_by_correlation
from transformations.feature_selection.variance_selection import build_variance_reduction_transf, reduce_by_variance
from transformations.feature_selection.ANOVA import build_anova_transf, apply_ANOVA
def get_feature_config(choice, edit_values):
    """Get the appropriate config function for feature reduction"""
    if choice == "Chi-Squared Feature Selection":
        return build_chi_squared(st.session_state.df_to_show if not st.session_state.df_to_show.empty else st.session_state.df_original, edit_values)
    elif choice == "REF Feature Selection":
        return build_ref_selection(st.session_state.df_to_show if not st.session_state.df_to_show.empty else st.session_state.df_original, edit_values)
    elif choice == "Correlation Feature Selection":
        return build_corr_reduction_transf(st.session_state.df_to_show if not st.session_state.df_to_show.empty else st.session_state.df_original, edit_values)
    elif choice == "Variance Feature Selection":
        return build_variance_reduction_transf(st.session_state.df_to_show if not st.session_state.df_to_show.empty else st.session_state.df_original, edit_values)
    elif choice == "ANOVA":
        return build_anova_transf(
                df=st.session_state.df_to_show if not st.session_state.df_to_show.empty else st.session_state.df_original,
                edit_values=edit_values
            )
    else:
        return {}

def get_feature_execution(df, step):
    """Get the appropriate execution function for feature reduction"""
    if step["category"] == "Chi-Squared Feature Selection":
        return apply_chi_squared_selection(st.session_state.df_to_show if not st.session_state.df_to_show.empty else st.session_state.df_original, step)
    elif step["category"] == "REF Feature Selection":
        return apply_ref_selection(st.session_state.df_to_show if not st.session_state.df_to_show.empty else st.session_state.df_original, step)
    elif step["category"] == "Correlation Feature Selection":
        return reduce_by_correlation(st.session_state.df_to_show if not st.session_state.df_to_show.empty else st.session_state.df_original, step)
    elif step["category"] == "Variance Feature Selection":
        return  reduce_by_variance(st.session_state.df_to_show if not st.session_state.df_to_show.empty else st.session_state.df_original, step)
    elif step["category"] == "ANOVA":
        return apply_ANOVA(st.session_state.df_to_show if not st.session_state.df_to_show.empty else st.session_state.df_original, step)
    
    return df
    