import streamlit as st
from transformations.encoding.onehot_encoder import apply_onehot_encoding,build_onehot_encoding
from transformations.encoding.label_encoder import apply_label_encoding,build_label_encoding
from transformations.encoding.ordinal_encoding import build_ordinal_encoding, apply_ordinal_encoding
from transformations.encoding.target_encoder import apply_target_encoding, build_target_encoding_transf
from transformations.encoding.binary_encoder import apply_binary_encoding, build_binary_encoding_transf
def encoding_config(choice,edit_values):
            # Display appropriate fields based on transformation type
            transformation_params = {}
            # transformations = ["delete", "computation", "filter", "group"]
            # choice =st.selectbox("Select Transformation", transformations, index=transformations.index(edit_values.get("type", "delete")))
            if st.session_state.df_original is not None:
                if choice == "Ordinal Encoding":
                    transformation_params = build_ordinal_encoding(
                        df=st.session_state.df_to_show if not st.session_state.df_to_show.empty else st.session_state.df_original,
                        edit_values=edit_values
                    )
                elif choice == "Label Encoding":
                    transformation_params = build_label_encoding(
                        df=st.session_state.df_to_show if not st.session_state.df_to_show.empty else st.session_state.df_original,
                        edit_values=edit_values
                    )
                elif choice == "one-hot encoding":
                    transformation_params = build_onehot_encoding(
                        df=st.session_state.df_to_show if not st.session_state.df_to_show.empty else st.session_state.df_original,
                        edit_values=edit_values
                )
                elif choice == "Target Encoding":
                        transformation_params = build_target_encoding_transf(
                            df=st.session_state.df_to_show if not st.session_state.df_to_show.empty else st.session_state.df_original,
                            edit_values=edit_values
                        )
                elif choice == "Binary Encoding":
                    transformation_params = build_binary_encoding_transf(
                        df=st.session_state.df_to_show if not st.session_state.df_to_show.empty else st.session_state.df_original,
                        edit_values=edit_values
                    )
            return transformation_params


def apply_selected_encodings(df, step):
    """Apply the selected encodings to the original dataframe"""
    if df is None or not isinstance(df, __import__('pandas').DataFrame):
        return __import__('pandas').DataFrame()
    if df.empty:
        return df

    category = step.get("category", "")
    try:
        if category == "Ordinal Encoding":
            df = apply_ordinal_encoding(df, step)
        elif category == "one-hot encoding":
            df = apply_onehot_encoding(df, step)
        elif category == 'Label Encoding':
            df = apply_label_encoding(df, step)
        elif category == 'Target Encoding':
            df = apply_target_encoding(df, step)
        elif category == 'Binary Encoding':
            df = apply_binary_encoding(df, step)
    except Exception as e:
        import streamlit as st
        st.error(f"Error applying encoding '{category}': {str(e)}")
    return df


