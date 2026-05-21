import streamlit as st
import pandas as pd
import numpy as np
import re
def transformations_config(choice,edit_values):
            # Display appropriate fields based on transformation type
            transformation_params = {}
            # transformations = ["delete", "computation", "filter", "group"]
            # choice =st.selectbox("Select Transformation", transformations, index=transformations.index(edit_values.get("type", "delete")))
            if st.session_state.df_original is not None:
                if choice == "delete":
                    transformation_params = build_delete_transf(
                        df=st.session_state.df_to_show if not st.session_state.df_to_show.empty else st.session_state.df_original,
                        edit_values=edit_values
                    )
                elif choice == "computation":
                    transformation_params = build_computation_transf(
                        df=st.session_state.df_to_show if not st.session_state.df_to_show.empty else st.session_state.df_original,
                        edit_values=edit_values
                    )
                elif choice == "filter":
                    transformation_params = build_filter_transf(
                        df=st.session_state.df_to_show if not st.session_state.df_to_show.empty else st.session_state.df_original,
                        edit_values=edit_values
                    )
                elif choice == "group":
                    transformation_params = build_group_transf(
                        df=st.session_state.df_to_show if not st.session_state.df_to_show.empty else st.session_state.df_original,
                        edit_values=edit_values
                    )
            return transformation_params
def apply_selected_transformations(df, step):
    """Apply the selected transformations to the original dataframe"""
    if df is None or not isinstance(df, pd.DataFrame):
        return pd.DataFrame()
    if df.empty:
        return df

    try:
        category = step.get("category", "")

        if category == "delete":
            col = step.get("column")
            if col and col in df.columns:
                df = df.drop(columns=[col])
            else:
                import streamlit as st
                st.warning(f"Column '{col}' not found for delete transformation")
        elif category == "computation":
            new_col = step.get("new_column", "")
            expr = step.get('expr', "")
            if new_col and expr:
                result = calculate_expression(df, expr)
                if result is not None:
                    df = df.copy()
                    df[new_col] = result
            else:
                import streamlit as st
                st.warning("Computation transformation missing expression or column name")
        elif category == "filter":
            col = step.get("column")
            value = step.get("value", "")
            if col and col in df.columns:
                try:
                    # Try to coerce type for numeric columns
                    if pd.api.types.is_numeric_dtype(df[col]):
                        value = float(value)
                except (ValueError, TypeError):
                    pass
                df = df[df[col] == value]
            else:
                import streamlit as st
                st.warning(f"Column '{col}' not found for filter transformation")
        elif category == "group":
            group_col = step.get("group_col")
            target_col = step.get("target_col")
            agg_choice = step.get("agg_choice", "sum")
            if group_col and target_col and group_col in df.columns and target_col in df.columns:
                df = group_and_aggregate(df, group_col, target_col, agg_choice)
            else:
                import streamlit as st
                st.warning("Group transformation missing valid columns")
    except Exception as e:
        import streamlit as st
        st.error(f"Error applying transformation: {str(e)}")

    return df

def calculate_expression(df, expr: str) -> pd.Series:
    """Evaluates a mathematical expression string with column names in brackets"""
    columns = re.findall(r'\[([^\[\]]+)\]', expr)
    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in dataframe")
        expr = expr.replace(f"[{col}]", f"df['{col}']")
    
    try:
        return eval(expr)
    except Exception as e:
        raise ValueError(f"Failed to evaluate expression: {e}")

def group_and_aggregate(df: pd.DataFrame, group_col: str, target_col: str, agg_choice: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if group_col not in df.columns or target_col not in df.columns:
        return df
    try:
        grouped_df = (
            df.groupby(group_col)[target_col]
            .agg(agg_choice)
            .reset_index()
            .rename(columns={target_col: f"{agg_choice}_{target_col}"})
        )
        st.write("### 📊 Aggregated Data")
        return grouped_df
    except Exception as e:
        st.error(f"Error in group_and_aggregate: {str(e)}")
        return df

def build_group_transf(df, edit_values=None) -> dict:
    """Build the group transformation UI"""
    if df is None or df.empty:
        st.warning("No data available for group transformation.")
        return {"group_col": None, "target_col": None, "agg_choice": "sum"}

    cols = df.columns.tolist()
    default_group_col = edit_values.get("group_col", cols[0]) if edit_values and edit_values.get("group_col") in cols else cols[0]
    group_col = st.selectbox(
        "Column to group by",
        cols,
        index=cols.index(default_group_col) if default_group_col in cols else 0
    )
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if not numeric_cols:
        st.warning("No numeric columns available for aggregation.")
        return {"group_col": group_col, "target_col": None, "agg_choice": "sum"}
    default_target_col = edit_values.get("target_col", numeric_cols[0]) if edit_values and edit_values.get("target_col") in numeric_cols else numeric_cols[0]
    target_col = st.selectbox(
        "Numeric column to aggregate",
        numeric_cols,
        index=numeric_cols.index(default_target_col) if default_target_col in numeric_cols else 0
    )

    agg_functions = ['sum', 'mean', 'max', 'min', 'count']
    default_agg = edit_values.get("agg_choice", agg_functions[0]) if edit_values and edit_values.get("agg_choice") in agg_functions else agg_functions[0]
    agg_choice = st.selectbox(
        "Aggregation function",
        agg_functions,
        index=agg_functions.index(default_agg)
    )

    return {"group_col": group_col, "target_col": target_col, "agg_choice": agg_choice}
    
def build_delete_transf(df, edit_values=None) -> dict:
    """Build the delete transformation UI"""
    default_col = edit_values.get("column", df.columns[0]) if edit_values else df.columns[0]
    col_to_delete = st.selectbox(
        "Column to delete",
        df.columns.tolist(),
        index=df.columns.tolist().index(default_col) if edit_values else 0
    )
    return {"column": col_to_delete}
def build_computation_transf(df, edit_values=None) -> dict:
    """Build the computation transformation UI"""
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    st.markdown(f"**Numeric Columns:** {' | '.join(numeric_cols)}")
    
    default_expr = edit_values.get("expr", "") if edit_values else ""
    expr_cols = st.text_input("Expression", value=default_expr)
    
    default_new_col = edit_values.get("new_column", "") if edit_values else ""
    new_col_name = st.text_input("New column name", value=default_new_col)
    
    return {"expr": expr_cols, "new_column": new_col_name}

def build_filter_transf(df, edit_values=None) -> dict:
    """Build the filter transformation UI"""
    default_col = edit_values.get("column", df.columns[0]) if edit_values else df.columns[0]
    col_to_filter = st.selectbox(
        "Column to filter",
        df.columns.tolist(),
        index=df.columns.tolist().index(default_col) if edit_values else 0
    )
    default_value = edit_values.get("value", "") if edit_values else ""
    value_to_filter = st.text_input("Value to filter by", value=default_value)
    
    return {"column": col_to_filter, "value": value_to_filter}