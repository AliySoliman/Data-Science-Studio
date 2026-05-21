import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from scipy import stats

def cleanings_config(choice, edit_values=None):
    """Build the cleaning transformation UI based on choice"""
    transformation_params = {}
    
    if st.session_state.df_original is not None:
        df = st.session_state.df_original
        
        if choice == "Duplicate Handling":
            transformation_params = build_duplicate_handling(df, edit_values)
        elif choice == "Null handling":
            transformation_params = build_null_handling(df, edit_values)
        elif choice == "Outlier Handling":
            transformation_params = build_outlier_handling(df, edit_values)
        elif choice == "Rename Columns":
            transformation_params = build_rename_columns(df, edit_values)
        elif choice == "Binary Encoding":
            transformation_params = build_binary_encoding(df, edit_values)
        elif choice == "Date Validation":
            transformation_params = build_date_validation(df, edit_values)
    
    return transformation_params

def apply_cleaning_transformation(df, step):
    """Apply the selected cleaning transformation to the dataframe"""
    df_copy = df.copy()
    
    try:
        if step["category"] == "Duplicate Handling":
            if step["action"] == "remove":
                df_copy = df_copy.drop_duplicates()
            elif step["action"] == "keep_first":
                df_copy = df_copy.drop_duplicates(keep='first')
            elif step["action"] == "keep_last":
                df_copy = df_copy.drop_duplicates(keep='last')
                
        elif step["category"] == "Null handling":
            df_copy = handle_missing_values(df_copy, step)
            
        elif step["category"] == "Outlier Handling":
            df_copy = handle_outliers(df_copy, step)
            
        elif step["category"] == "Rename Columns":
            for old_name, new_name in step["rename_mapping"].items():
                if old_name in df_copy.columns:
                    df_copy = df_copy.rename(columns={old_name: new_name})
                    
        elif step["category"] == "Binary Encoding":
            df_copy = apply_binary_encoding(df_copy, step)
            
        elif step["category"] == "Date Validation":
            df_copy = validate_dates(df_copy, step)
            
    except Exception as e:
        st.error(f"Error applying cleaning transformation: {str(e)}")
        return df
    
    return df_copy

# =============================================================================
# Individual Cleaning Function Builders
# =============================================================================

def show_duplicate_counts(df):
    """Display the number of duplicate rows in the DataFrame using Streamlit."""
    total_rows = len(df)
    total_duplicates = df.duplicated().sum()

    if total_duplicates:
        st.info(f"📊 **Duplicate Analysis:** {total_duplicates} duplicate rows found out of {total_rows} total rows.")
    else:
        st.success(f"✅ No duplicate rows found in the dataset ({total_rows} rows total).")

    return total_duplicates

def show_null_counts(df, selected_columns=None):
    """Display null value counts for selected columns"""
    if selected_columns is None:
        selected_columns = df.columns
    
    st.write("📊 **Missing Value Analysis:**")
    null_counts = df[selected_columns].isnull().sum()
    total_nulls = null_counts.sum()
    
    for col in selected_columns:
        null_count = null_counts[col]
        if null_count > 0:
            null_percentage = (null_count / len(df)) * 100
            st.write(f"   - `{col}`: {null_count} null values ({null_percentage:.1f}%)")
    
    if total_nulls == 0:
        st.success("✅ No missing values found in selected columns")
    
    return null_counts

def show_outlier_counts(df, selected_columns, method="IQR"):
    """Display outlier counts for selected numeric columns"""
    st.write("📊 **Outlier Analysis:**")
    outlier_info = {}
    
    for col in selected_columns:
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue
            
        if method == "IQR":
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outlier_count = len(outliers)
            
        elif method == "Z-score":
            z_scores = stats.zscore(df[col].dropna())
            outlier_count = (np.abs(z_scores) > 3).sum()
        
        if outlier_count > 0:
            outlier_percentage = (outlier_count / len(df)) * 100
            st.write(f"   - `{col}`: {outlier_count} outliers ({outlier_percentage:.1f}%)")
            outlier_info[col] = outlier_count
    
    if not outlier_info:
        st.success("✅ No outliers found in selected columns")
    
    return outlier_info

def build_duplicate_handling(df, edit_values=None):
    """Build duplicate handling UI"""
    st.write("### Duplicate Handling")
    
    # Show duplicate counts
    total_duplicates = show_duplicate_counts(df)
    if total_duplicates == 0:
        st.info("No duplicates to handle.")
        return {"action": None}
    
    actions = ["remove", "keep_first", "keep_last"]
    default_action = edit_values.get("action", "remove") if edit_values else "remove"
    action = st.selectbox(
        "Action for duplicates",
        actions,
        index=actions.index(default_action) if edit_values else 0
    )
    
    return {"action": action}

def build_null_handling(df, edit_values=None):
    """Build null handling UI"""
    st.write("### Missing Value Imputation")
    
    # Column selection
    all_columns = df.columns.tolist()
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    
    default_columns = edit_values.get("columns", all_columns) if edit_values else all_columns
    selected_columns = st.multiselect(
        "Select columns to handle missing values",
        all_columns,
        default=default_columns
    )
    
    # Show null counts for selected columns
    if selected_columns:
        show_null_counts(df, selected_columns)
    
    # Strategy selection
    strategies = {}
    for col in selected_columns:
        st.write(f"**{col}**")
        if col in numeric_columns:
            num_strategies = ["mean", "median", "knn", "drop"]
            default_strategy = edit_values.get("strategies", {}).get(col, "mean") if edit_values else "mean"
            strategy = st.selectbox(
                f"Strategy for {col}",
                num_strategies,
                index=num_strategies.index(default_strategy) if default_strategy in num_strategies else 0,
                key=f"strategy_{col}"
            )
            strategies[col] = strategy
            
            if strategy == "knn":
                default_neighbors = edit_values.get("knn_neighbors", 5) if edit_values else 5
                strategies["knn_neighbors"] = st.number_input(
                    "KNN neighbors",
                    min_value=1,
                    max_value=20,
                    value=default_neighbors,
                    key=f"knn_{col}"
                )
                
        elif col in categorical_columns:
            cat_strategies = ["mode", "unknown", "drop"]
            default_strategy = edit_values.get("strategies", {}).get(col, "mode") if edit_values else "mode"
            strategy = st.selectbox(
                f"Strategy for {col}",
                cat_strategies,
                index=cat_strategies.index(default_strategy) if default_strategy in cat_strategies else 0,
                key=f"strategy_{col}"
            )
            strategies[col] = strategy
    
    return {"columns": selected_columns, "strategies": strategies}

def build_outlier_handling(df, edit_values=None):
    """Build outlier handling UI"""
    st.write("### Outlier Handling")
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    default_columns = edit_values.get("columns", numeric_columns) if edit_values else numeric_columns
    selected_columns = st.multiselect(
        "Select numeric columns for outlier handling",
        numeric_columns,
        default=default_columns
    )
    
    # Show outlier counts for selected columns
    if selected_columns:
        methods = ["IQR", "Z-score"]
        default_method = edit_values.get("method", "IQR") if edit_values else "IQR"
        method = st.selectbox(
            "Outlier detection method",
            methods,
            index=methods.index(default_method) if edit_values else 0
        )
        
        show_outlier_counts(df, selected_columns, method)
        
        actions = ["capping", "deletion"]
        default_action = edit_values.get("action", "capping") if edit_values else "capping"
        action = st.selectbox(
            "Action for outliers",
            actions,
            index=actions.index(default_action) if edit_values else 0
        )
        
        return {
            "columns": selected_columns,
            "method": method,
            "action": action
        }
    else:
        st.warning("Please select at least one numeric column for outlier detection")
        return {"columns": [], "method": "IQR", "action": "capping"}


def build_rename_columns(df, edit_values=None):
    """Build rename columns UI"""
    st.write("### Rename Columns")
    
    columns = df.columns.tolist()
    rename_mapping = edit_values.get("rename_mapping", {}) if edit_values else {}
    
    new_mapping = {}
    for col in columns:
        default_new_name = rename_mapping.get(col, col)
        new_name = st.text_input(
            f"New name for '{col}'",
            value=default_new_name,
            key=f"rename_{col}"
        )
        if new_name != col:
            new_mapping[col] = new_name
    
    return {"rename_mapping": new_mapping}


def build_binary_encoding(df, edit_values=None):
    """Build binary encoding UI"""
    st.write("### Binary Encoding for Boolean/Categorical Columns")
    
    categorical_columns = df.select_dtypes(include=['object', 'bool']).columns.tolist()
    
    default_columns = edit_values.get("columns", categorical_columns) if edit_values else categorical_columns
    selected_columns = st.multiselect(
        "Select columns for binary encoding",
        categorical_columns,
        default=default_columns
    )
    
    return {"columns": selected_columns}

def build_date_validation(df, edit_values=None):
    """Build date validation UI"""
    st.write("### Date Validation and Cleaning")
    
    # Try to identify date columns
    potential_date_cols = []
    for col in df.columns:
        if df[col].dtype == 'object':
            # Check if column might contain dates
            sample = df[col].dropna().head(10)
            if len(sample) > 0 and any('/' in str(x) or '-' in str(x) for x in sample):
                potential_date_cols.append(col)
    
    default_columns = edit_values.get("date_columns", potential_date_cols) if edit_values else potential_date_cols
    date_columns = st.multiselect(
        "Select date columns",
        df.columns.tolist(),
        default=default_columns
    )
    
    actions = ["validate", "drop_invalid", "impute"]
    default_action = edit_values.get("action", "validate") if edit_values else "validate"
    action = st.selectbox(
        "Action for invalid dates",
        actions,
        index=actions.index(default_action) if edit_values else 0
    )
    
    return {
        "date_columns": date_columns,
        "action": action
    }

# =============================================================================
# Core Cleaning Functions
# =============================================================================

def handle_missing_values(df, step):
    """Handle missing values based on selected strategies"""
    df_copy = df.copy()
    strategies = step["strategies"]
    
    for col, strategy in strategies.items():
        if col not in df_copy.columns:
            continue
            
        if strategy == "drop":
            df_copy = df_copy.dropna(subset=[col])
        elif strategy == "mean" and pd.api.types.is_numeric_dtype(df_copy[col]):
            df_copy[col] = df_copy[col].fillna(df_copy[col].mean())
        elif strategy == "median" and pd.api.types.is_numeric_dtype(df_copy[col]):
            df_copy[col] = df_copy[col].fillna(df_copy[col].median())
        elif strategy == "knn" and pd.api.types.is_numeric_dtype(df_copy[col]):
            # Use KNN imputation for numeric columns
            knn_imputer = KNNImputer(n_neighbors=step.get("knn_neighbors", 5))
            numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
            df_copy[numeric_cols] = knn_imputer.fit_transform(df_copy[numeric_cols])
        elif strategy == "mode":
            df_copy[col] = df_copy[col].fillna(df_copy[col].mode()[0] if not df_copy[col].mode().empty else "Unknown")
        elif strategy == "unknown":
            df_copy[col] = df_copy[col].fillna("Unknown")
    
    return df_copy

def handle_outliers(df, step):
    """Handle outliers using IQR or Z-score method"""
    df_copy = df.copy()
    method = step["method"]
    action = step["action"]
    
    for col in step["columns"]:
        if col not in df_copy.columns or not pd.api.types.is_numeric_dtype(df_copy[col]):
            continue
            
        if method == "IQR":
            Q1 = df_copy[col].quantile(0.25)
            Q3 = df_copy[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            if action == "capping":
                df_copy[col] = np.where(df_copy[col] < lower_bound, lower_bound, df_copy[col])
                df_copy[col] = np.where(df_copy[col] > upper_bound, upper_bound, df_copy[col])
            elif action == "deletion":
                df_copy = df_copy[(df_copy[col] >= lower_bound) & (df_copy[col] <= upper_bound)]
                
        elif method == "Z-score":
            z_scores = stats.zscore(df_copy[col].dropna())
            abs_z_scores = np.abs(z_scores)
            
            if action == "capping":
                # Cap at 3 standard deviations
                mean_val = df_copy[col].mean()
                std_val = df_copy[col].std()
                lower_bound = mean_val - 3 * std_val
                upper_bound = mean_val + 3 * std_val
                df_copy[col] = np.where(df_copy[col] < lower_bound, lower_bound, df_copy[col])
                df_copy[col] = np.where(df_copy[col] > upper_bound, upper_bound, df_copy[col])
            elif action == "deletion":
                filtered_entries = abs_z_scores < 3
                df_copy = df_copy.iloc[filtered_entries]
    
    return df_copy

def apply_binary_encoding(df, step):
    """Apply binary encoding to selected columns"""
    df_copy = df.copy()
    
    for col in step["columns"]:
        if col in df_copy.columns:
            # Convert to binary (0/1) encoding
            unique_vals = df_copy[col].unique()
            if len(unique_vals) == 2:
                # Binary case
                df_copy[col] = pd.factorize(df_copy[col])[0]
            else:
                # Multiple categories - create dummy variables
                dummies = pd.get_dummies(df_copy[col], prefix=col)
                df_copy = pd.concat([df_copy, dummies], axis=1)
                df_copy = df_copy.drop(columns=[col])
    
    return df_copy

def validate_dates(df, step):
    """Validate and clean date columns"""
    df_copy = df.copy()
    
    for col in step["date_columns"]:
        if col in df_copy.columns:
            # Try to convert to datetime
            df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
            
            if step["action"] == "drop_invalid":
                # Drop rows with invalid dates
                df_copy = df_copy.dropna(subset=[col])
            elif step["action"] == "impute":
                # Impute missing/invalid dates with a placeholder
                df_copy[col] = df_copy[col].fillna(pd.Timestamp('1970-01-01'))
    
    return df_copy