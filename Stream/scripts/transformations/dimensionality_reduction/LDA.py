import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder

def show_dataset_info(df, target_column=None):
    """Display information about the dataset"""
    st.write("📊 **Dataset Information:**")
    st.write(f"   - Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    st.write(f"   - Numeric columns: {len(df.select_dtypes(include=[np.number]).columns.tolist())}")
    st.write(f"   - Categorical columns: {len(df.select_dtypes(include=['object', 'category']).columns.tolist())}")
    
    if target_column and target_column in df.columns:
        st.write(f"   - Target column: '{target_column}'")
        st.write(f"   - Target classes: {df[target_column].nunique()}")
        st.write(f"   - Class distribution:")
        class_counts = df[target_column].value_counts()
        for class_name, count in class_counts.items():
            percentage = (count / len(df)) * 100
            st.write(f"     - {class_name}: {count} ({percentage:.1f}%)")

def build_lda_reduction(df, edit_values=None):
    """Build LDA dimensionality reduction UI"""
    st.write("### LDA (Linear Discriminant Analysis)")
    st.info("LDA is a supervised dimensionality reduction technique that requires a target variable for classification.")
    
    # Initialize with safe default values
    result_params = {
        "features": [],
        "target": None,
        "n_components": 1,
        "keep_original_features": [],
        "keep_target": True
    }
    
    # Get numeric columns for features
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_columns) < 2:
        st.warning("Need at least 2 numeric columns for LDA")
        return result_params
    
    # Get potential target columns (categorical with reasonable number of classes)
    potential_targets = []
    for col in df.columns:
        if df[col].dtype in ['object', 'category'] or df[col].nunique() <= 20:  # Increased limit to 20
            potential_targets.append(col)
    
    if not potential_targets:
        st.warning("No suitable target column found for LDA. Need a categorical column or column with <= 20 unique values.")
        return result_params
    
    # Feature selection
    st.write("### Feature Selection")
    default_features = edit_values.get("features", numeric_columns) if edit_values else numeric_columns
    selected_features = st.multiselect(
        "Select numeric features for LDA",
        numeric_columns,
        default=default_features
    )
    
    # Target selection
    st.write("### Target Selection")
    default_target = edit_values.get("target", potential_targets[0]) if edit_values else potential_targets[0]
    target_column = st.selectbox(
        "Select target variable",
        potential_targets,
        index=potential_targets.index(default_target) if default_target in potential_targets else 0
    )
    
    if selected_features and target_column:
        # Show dataset info
        show_dataset_info(df, target_column)
        
        # Calculate maximum possible components
        n_classes = df[target_column].nunique()
        n_features = len(selected_features)
        max_components = min(n_features, n_classes - 1)
        
        # Debug information
        st.write("### LDA Constraints")
        st.write(f"   - Number of features selected: {n_features}")
        st.write(f"   - Number of classes in target: {n_classes}")
        st.write(f"   - Maximum LDA components possible: {max_components} (min({n_features}, {n_classes}-1))")
        
        if max_components < 1:
            st.error(f"⚠️ Cannot perform LDA: Target variable '{target_column}' has only {n_classes} class(es). LDA requires at least 2 classes.")
            return result_params
        
        if max_components == 1:
            st.warning(f"⚠️ Limited to 1 component: With {n_classes} classes and {n_features} features, maximum components is {max_components}")
        
        # Number of components selection
        st.write("### LDA Configuration")
        
        if max_components > 1:
            n_components = st.slider(
                "Number of LDA components",
                min_value=1,
                max_value=max_components,
                value=min(edit_values.get("n_components", min(2, max_components)), max_components) if edit_values else min(2, max_components),
                help=f"Maximum possible components: {max_components} (min(n_features={n_features}, n_classes-1={n_classes-1}))"
            )
        else:
            n_components = 1
            st.info(f"Automatically set to 1 component (maximum possible with current selection)")
        
        # Handle categorical target option
        handle_categorical_target = st.checkbox(
            "Encode categorical target automatically",
            value=True,
            help="Convert categorical target labels to numeric values for LDA"
        )
        
        # Feature retention options
        st.write("### Feature Retention")
        
        # Option to keep target column
        keep_target = st.checkbox(
            "Keep target column in final dataset",
            value=True,
            help="Include the target variable in the transformed dataset"
        )
        
        # Option to keep original features
        st.write("Select which original features to keep (optional):")
        keep_original_features = st.multiselect(
            "Features to retain along with LDA components",
            selected_features,
            default=edit_values.get("keep_original_features", []) if edit_values else [],
            help="Choose which original features to keep in addition to LDA components"
        )
        
        # Show what the final dataset will contain
        st.write("### Final Dataset Composition")
        final_features = [f'LDA_{i+1}' for i in range(n_components)]
        if keep_original_features:
            final_features.extend(keep_original_features)
        if keep_target:
            final_features.append(target_column)
        
        st.write(f"**Total columns in final dataset:** {len(final_features)}")
        st.write(f"**Columns:** {final_features}")
        
        # Show dimension reduction summary
        original_cols = len(selected_features) + (1 if keep_target else 0)
        final_cols = len(final_features)
        reduction = original_cols - final_cols
        
        if reduction > 0:
            st.success(f"📉 Dimension reduction: {original_cols} → {final_cols} columns ({reduction})")
        else:
            st.info(f"📊 No dimension reduction: {original_cols} → {final_cols} columns")
        
        # Preview LDA
        if st.checkbox("Preview LDA transformation"):
            preview_lda_reduction(df, selected_features, target_column, n_components, handle_categorical_target, keep_original_features, keep_target)
        
        # Update result parameters
        result_params = {
            "features": selected_features,
            "target": target_column,
            "n_components": n_components,
            "handle_categorical_target": handle_categorical_target,
            "keep_original_features": keep_original_features,
            "keep_target": keep_target
        }
    else:
        st.warning("Please select features and target variable for LDA")
    
    return result_params

def preview_lda_reduction(df, features, target, n_components, handle_categorical_target, keep_original_features, keep_target):
    """Preview LDA transformation"""
    st.write("### LDA Preview")
    
    try:
        # Prepare data
        X = df[features].copy()
        y = df[target].copy()
        
        # Handle missing values in features
        X = X.fillna(X.mean())
        
        # Handle categorical target if needed
        if handle_categorical_target and y.dtype in ['object', 'category']:
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            st.write(f"**Target encoding:** {dict(zip(le.classes_, range(len(le.classes_))))}")
        else:
            y_encoded = y.values
        
        # Remove rows where target is missing
        valid_indices = ~y.isna()
        X = X[valid_indices]
        y_encoded = y_encoded[valid_indices]
        y_original = y[valid_indices]
        
        # Apply LDA
        lda = LinearDiscriminantAnalysis(n_components=n_components)
        X_lda = lda.fit_transform(X, y_encoded)
        
        # Create results dataframe
        lda_columns = [f'LDA_{i+1}' for i in range(n_components)]
        lda_df = pd.DataFrame(X_lda, columns=lda_columns, index=X.index)
        
        # Add selected original features back
        if keep_original_features:
            for feature in keep_original_features:
                if feature in df.columns:
                    lda_df[feature] = df.loc[X.index, feature]
        
        # Add target if requested
        if keep_target:
            lda_df[target] = y_original.values
        
        st.write("**LDA Transformed Data (first 5 rows):**")
        st.dataframe(lda_df.head())
        
        # Show explained variance ratio
        if hasattr(lda, 'explained_variance_ratio_'):
            st.write("**Explained Variance Ratio:**")
            for i, ratio in enumerate(lda.explained_variance_ratio_):
                st.write(f"   - LDA_{i+1}: {ratio:.3f} ({ratio*100:.1f}%)")
            total_variance = lda.explained_variance_ratio_.sum()
            st.write(f"   - **Total variance explained: {total_variance:.3f} ({total_variance*100:.1f}%)**")
        
        # Plot if 2 or more components
        if n_components >= 2:
            st.write("**LDA Visualization (First 2 Components):**")
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Use original labels for coloring
            if handle_categorical_target and y_original.dtype in ['object', 'category']:
                colors = pd.factorize(y_original)[0]
                class_labels = y_original.unique()
            else:
                colors = y_encoded
                class_labels = np.unique(y_encoded)
            
            scatter = ax.scatter(X_lda[:, 0], X_lda[:, 1], c=colors, cmap='viridis', alpha=0.7)
            ax.set_xlabel('LDA Component 1')
            ax.set_ylabel('LDA Component 2')
            ax.set_title('LDA Projection')
            
            # Add legend
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=plt.cm.viridis(i/len(class_labels)), 
                                        markersize=8, label=str(cls)) 
                            for i, cls in enumerate(class_labels)]
            ax.legend(handles=legend_elements, title='Classes')
            
            st.pyplot(fig)
        elif n_components == 1:
            st.write("**LDA Visualization (Single Component):**")
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Use original labels for coloring
            if handle_categorical_target and y_original.dtype in ['object', 'category']:
                colors = pd.factorize(y_original)[0]
                class_labels = y_original.unique()
            else:
                colors = y_encoded
                class_labels = np.unique(y_encoded)
            
            scatter = ax.scatter(X_lda[:, 0], np.zeros_like(X_lda[:, 0]), c=colors, cmap='viridis', alpha=0.7)
            ax.set_xlabel('LDA Component 1')
            ax.set_title('LDA Projection (1D)')
            ax.set_yticks([])
            
            # Add legend
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=plt.cm.viridis(i/len(class_labels)), 
                                        markersize=8, label=str(cls)) 
                            for i, cls in enumerate(class_labels)]
            ax.legend(handles=legend_elements, title='Classes')
            
            st.pyplot(fig)
        
        # Show component coefficients
        st.write("**Feature Coefficients for LDA Components:**")
        coeff_df = pd.DataFrame(
            lda.scalings_[:, :n_components],
            index=features,
            columns=lda_columns
        )
        st.dataframe(coeff_df.style.background_gradient(cmap='RdBu_r', axis=None))
            
    except Exception as e:
        st.error(f"Error in LDA preview: {str(e)}")

# =============================================================================
# Core Dimensionality Reduction Functions
# =============================================================================

def apply_lda_reduction(df, step):
    """Apply LDA dimensionality reduction"""
    df_copy = df.copy()
    
    features = step.get("features", [])
    target = step.get("target")
    n_components = step.get("n_components", 1)
    handle_categorical_target = step.get("handle_categorical_target", True)
    keep_original_features = step.get("keep_original_features", [])
    keep_target = step.get("keep_target", True)
    
    if not features or not target:
        st.warning("No features or target selected for LDA")
        return df_copy
    
    if target not in df_copy.columns:
        st.error(f"Target column '{target}' not found in dataframe")
        return df_copy
    
    try:
        # Prepare data
        X = df_copy[features].copy()
        y = df_copy[target].copy()
        
        # Handle missing values in features
        X = X.fillna(X.mean())
        
        # Handle categorical target if needed
        if handle_categorical_target and y.dtype in ['object', 'category']:
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            target_encoder = le
        else:
            y_encoded = y.values
            target_encoder = None
        
        # Remove rows where target is missing
        valid_indices = ~y.isna()
        X = X[valid_indices]
        y_encoded = y_encoded[valid_indices]
        y_original = y[valid_indices]
        
        # Apply LDA
        lda = LinearDiscriminantAnalysis(n_components=n_components)
        X_lda = lda.fit_transform(X, y_encoded)
        
        # Create LDA column names
        lda_columns = [f'LDA_{i+1}' for i in range(n_components)]
        
        # Create new dataframe with LDA components
        lda_df = pd.DataFrame(X_lda, columns=lda_columns, index=X.index)
        
        # Add selected original features back
        if keep_original_features:
            for feature in keep_original_features:
                if feature in df_copy.columns:
                    lda_df[feature] = df_copy.loc[X.index, feature]
        
        # Add target if requested
        if keep_target:
            lda_df[target] = y_original.values
        
        # Drop original features that are not being kept
        features_to_drop = [f for f in features if f not in keep_original_features]
        df_copy = df_copy.drop(columns=features_to_drop)
        
        # Merge LDA components with remaining dataframe
        df_copy = pd.concat([df_copy, lda_df], axis=1)
        
        # Store LDA model for potential inverse transformation
        if 'dimensionality_reduction' not in st.session_state:
            st.session_state.dimensionality_reduction = {}
        
        st.session_state.dimensionality_reduction['lda'] = {
            'model': lda,
            'features': features,
            'target': target,
            'n_components': n_components,
            'target_encoder': target_encoder,
            'handle_categorical_target': handle_categorical_target,
            'keep_original_features': keep_original_features,
            'keep_target': keep_target
        }
        
        # Show transformation summary
        st.success(f"✅ Applied LDA and reduced {len(features)} features to {n_components} components")
        if hasattr(lda, 'explained_variance_ratio_'):
            total_variance = lda.explained_variance_ratio_.sum()
            st.info(f"📊 Total variance explained: {total_variance:.3f} ({total_variance*100:.1f}%)")
        
        # Show final dataset composition
        final_columns = lda_columns.copy()
        if keep_original_features:
            final_columns.extend(keep_original_features)
        if keep_target:
            final_columns.append(target)
        
        st.write(f"**Final dataset contains {len(final_columns)} columns:** {final_columns}")
        
        # Show feature importance
        st.write("**Feature Importance in LDA:**")
        if hasattr(lda, 'coef_'):
            importance_df = pd.DataFrame({
                'Feature': features,
                'Importance': np.abs(lda.coef_[0]) if lda.coef_.shape[0] == 1 else np.abs(lda.coef_).mean(axis=0)
            }).sort_values('Importance', ascending=False)
            st.dataframe(importance_df)
        
    except Exception as e:
        st.error(f"Error during LDA: {str(e)}")
        return df
    
    return df_copy