import streamlit as st

def model_config(model_data, edit):
    st.subheader("Support Vector Regression Configuration")
    
    if 'selected_data' in st.session_state and not st.session_state.selected_data.empty:
        df = st.session_state.selected_data
        all_columns = df.columns.tolist()
        
        default_features = model_data.get('features', []) if edit else []
        default_target = model_data.get('target', all_columns[0]) if edit else all_columns[0]
        
        selected_features = st.multiselect("Select Feature Columns", all_columns, default=default_features)
        selected_target = st.selectbox("Select Target Column", all_columns, index=all_columns.index(default_target) if default_target in all_columns else 0)
        
        use_grid_search = st.checkbox("Use Grid Search for Optimization", value=model_data.get('use_grid_search', False))
        
        manual_params = {}
        param_grid = {}
        cv_folds = 5
        
        if use_grid_search:
            st.write("Grid Search Parameters:")
            cv_folds = st.slider("CV Folds", 2, 10, int(model_data.get('cv_folds', 5)))
            param_grid = {
                'C': st.multiselect("Select C range", [0.1, 1, 10, 100], default=model_data.get('param_grid', {}).get('C', [1, 10])),
                'kernel': st.multiselect("Select Kernels", ['rbf', 'linear', 'poly'], default=model_data.get('param_grid', {}).get('kernel', ['rbf'])),
                'epsilon': st.multiselect("Select Epsilon range", [0.01, 0.1, 0.2, 0.5], default=model_data.get('param_grid', {}).get('epsilon', [0.1]))
            }
        else:
            st.write("Manual Hyperparameters:")
            manual_params['C'] = st.number_input("C (Regularization)", 0.01, 1000.0, float(model_data.get('manual_params', {}).get('C', 1.0)))
            manual_params['kernel'] = st.selectbox("Kernel", ['rbf', 'linear', 'poly', 'sigmoid'], index=['rbf', 'linear', 'poly', 'sigmoid'].index(model_data.get('manual_params', {}).get('kernel', 'rbf')))
            manual_params['gamma'] = st.selectbox("Gamma", ['scale', 'auto'], index=['scale', 'auto'].index(model_data.get('manual_params', {}).get('gamma', 'scale')))
            manual_params['epsilon'] = st.number_input("Epsilon", 0.0, 1.0, float(model_data.get('manual_params', {}).get('epsilon', 0.1)))
            
        return {
            "df": df,
            "features": selected_features,
            "target": selected_target,
            "edit": edit,
            "use_grid_search": use_grid_search,
            "param_grid": param_grid,
            "manual_params": manual_params,
            "cv_folds": cv_folds
        }
    else:
        st.warning("Please load or transform data before configuring the model.")
        return None

def model_description():
    return """
    **Support Vector Regression (SVR)** is a type of Support Vector Machine that is used for regression problems.
    It tries to find a function that deviates from the actual observed targets by no more than a certain value (epsilon).
    """

def model_reference_code():
    return """
from sklearn.svm import SVR
model = SVR(C=1.0, epsilon=0.2)
model.fit(X_train, y_train)
    """
