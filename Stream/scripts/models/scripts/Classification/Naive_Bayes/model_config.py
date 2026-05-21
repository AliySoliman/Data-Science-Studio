import streamlit as st
import pandas as pd
import numpy as np

def model_config(model_data, edit):
    # Check if selected_data exists and has columns
    if 'selected_data' not in st.session_state or st.session_state.selected_data.empty:
        st.error("No data available. Please load data first.")
        return {
            "features": [],
            "target": None,
            "df": pd.DataFrame(),
            "edit": edit,
            "model_type": "gaussian",
            "use_grid_search": True,
            "param_grid": {},
            "manual_params": {},
            "cv_folds": 5
        }
    
    # For Naive Bayes, we can use both numeric and categorical features depending on the type
    all_cols = st.session_state.selected_data.columns.tolist()
    
    # Feature and target selection
    col1, col2 = st.columns(2)
    with col1:
        # Handle default features for edit mode
        default_features = []
        if edit and model_data.get('model param'):
            for param in model_data['model param']:
                if param['name'] == 'features':
                    suggested_features = param['value'] if isinstance(param['value'], list) else []
                    # Safety filter: ensure recommended features exist in current numeric options
                    default_features = [f for f in suggested_features if f in all_cols]
                    break
        
        features = st.multiselect(
            "Select feature columns:",
            options=all_cols,
            default=default_features
        )
    
    with col2:
        if len(all_cols) == 0:
            st.warning("No columns available for target selection")
            target = None
        else:
            # For classification, target can be categorical or numeric
            target_options = [col for col in all_cols if col not in features]
            default_index = 0
            if edit and model_data.get('model param'):
                target_value = None
                for param in model_data['model param']:
                    if param['name'] == 'target':
                        target_value = param['value']
                        break
                
                if target_value and target_value in target_options:
                    default_index = target_options.index(target_value)
            
            target = st.selectbox(
                "Select target column:",
                options=target_options,
                index=default_index,
                key="target_column"
            )
    
    # Model Type Selection
    st.subheader("Model Configuration")
    
    # Default values
    default_model_type = "gaussian"
    default_use_grid_search = True
    default_cv_folds = 5
    
    # Extract parameters from model_data if in edit mode
    if edit and model_data.get('model param'):
        param_dict = {}
        for param in model_data['model param']:
            param_dict[param['name']] = param['value']
        
        default_model_type = param_dict.get('model_type', 'gaussian')
        default_use_grid_search = param_dict.get('use_grid_search', True)
        default_cv_folds = int(param_dict.get('cv_folds', 5))
    
    model_type = st.selectbox(
        "Select Naive Bayes Model:",
        options=['gaussian', 'multinomial', 'bernoulli'],
        index=['gaussian', 'multinomial', 'bernoulli'].index(default_model_type) if default_model_type in ['gaussian', 'multinomial', 'bernoulli'] else 0
    )
    
    # Grid Search Configuration
    use_grid_search = st.checkbox(
        "Use Grid Search for hyperparameter tuning",
        value=default_use_grid_search
    )
    
    if use_grid_search:
        st.info("Grid Search will automatically find the best hyperparameters")
        
        col3, col4 = st.columns(2)
        with col3:
            cv_folds = st.number_input(
                "Cross-validation folds:",
                min_value=2,
                max_value=10,
                value=default_cv_folds
            )
        
        with col4:
            if model_type == 'gaussian':
                var_smoothing_range = st.text_input(
                    "Var smoothing values (comma-separated, scientific notation):",
                    value="1e-9,1e-8,1e-7,1e-6,1e-5"
                )
                try:
                    var_smoothing_values = [float(x.strip()) for x in var_smoothing_range.split(',')]
                except (ValueError, TypeError):
                    var_smoothing_values = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
                    st.warning("Invalid format. Using default var_smoothing values")
                
                param_grid = {
                    'var_smoothing': var_smoothing_values
                }
                
            elif model_type == 'multinomial':
                alpha_range = st.text_input(
                    "Alpha values (comma-separated):",
                    value="0.1,0.5,1.0,2.0,5.0"
                )
                try:
                    alpha_values = [float(x.strip()) for x in alpha_range.split(',')]
                except (ValueError, TypeError):
                    alpha_values = [0.1, 0.5, 1.0, 2.0, 5.0]
                    st.warning("Invalid format. Using default alpha values")
                
                param_grid = {
                    'alpha': alpha_values,
                    'fit_prior': [True, False]
                }
                
            else:  # bernoulli
                alpha_range = st.text_input(
                    "Alpha values (comma-separated):",
                    value="0.1,0.5,1.0,2.0,5.0"
                )
                try:
                    alpha_values = [float(x.strip()) for x in alpha_range.split(',')]
                except (ValueError, TypeError):
                    alpha_values = [0.1, 0.5, 1.0, 2.0, 5.0]
                    st.warning("Invalid format. Using default alpha values")
                
                binarize_range = st.text_input(
                    "Binarize threshold values (comma-separated):",
                    value="0.0,0.1,0.5,1.0"
                )
                try:
                    binarize_values = [float(x.strip()) for x in binarize_range.split(',')]
                except (ValueError, TypeError):
                    binarize_values = [0.0, 0.1, 0.5, 1.0]
                    st.warning("Invalid format. Using default binarize values")
                
                param_grid = {
                    'alpha': alpha_values,
                    'fit_prior': [True, False],
                    'binarize': binarize_values
                }
        
        manual_params = {}
        
    else:
        # Manual parameter configuration
        st.info("Configure hyperparameters manually")
        
        cv_folds = 5
        
        if model_type == 'gaussian':
            col3, col4 = st.columns(2)
            with col3:
                var_smoothing = st.number_input(
                    "Var Smoothing:",
                    min_value=1e-12,
                    max_value=1e-1,
                    value=1e-9,
                    step=1e-10,
                    format="%.12f"
                )
            with col4:
                st.info("Var smoothing stabilizes calculations by adding variance")
            
            manual_params = {
                'var_smoothing': var_smoothing
            }
            param_grid = {}
            
        elif model_type == 'multinomial':
            col3, col4 = st.columns(2)
            with col3:
                alpha = st.number_input(
                    "Alpha (smoothing parameter):",
                    min_value=0.0,
                    max_value=10.0,
                    value=1.0,
                    step=0.1
                )
            with col4:
                fit_prior = st.checkbox("Fit Prior Class Probabilities", value=True)
            
            manual_params = {
                'alpha': alpha,
                'fit_prior': fit_prior
            }
            param_grid = {}
            
        else:  # bernoulli
            col3, col4, col5 = st.columns(3)
            with col3:
                alpha = st.number_input(
                    "Alpha (smoothing parameter):",
                    min_value=0.0,
                    max_value=10.0,
                    value=1.0,
                    step=0.1
                )
            with col4:
                fit_prior = st.checkbox("Fit Prior Class Probabilities", value=True)
            with col5:
                binarize = st.number_input(
                    "Binarize Threshold:",
                    min_value=0.0,
                    max_value=10.0,
                    value=0.0,
                    step=0.1
                )
            
            manual_params = {
                'alpha': alpha,
                'fit_prior': fit_prior,
                'binarize': binarize
            }
            param_grid = {}
    
    return {
        "features": features,
        "target": target,
        "df": st.session_state.selected_data,
        "edit": edit,
        "model_type": model_type,
        "use_grid_search": use_grid_search,
        "param_grid": param_grid,
        "manual_params": manual_params,
        "cv_folds": cv_folds
    }

model_description = """
<div class="code-container">
    <div class="code-header">
        <span>MODEL DESCRIPTION</span>
        <span>Naive Bayes Classifiers</span>
    </div>
    <div>
        Naive Bayes classifiers are probabilistic models based on Bayes' theorem with strong independence 
        assumptions between features. Three variants are available:

        <strong>1. Gaussian Naive Bayes:</strong>
        • Assumes continuous features follow Gaussian distribution
        • Suitable for continuous data with normal distribution
        • Uses mean and variance of each feature per class
        • Key parameter: <strong>var_smoothing</strong> - stabilizes variance calculation

        <strong>2. Multinomial Naive Bayes:</strong>
        • Designed for discrete count data (word counts, frequencies)
        • Commonly used for text classification
        • Models feature counts with multinomial distribution
        • Key parameter: <strong>alpha</strong> - Laplace/Lidstone smoothing

        <strong>3. Bernoulli Naive Bayes:</strong>
        • For binary/boolean features (presence/absence)
        • Models binary feature occurrences
        • Useful for document classification with binary term occurrence
        • Key parameters: <strong>alpha</strong>, <strong>binarize</strong> - threshold for binarization

        <strong>Key Evaluation Metrics:</strong>
        • <strong>Accuracy:</strong> Overall correctness of predictions
        • <strong>Precision:</strong> True positives / (True positives + False positives)
        • <strong>Recall:</strong> True positives / (True positives + False negatives)
        • <strong>F1 Score:</strong> Harmonic mean of precision and recall
        • <strong>Confusion Matrix:</strong> Detailed breakdown of predictions vs actual

        <strong>Hyperparameters:</strong>
        • <strong>alpha:</strong> Additive smoothing parameter (Multinomial/Bernoulli)
        • <strong>var_smoothing:</strong> Variance stabilization (Gaussian)
        • <strong>fit_prior:</strong> Whether to learn class prior probabilities
        • <strong>binarize:</strong> Threshold for feature binarization (Bernoulli)
    </div>
</div>
"""

model_reference_code = """
<div class="code-container">
    <div class="code-header">
        <span>REFERENCE CODE</span>
        <span>Naive Bayes Implementation</span>
    </div>
    <div>
        <pre style="background: #f4f4f4; padding: 10px; border-radius: 5px; overflow-x: auto;">
<code>
# Naive Bayes Implementation
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load and prepare data
data = pd.read_csv('your_data.csv')
X = data[['feature1', 'feature2', 'feature3']]
y = data['target']

# Encode target if categorical
if not np.issubdtype(y.dtype, np.number):
    le = LabelEncoder()
    y = le.fit_transform(y)

# Preprocessing (scale for GaussianNB)
model_type = 'gaussian'  # 'gaussian', 'multinomial', or 'bernoulli'

if model_type == 'gaussian':
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
else:
    X_scaled = X

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

# Model Selection and Grid Search
if model_type == 'gaussian':
    model = GaussianNB()
    param_grid = {
        'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
    }
elif model_type == 'multinomial':
    model = MultinomialNB()
    param_grid = {
        'alpha': [0.1, 0.5, 1.0, 2.0, 5.0],
        'fit_prior': [True, False]
    }
else:  # bernoulli
    model = BernoulliNB()
    param_grid = {
        'alpha': [0.1, 0.5, 1.0, 2.0, 5.0],
        'fit_prior': [True, False],
        'binarize': [0.0, 0.1, 0.5, 1.0]
    }

# With Grid Search
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
cm = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print("Confusion Matrix:")
print(cm)
print("Best parameters:", grid_search.best_params_)
</code>
        </pre>
    </div>
</div>
"""