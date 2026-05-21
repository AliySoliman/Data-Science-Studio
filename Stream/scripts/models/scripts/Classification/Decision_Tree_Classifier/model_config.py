import streamlit as st
def model_config(model_data,edit):        
        # Get ALL columns for feature and target selection (numerical and categorical)
        all_cols = st.session_state.selected_data.columns.tolist() 
        
        # Target options: any column not currently selected as a feature
        # (Classification models handle all types via LabelEncoder/OneHot)
        
        # Feature and target selection
        col1, col2 = st.columns(2)
        with col1:
            # Features remain ALL columns (numerical and categorical)
            features = st.multiselect(
                "Select feature columns (Numerical or Categorical):",
                options=all_cols if len(all_cols) != 0 else [] ,
                default=model_data['model param'][0]['value'] if edit else []
            )
        
        with col2:
            target_options = [col for col in all_cols if col not in features]
            if len(target_options) == 0:
                st.warning("No columns available for target selection.")
                target = None
                use_grid_search = False 
            else:
                # Determine the default target value for selection
                default_target_value = model_data['model param'][1]['value'] if edit else target_options[-1]
                
                # Find the index of the default target value
                try:
                    default_index = target_options.index(default_target_value)
                except ValueError:
                    default_index = 0
                    
                target = st.selectbox(
                    "Select target column (Categorical/Numeric):",
                    options=target_options,
                    index=default_index,
                    key="target_column_dt"
                )    
                
                # --- NEW MODIFICATIONS START HERE ---
                
                # 1. Check for existing 'use_grid_search' value in model_data for edit mode
                default_grid_search = next(
                    (p['value'] for p in model_data['model param'] if p['name'] == 'use_grid_search'),
                    False # Default to False if parameter not found
                ) if edit else False
                
                # 2. Add the Streamlit Checkbox
                use_grid_search = st.checkbox(
                    "Use Grid Search for optimal hyperparameters",
                    value=default_grid_search,
                    key="use_grid_search_checkbox"
                )
                max_depth = None
                min_samples_leaf = None
                random_state = None
        if not use_grid_search :
            max_depth = int(st.text_input(label = 'max_depth',value =5))
            min_samples_leaf = int(st.text_input(label = 'min_samples_leaf',value =10))
            random_state =int(st.text_input(label = 'random_state',value = 42))

                
        # 3. Return the new parameter in the output dictionary
        if target is None:
             # Return only features and target/df/edit if target is None (empty data)
            return {"features":features,"target":target,"df":st.session_state.selected_data,"edit":edit, "use_grid_search": False}
        else:
            return {"features":features,"target":target,"df":st.session_state.selected_data,"edit":edit, "use_grid_search": use_grid_search,"max_depth":max_depth,"min_samples_leaf":min_samples_leaf,"random_state":random_state}
# Decision Tree model reference code
model_reference_code = """
                <div class="code-container">
                    <div class="code-header">
                        <span>MODEL DESCRIPTION</span>
                        <span>Decision Tree Classifier</span>
                    </div>
                    <div>
                            # Prepare data
                            X = df[features].values
                            y = df[target].values
                            
                            # Split data
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                            
                            # Train model
                            model = DecisionTreeClassifier(random_state=42)
                            model.fit(X_train, y_train)
                            
                            # Make predictions
                            y_pred = model.predict(X_test)
                            
                            # Calculate metrics
                            accuracy = accuracy_score(y_test, y_pred)
                            precision = precision_score(y_test, y_pred, average='weighted')
                            recall = recall_score(y_test, y_pred, average='weighted')
                            f1 = f1_score(y_test, y_pred, average='weighted')
                            conf_matrix = confusion_matrix(y_test, y_pred)
                            
                            # Store results
                            st.session_state.model_results = {
                                'model': model,
                                'metrics': {
                                    'Accuracy': accuracy,
                                    'Precision': precision,
                                    'Recall': recall,
                                    'F1 Score': f1,
                                    'Confusion Matrix': conf_matrix
                                },
                                'features': features,
                                'target': target
                            }
                            </div>
                </div>
                """

model_description = """
            <div class="code-container">
                <div class="code-header">
                    <span>MODEL DESCRIPTION</span>
                    <span>Decision Tree Classifier</span>
                </div>
                <div>
                    A Decision Tree Classifier is a supervised learning algorithm used for classification tasks. 
                    It creates a tree-like model of decisions and their possible consequences. The algorithm 
                    recursively partitions the data based on feature values to maximize information gain or 
                    minimize impurity at each node.
                    
                    Key concepts:
                    - Root Node: Represents the entire dataset
                    - Decision Nodes: Sub-nodes that split the data
                    - Leaf Nodes: Terminal nodes representing final classifications
                    - Splitting: Dividing nodes into sub-nodes based on feature conditions
                    - Pruning: Reducing tree size to prevent overfitting
                    
                    Advantages:
                    - Easy to interpret and visualize
                    - Handles both numerical and categorical data
                    - Requires little data preprocessing
                </div>
            </div>
            """