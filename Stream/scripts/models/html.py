model_reference_code = """
                <div class="code-container">
                    <div class="code-header">
                        <span>MODEL DESCRIPTION</span>
                        <span>Linear Regression</span>
                    </div>
                    <div>
                            # Prepare data
                            X = df[features].values
                            y = df[target].values
                            
                            # Split data
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                            
                            # Train model
                            model = LinearRegression()
                            model.fit(X_train, y_train)
                            
                            # Make predictions
                            y_pred = model.predict(X_test)
                            
                            # Calculate metrics
                            mse = mean_squared_error(y_test, y_pred)
                            r2 = r2_score(y_test, y_pred)
                            
                            # Store results
                            st.session_state.model_results = {
                                'model': model,
                                'metrics': {
                                    'MSE': mse,
                                    'R2 Score': r2,
                                    'Coefficients': model.coef_,
                                    'Intercept': model.intercept_
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
                    <span>Linear Regression</span>
                </div>
                <div>
                    Linear regression models the relationship between a dependent variable and one or more 
                    independent variables by fitting a linear equation to observed data. The model assumes 
                    a linear relationship between the input variables (x) and the single output variable (y).
                    
                    Equation: y = β₀ + β₁x₁ + ... + βₙxₙ
                    Where:
                    - y is the predicted value
                    - β₀ is the bias term
                    - β₁...βₙ are the weights for each feature
                    - x₁...xₙ are the input features
                </div>
            </div>
            """
def editor(html_comments):
    return f"""
                <div class="code-container">
                    <div class="code-header">
                        <span>MODEL DESCRIPTION</span>
                        <span>Linear Regression</span>
                    </div>
                    <div>{html_comments or "<em>No highlights yet.</em>"}</div>
                </div>
                """