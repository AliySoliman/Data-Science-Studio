# data_manager.py
import pandas as pd
from typing import Optional
import joblib, base64, json
from io import BytesIO
class DataManager:
    _instance = None
    _current_df: Optional[pd.DataFrame] = None
    app_style = """
    <style>
        .transformation-card {
            border-radius: 8px;
            padding: 12px 16px;
            background-color: #010911;
            border-left: 4px solid #4e79a7;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
        }
        .transformation-card:hover {
            background-color: #440032;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .initials-badge {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            color: white;
            margin-right: 12px;
            flex-shrink: 0;
        }
        .card-content {
            flex-grow: 1;
            min-width: 0;
        }
        .card-title {
            font-weight: 600;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .card-type {
            font-size: 0.8em;
            color: #6c757d;
        }
        .action-button {
            background: none;
            border: none;
            padding: 6px;
            margin: 0 4px;
            cursor: pointer;
            font-size: 16px;
            opacity: 0.7;
            transition: all 0.2s ease;
        }
        .action-button:hover {
            opacity: 1;
            transform: scale(1.1);
        }
        .edit-button {
            color: #4e79a7;
        }
        .delete-button {
            color: #e15759;
        }
        .container-border {
            border: 1px solid #dee2e6;
            border-radius: 10px;
            padding: 16px;
            margin-bottom: 20px;
        }
        .scrollable-list {
                    max-height: 100px;   /* adjust height as you want */
                    overflow-y: auto;
                    padding-right: 10px; /* to avoid overlap with scrollbar */
        }
    </style>
    """
    label_style = """
            <style>
                .custom-container {
                    background-color: #112343;  /* container background */
                    padding: 20px;
                    border-radius: 30px;
                    margin-bottom: 10px;
                    width : 70%;
                }
                .custom-header {
                    color: white;               /* header text color */
                    font-size: 24px;
                    font-weight: bold;
                    margin: 0;                  /* remove default margin */
                }
            </style>
            """

        # Example container with header
    label_data ="""
            <div class="custom-container">
                <h2 class="custom-header">🚀 Dynamic Table with Transformation Pipeline</h2>
            </div>
            """
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DataManager, cls).__new__(cls)
        return cls._instance
    
    @property
    def current_df(self) -> pd.DataFrame:
        if self._current_df is None:
            raise ValueError("No DataFrame has been loaded yet")
        return self._current_df
    
    @current_df.setter
    def current_df(self, df: pd.DataFrame):
        self._current_df = df.copy()
    
    def get_columns(self, dtype=None) -> list:
        if dtype:
            return [col for col in self.current_df.columns if self.current_df[col].dtype == dtype]
        return list(self.current_df.columns)
    
    def get_sample(self, n=5) -> pd.DataFrame:
        return self.current_df.head(n)
    

        

    #
        # STRUCTURE 
        #  {
        #     "model name":LRM,
        #     "model type" : Linear Regression Model,
        #     "model param" : 
        #     [
        #         {
        #             "name":features,
        #             "value" : age
        #         },
        #         {
        #             "name":target,
        #             "value" : Blood Pressure
        #         },
        #     ]
        
        # } 
    #
    @staticmethod
    def create_Linear_REG_Model(model_name, model_param,trans) -> dict:
        return {
            "model name": model_name,
            "model type": "Simple Linear Regression",
            "model param": model_param,
            "transformations" : trans
        }
    @staticmethod
    def create_REG_Model(model_name, model_param,trans,model,metrics_snapshot) -> dict:
        buffer = BytesIO()
        joblib.dump(model, buffer)
        buffer.seek(0)
        model_b64 = base64.b64encode(buffer.read()).decode('utf-8')
        return {
            "model name": model_name,
            "model type": "Linear Regression",
            "model param": model_param,
            "transformations" : trans,
            "model":model_b64,
            "metrics_snapshot": metrics_snapshot  # ADD THIS LINE

        }
    @staticmethod
    def create_KMeans_Model(model_name, model_param,trans) -> dict:
        return {
            "model name": model_name,
            "model type": "KMeans Clustering",
            "model param": model_param,
            "transformations" : trans,
            "metrics_snapshot": {}  # ADD THIS LINE
            
        }
######################################################################################################################
# Ragab

    @staticmethod
    def create_KNN_Model(model_name, model_param, trans,model,metrics_snapshot):
        # Ensure proper parameter types when creating the model
        processed_params = []
        buffer = BytesIO()
        joblib.dump(model, buffer)
        buffer.seek(0)
        model_b64 = base64.b64encode(buffer.read()).decode('utf-8')
        for param in model_param:
            # Convert numeric values to appropriate types
            if param['name'] in ['n_neighbors_range', 'cv_folds']:
                try:
                    if param['name'] == 'cv_folds':
                        param['value'] = int(param['value'])
                    elif isinstance(param['value'], str) and ',' in param['value']:
                        # It's a list, keep as string representation
                        pass
                    else:
                        param['value'] = int(param['value'])
                except (ValueError, TypeError):
                    # Keep original value if conversion fails
                    pass
            processed_params.append(param)
        
        return {
            "model name": model_name,
            "model type": "KNN Classifier",
            "model param": processed_params,
            "transformations": trans,
            "model": model_b64,
            "metrics_snapshot":metrics_snapshot
        }
    
# **************************************add this part**************************************************************************
    @staticmethod
    def create_Logistic_Regression_Model(model_name, model_param, trans):
        # Ensure proper parameter types
        processed_params = []
        for param in model_param:
            # Handle max_iter conversion
            if param['name'] == 'max_iter' and isinstance(param['value'], str) and param['value'].startswith('['):
                try:
                    import re
                    numbers = re.findall(r'\d+', param['value'])
                    param['value'] = int(numbers[0]) if numbers else 100
                except (ValueError, TypeError):
                    param['value'] = 100
            processed_params.append(param)
        
        return {
            "model name": model_name,
            "model type": "Logistic Regression",
            "model param": processed_params,
            "transformations": trans
        }
    # ****************************************************************************************************************

    @staticmethod
    def create_DecisionTree_Regressor_Model(model_name, model_param, trans, metrics_snapshot=None):
        # Handle parameter type conversions
        processed_params = []
        for param in model_param:
            # Handle numeric parameter conversions
            if param['name'] in ['max_depth', 'min_samples_split', 'cv_folds']:
                try:
                    if isinstance(param['value'], str) and param['value'].startswith('['):
                        # It's a list, keep as string
                        pass
                    else:
                        param['value'] = int(param['value'])
                except (ValueError, TypeError):
                    pass
            processed_params.append(param)
        
        return {
            "model name": model_name,
            "model type": "Decision Tree Regressor",
            "model param": processed_params,
            "transformations": trans,
            "metrics_snapshot": metrics_snapshot or {}
        }
###################################################################################################################################################
# Nouh
    @staticmethod
    def create_RF_Model(model_name, model_param, trans, model, metrics_snapshot):
        buffer = BytesIO()
        joblib.dump(model, buffer)
        buffer.seek(0)
        model_b64 = base64.b64encode(buffer.read()).decode('utf-8')
        # Handle parameter type conversions for Random Forest
        processed_params = []
        for param in model_param:
            # Handle numeric parameter conversions
            if param['name'] in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'cv_folds']:
                try:
                    if isinstance(param['value'], str) and param['value'].startswith('['):
                        # It's a list, keep as string representation
                        pass
                    elif param['value'] == 'None':
                        param['value'] = None
                    else:
                        param['value'] = int(param['value'])
                except (ValueError, TypeError):
                    pass
            # Handle boolean parameters
            elif param['name'] == 'bootstrap':
                if isinstance(param['value'], str):
                    param['value'] = param['value'].lower() == 'true'
            processed_params.append(param)
        
        return {
            "model name": model_name,
            "model type": "Random Forest Classifier",
            "model param": processed_params,
            "transformations": trans,
            "metrics_snapshot":metrics_snapshot,
            "model":model_b64

        }

    @staticmethod
    def create_gbc_Model(model_name, model_param, trans,model,metrics_snapshot):  
        buffer = BytesIO()
        joblib.dump(model, buffer)
        buffer.seek(0)
        model_b64 = base64.b64encode(buffer.read()).decode('utf-8')
        return {
            "model name": model_name,
            "model type": "Gradient Boosting Classifier",
            "model param": model_param,
            "transformations": trans,
            "model": model_b64,
            "metrics_snapshot": metrics_snapshot
        }
    
    @staticmethod
    def create_UnsupKNN_Model(model_name, model_param, trans):
        # Handle parameter type conversions for Unsupervised KNN
        processed_params = []
        for param in model_param:
            # Handle numeric parameter conversions
            if param['name'] in ['n_neighbors_range', 'n_clusters']:
                try:
                    if isinstance(param['value'], str) and param['value'].startswith('['):
                        # It's a list, keep as string representation
                        pass
                    else:
                        param['value'] = int(param['value'])
                except (ValueError, TypeError):
                    pass
            processed_params.append(param)
        
        return {
            "model name": model_name,
            "model type": "Unsupervised KNN Clustering",
            "model param": processed_params,
            "transformations": trans
        }
################################################################################################################################################################################################################################################################################################################
# Ali
    @staticmethod
    def create_Gradient_Boosting_REG_Model(model_name, model_param, trans, metrics_snapshot):
    # Ensure proper parameter types
        processed_params = []
        for param in model_param:
            # Handle n_estimators conversion
            if param['name'] == 'n_estimators' and isinstance(param['value'], str) and param['value'].startswith('['):
                try:
                    import re
                    numbers = re.findall(r'\d+', param['value'])
                    param['value'] = int(numbers[0]) if numbers else 100
                except (ValueError, TypeError):
                    param['value'] = 100
            # Handle learning_rate conversion
            elif param['name'] == 'learning_rate' and isinstance(param['value'], str):
                try:
                    param['value'] = float(param['value'])
                except (ValueError, TypeError):
                    param['value'] = 0.1
            # Handle max_depth conversion
            elif param['name'] == 'max_depth' and isinstance(param['value'], str):
                try:
                    param['value'] = int(param['value'])
                except (ValueError, TypeError):
                    param['value'] = 3
            processed_params.append(param)
    
        return {
            "model name": model_name,
            "model type": "Gradient Boosting Regressor", 
            "model param": processed_params,
            "transformations": trans,
            "metrics_snapshot": metrics_snapshot  # Only store JSON-safe metrics
            # No model stored in pipeline - only in session_state
        }
    @staticmethod
    def create_DBSCAN_Model(model_name, model_param, trans, metrics_snapshot):
        # Ensure proper parameter types
        processed_params = []
        for param in model_param:
            # Handle eps conversion
            if param['name'] == 'eps' and isinstance(param['value'], str):
                try:
                    param['value'] = float(param['value'])
                except (ValueError, TypeError):
                    param['value'] = 0.5
            # Handle min_samples conversion
            elif param['name'] == 'min_samples' and isinstance(param['value'], str):
                try:
                    param['value'] = int(param['value'])
                except (ValueError, TypeError):
                    param['value'] = 5
            # Handle metric conversion (ensure it's a string)
            elif param['name'] == 'metric' and not isinstance(param['value'], str):
                param['value'] = 'euclidean'
            # Handle grid search parameter conversions
            elif param['name'] == 'eps_range' and isinstance(param['value'], str):
                try:
                    param['value'] = [float(x.strip()) for x in param['value'].strip('[]').split(',')]
                except (ValueError, TypeError):
                    param['value'] = [0.3, 0.5, 0.7, 1.0]
            elif param['name'] == 'min_samples_range' and isinstance(param['value'], str):
                try:
                    param['value'] = [int(x.strip()) for x in param['value'].strip('[]').split(',')]
                except (ValueError, TypeError):
                    param['value'] = [3, 5, 7, 10]
            processed_params.append(param)
        
        return {
            "model name": model_name,
            "model type": "DBSCAN",
            "model param": processed_params,
            "transformations": trans,
            "metrics_snapshot": metrics_snapshot  # Add this line to match KNN/GBR
        }
######################################################################################################################
# Ahmed Hisham        
    @staticmethod
    def create_NB_Model(model_name, model_param, trans,metrics_snapshot):
        # Handle parameter type conversions for Naive Bayes
        processed_params = []
        for param in model_param:
            # Handle numeric parameter conversions
            if param['name'] in ['cv_folds']:
                try:
                    param['value'] = int(param['value'])
                except (ValueError, TypeError):
                    pass
            # Handle float parameters for alpha and var_smoothing
            elif param['name'] in ['alpha', 'var_smoothing', 'binarize']:
                try:
                    if isinstance(param['value'], str) and param['value'].startswith('['):
                        # It's a list, keep as string representation
                        pass
                    else:
                        param['value'] = float(param['value'])
                except (ValueError, TypeError):
                    pass
            # Handle boolean parameters
            elif param['name'] == 'fit_prior':
                if isinstance(param['value'], str):
                    param['value'] = param['value'].lower() == 'true'
            processed_params.append(param)
        
        return {
            "model name": model_name,
            "model type": "Naive Bayes Classifier",
            "model param": processed_params,
            "transformations": trans,
            "metrics_snapshot":metrics_snapshot

        }
    @staticmethod
    def create_SVM_Model(model_name, model_param, trans, metrics_snapshot):
        # Handle parameter type conversions for Support Vector Machine
        processed_params = []
        for param in model_param:
            # Handle numeric parameter conversions
            if param['name'] in ['cv_folds', 'degree']:
                try:
                    if isinstance(param['value'], str) and param['value'].startswith('['):
                        # It's a list, keep as string representation
                        pass
                    else:
                        param['value'] = int(param['value'])
                except (ValueError, TypeError):
                    pass
            # Handle float parameters for C
            elif param['name'] == 'C':
                try:
                    if isinstance(param['value'], str) and param['value'].startswith('['):
                        # It's a list, keep as string representation
                        pass
                    else:
                        param['value'] = float(param['value'])
                except (ValueError, TypeError):
                    pass
            # Handle gamma parameter (can be string 'scale', 'auto' or float)
            elif param['name'] == 'gamma':
                if isinstance(param['value'], str) and not param['value'].startswith('['):
                    if param['value'] not in ['scale', 'auto']:
                        try:
                            param['value'] = float(param['value'])
                        except (ValueError, TypeError):
                            param['value'] = 'scale'
            # Handle kernel parameter (ensure it's a valid string)
            elif param['name'] == 'kernel':
                if not isinstance(param['value'], str):
                    param['value'] = 'rbf'
                elif param['value'] not in ['linear', 'poly', 'rbf', 'sigmoid']:
                    param['value'] = 'rbf'
            processed_params.append(param)
        
        return {
            "model name": model_name,
            "model type": "Support Vector Machine Classifier",
            "model param": processed_params,
            "transformations": trans,
            "metrics_snapshot":metrics_snapshot
        }    
# ****************************************************************
# Hager
    @staticmethod
    def create_DecisionTree_Model(model_name, model_param, trans, model, metrics_snapshot=None) -> dict:
        buffer = BytesIO()
        joblib.dump(model, buffer)
        buffer.seek(0)
        model_b64 = base64.b64encode(buffer.read()).decode('utf-8')
        return {
            "model name": model_name,
            "model type": "Decision Tree Classifier",
            "model param": model_param,
            "transformations" : trans,
            "model":model_b64,
            "metrics_snapshot": metrics_snapshot or {}
        }
# ****************************************************************

    def create_KNN_Regressor_Model(model_name, model_param,trans,model,metrics_snapshot) -> dict:
        buffer = BytesIO()
        joblib.dump(model, buffer)
        buffer.seek(0)
        model_b64 = base64.b64encode(buffer.read()).decode('utf-8')
        return {
            "model name": model_name,
            "model type": "KNN_Regressor",
            "model param": model_param,
            "transformations" : trans,
            "model":model_b64,
            "metrics_snapshot":metrics_snapshot


        }
# )***************************************************************************88
# Ahmed Hassan
    @staticmethod    
    def create_Hierarchical_Clustering_Model(model_name, model_param,trans,model,metrics_snapshot) -> dict:
        buffer = BytesIO()
        joblib.dump(model, buffer)
        buffer.seek(0)
        model_b64 = base64.b64encode(buffer.read()).decode('utf-8')
        return {
            "model name": model_name,
            "model type": "Hierarchical Clustering",
            "model param": model_param,
            "transformations" : trans,
            "model":model_b64,
            "metrics_snapshot":metrics_snapshot
            
            }
    
    @staticmethod
    def create_RandomForest_REG_Model(model_name, model_param,trans,model,metrics_snapshot) -> dict:
        buffer = BytesIO()
        joblib.dump(model, buffer)
        buffer.seek(0)
        model_b64 = base64.b64encode(buffer.read()).decode('utf-8')
        return {
            "model name": model_name,
            "model type": "Random Forest Regression",
            "model param": model_param,
            "transformations" : trans,
            "model":model_b64,
            "metrics_snapshot":metrics_snapshot
            
            }

    @staticmethod
    def create_XGBoost_REG_Model(model_name, model_param, trans, model, metrics_snapshot) -> dict:
        buffer = BytesIO()
        joblib.dump(model, buffer)
        buffer.seek(0)
        model_b64 = base64.b64encode(buffer.read()).decode('utf-8')
        return {
            "model name": model_name,
            "model type": "XGBoost Regressor",
            "model param": model_param,
            "transformations": trans,
            "model": model_b64,
            "metrics_snapshot": metrics_snapshot
        }

    @staticmethod
    def create_XGBoost_CLS_Model(model_name, model_param, trans, model, metrics_snapshot) -> dict:
        buffer = BytesIO()
        joblib.dump(model, buffer)
        buffer.seek(0)
        model_b64 = base64.b64encode(buffer.read()).decode('utf-8')
        return {
            "model name": model_name,
            "model type": "XGBoost Classifier",
            "model param": model_param,
            "transformations": trans,
            "model": model_b64,
            "metrics_snapshot": metrics_snapshot
        }