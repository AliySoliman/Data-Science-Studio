#----------------------------------------------------------Classification----------------------------------------------------------
from models.scripts.Classification.Decision_Tree_Classifier.model_script import model_script as dt_script, validate_model as dt_validate
from models.scripts.Classification.Decision_Tree_Classifier.model_report import model_report as dt_report
from models.scripts.Classification.Decision_Tree_Classifier.model_config import model_config as dt_config, model_description, model_reference_code

from models.scripts.Classification.Gradient_Boosting_Classifier.model_script import model_script as gbc_script, validate_model as gbc_validate
from models.scripts.Classification.Gradient_Boosting_Classifier.model_report import model_report as gbc_report
from models.scripts.Classification.Gradient_Boosting_Classifier.model_config import model_config as gbc_config, model_description, model_reference_code

from models.scripts.Classification.Knn_Classifier.model_script import model_script as knnc_script, validate_model as knnc_validate
from models.scripts.Classification.Knn_Classifier.model_report import model_report as knnc_report
from models.scripts.Classification.Knn_Classifier.model_config import model_config as knnc_config, model_description, model_reference_code

from models.scripts.Classification.Logistic_Regression.model_script import model_script as lr_script, validate_model as lr_validate
from models.scripts.Classification.Logistic_Regression.model_report import model_report as lr_report
from models.scripts.Classification.Logistic_Regression.model_config import model_config as lr_config, model_description, model_reference_code

from models.scripts.Classification.Naive_Bayes.model_script import model_script as nb_script, validate_model as nb_validate
from models.scripts.Classification.Naive_Bayes.model_report import model_report as nb_report
from models.scripts.Classification.Naive_Bayes.model_config import model_config as nb_config, model_description, model_reference_code

from models.scripts.Classification.Random_Forest_Classifier.model_script import model_script as rf_script, validate_model as rf_validate
from models.scripts.Classification.Random_Forest_Classifier.model_report import model_report as rf_report
from models.scripts.Classification.Random_Forest_Classifier.model_config import model_config as rf_config, model_description, model_reference_code

from models.scripts.Classification.SVC.model_script import model_script as svc_script, validate_model as svc_validate
from models.scripts.Classification.SVC.model_report import model_report as svc_report
from models.scripts.Classification.SVC.model_config import model_config as svc_config, model_description, model_reference_code

from models.scripts.Classification.Xgboost_Classifier.model_script import model_script as xgbc_script, validate_model as xgbc_validate
from models.scripts.Classification.Xgboost_Classifier.model_report import model_report as xgbc_report
from models.scripts.Classification.Xgboost_Classifier.model_config import model_config as xgbc_config, model_description, model_reference_code


#----------------------------------------------------------Regression----------------------------------------------------------
from models.scripts.Regression.Gradient_Boosting_Regressor.model_script import model_script as gbr_script, validate_model as gbr_validate
from models.scripts.Regression.Gradient_Boosting_Regressor.model_report import model_report as gbr_report
from models.scripts.Regression.Gradient_Boosting_Regressor.model_config import model_config as gbr_config, model_description, model_reference_code

from models.scripts.Regression.Knn_Regressor.model_script import model_script as knn_reg_script, validate_model as knn_reg_validate
from models.scripts.Regression.Knn_Regressor.model_report import model_report as knn_reg_report
from models.scripts.Regression.Knn_Regressor.model_config import model_config as knn_reg_config, model_description, model_reference_code

from models.scripts.Regression.Random_Forest_Regressor.model_script import model_script as rf_reg_script, validate_model as rf_reg_validate
from models.scripts.Regression.Random_Forest_Regressor.model_report import model_report as rf_reg_report
from models.scripts.Regression.Random_Forest_Regressor.model_config import model_config as rf_reg_config, model_description, model_reference_code

from models.scripts.Regression.Simple_Linear_Regression.model_script import model_script as slr_script, validate_model as slr_validate
from models.scripts.Regression.Simple_Linear_Regression.model_report import model_report as slr_report
from models.scripts.Regression.Simple_Linear_Regression.model_config import model_config as slr_config, model_description, model_reference_code

from models.scripts.Regression.SVR.model_script import model_script as svr_script, validate_model as svr_validate
from models.scripts.Regression.SVR.model_report import model_report as svr_report
from models.scripts.Regression.SVR.model_config import model_config as svr_config, model_description, model_reference_code

from models.scripts.Regression.Xgboost_Regressor.model_script import model_script as xgb_reg_script, validate_model as xgb_reg_validate
from models.scripts.Regression.Xgboost_Regressor.model_report import model_report as xgb_reg_report
from models.scripts.Regression.Xgboost_Regressor.model_config import model_config as xgb_reg_config, model_description, model_reference_code


#----------------------------------------------------------Unsupervised----------------------------------------------------------
from models.scripts.Unsupervised.DBSCAN.model_script import model_script as dbscan_script, validate_model as dbscan_validate
from models.scripts.Unsupervised.DBSCAN.model_report import model_report as dbscan_report
from models.scripts.Unsupervised.DBSCAN.model_config import model_config as dbscan_config, model_description, model_reference_code


from models.scripts.Unsupervised.Hierarchical_Clustering.model_script import model_script as hierarchy_script, validate_model as hierarchy_validate_model
from models.scripts.Unsupervised.Hierarchical_Clustering.model_report import model_report as hierarchy_report
from models.scripts.Unsupervised.Hierarchical_Clustering.model_config import model_config as hierarchy_config, model_description, model_reference_code

from models.scripts.Unsupervised.Kmeans.model_script import model_script as kmeans_script, validate_model as kmeans_validate
from models.scripts.Unsupervised.Kmeans.model_report import model_report as kmeans_report
from models.scripts.Unsupervised.Kmeans.model_config import model_config as kmeans_config, model_description, model_reference_code


from models.scripts.Unsupervised.Unsubervised_Knn.model_script import model_script as unsupervised_knn_script, validate_model as unsupervised_knn_validate
from models.scripts.Unsupervised.Unsubervised_Knn.model_report import model_report as unsupervised_knn_report
from models.scripts.Unsupervised.Unsubervised_Knn.model_config import model_config as unsupervised_knn_config, model_description, model_reference_code


def execute_model(model_name,action,param_dict):

    MODELS = {

#----------------------------------------------------------Classification----------------------------------------------------------


        "Decision Tree Classifier": {
                "script":dt_script ,
                "report":dt_report ,
                "config": dt_config,
                "validate":dt_validate,
                "script_params": ["df", "features", "edit", "target", "use_grid_search","max_depth","min_samples_leaf","random_state"],
                "config_params": ["model_data", "edit"],
                "validate_params": ["params"]
            },


        "Gradient Boosting Classifier": {
                "script": gbc_script,
                "report": gbc_report,
                "config": gbc_config,
                "validate": gbc_validate,
                "script_params": ["df", "features", "target", "edit", "use_grid_search", "param_grid", "manual_params", "cv_folds"],
                "config_params": ["model_data", "edit"],
                "validate_params": ["params"]
            },


        "KNN Classifier": {
                "script": knnc_script,
                "report": knnc_report,
                "config": knnc_config,
                "validate": knnc_validate,
                "script_params": ["df", "features", "target", "edit", "use_grid_search", "param_grid", "manual_params", "cv_folds"],
                "config_params": ["model_data", "edit"],
                "validate_params": ["params"],

            },

        "Logistic Regression": {
                "script": lr_script,
                "report": lr_report,
                "config": lr_config,
                "validate": lr_validate,
                "script_params": ["df", "features", "target", "edit", "use_grid_search", "param_grid", "manual_params", "cv_folds", "solver", "max_iter"],
                "config_params": ["model_data", "edit"],
                "validate_params": ["params"],

            },


        "Naive Bayes Classifier": {
                "script": nb_script,
                "report": nb_report,
                "config": nb_config,
                "validate": nb_validate,
                "script_params": ["df", "features", "target", "edit", "model_type", "use_grid_search", "param_grid", "manual_params", "cv_folds"],
                "config_params": ["model_data", "edit"],
                "validate_params": ["params"]
            },




        "Random Forest Classifier": {
                "script": rf_script,
                "report": rf_report,
                "config": rf_config,
                "validate": rf_validate,
                "script_params": ["df", "features", "target", "edit", "use_grid_search", "param_grid", "manual_params", "cv_folds"],
                "config_params": ["model_data", "edit"],
                "validate_params": ["params"]
            },


        "Support Vector Machine Classifier": {
                "script": svc_script,
                "report": svc_report,
                "config": svc_config,
                "validate": svc_validate,
                "script_params": ["df", "features", "target", "edit", "use_grid_search", "param_grid", "manual_params", "cv_folds"],
                "config_params": ["model_data", "edit"],
                "validate_params": ["params"]
            },



        "XGBoost Classifier": {
                "script": xgbc_script,
                "report": xgbc_report,
                "config": xgbc_config,
                "validate": xgbc_validate,
                "script_params": ["df", "features", "target", "edit", "use_grid_search", "param_grid", "manual_params", "cv_folds"],
                "config_params": ["model_data", "edit"],
                "validate_params": ["params"],
            },


#----------------------------------------------------------Regression----------------------------------------------------------



        "Gradient Boosting Regressor": {
                "script": gbr_script,
                "report": gbr_report,
                "config": gbr_config,
                "validate": gbr_validate,
                "script_params": ["df", "features", "target", "edit", "use_grid_search", "param_grid", "manual_params", "cv_folds"],
                "config_params": ["model_data", "edit"],
                "validate_params": ["params"],
        },




        "KNN_Regressor": {
                "script": knn_reg_script,
                "report": knn_reg_report,
                "config": knn_reg_config,
                "validate": knn_reg_validate,
                "script_params": ["df", "features", "target", "edit", "use_grid_search", "param_grid", "manual_params", "cv_folds"],
                "config_params": ["model_data", "edit"],
                "validate_params": ["params"]
            },


        "Random Forest Regression": {
                "script": rf_reg_script,
                "report": rf_reg_report,
                "config":rf_reg_config,
                "validate":rf_reg_validate,
                "script_params": ["df", "features", "target", "edit", "use_grid_search", "param_grid", "manual_params", "cv_folds"],
                "config_params":["model_data","edit"],
                "validate_params":["params"],

            },


        "Linear Regression": {
                "script": slr_script,
                "report": slr_report,
                "config": slr_config,
                "validate": slr_validate,
                "script_params": ["df", "features", "target", "edit", "model_type", "alpha"],
                "config_params": ["model_data", "edit"],
                "validate_params": ["params"],
            },


        "Support Vector Regression": {
                "script": svr_script,
                "report": svr_report,
                "config": svr_config,
                "validate": svr_validate,
                "script_params": ["df", "features", "target", "edit", "use_grid_search", "param_grid", "manual_params", "cv_folds"],
                "config_params": ["model_data", "edit"],
                "validate_params": ["params"]
            },

        "XGBoost Regressor": {
                "script": xgb_reg_script,
                "report": xgb_reg_report,
                "config": xgb_reg_config,
                "validate": xgb_reg_validate,
                "script_params": ["df", "features", "target", "edit", "use_grid_search", "param_grid", "manual_params", "cv_folds"],
                "config_params": ["model_data", "edit"],
                "validate_params": ["params"],
            },

#----------------------------------------------------------Unsupervised----------------------------------------------------------


        "DBSCAN": {
                "script": dbscan_script,
                "report": dbscan_report,
                "config": dbscan_config,
                "validate": dbscan_validate,
                "script_params": ["df", "features", "edit", "use_grid_search", "param_grid", "manual_params", "cv_folds"],
                "config_params": ["model_data", "edit"],
                "validate_params": ["params"],
            },
            
        

        "Hierarchical Clustering": {
                "script": hierarchy_script,
                "report": hierarchy_report,
                "config":hierarchy_config,
                "validate":hierarchy_validate_model,
                "script_params": ["df", "features", "n_clusters", "linkage", "metric", "compute_full_tree", "distance_threshold", "edit"],
                "config_params":["model_data","edit"],
                "validate_params":["params"],

            },
        
        
        
        "KMeans Clustering": {
                "script": kmeans_script,
                "report": kmeans_report,
                "config": kmeans_config,
                "validate": kmeans_validate,
                "script_params": ["df", "features", "k_method", "auto_k", "manual_k", "max_k", "edit"],
                "config_params": ["model_data", "edit"],
                "validate_params": ["params"],
            },
        
        
        
        "Unsupervised KNN Clustering": {
                "script": unsupervised_knn_script,
                "report": unsupervised_knn_report,
                "config": unsupervised_knn_config,
                "validate": unsupervised_knn_validate,
                "script_params": ["df", "features", "edit", "use_grid_search", "param_grid", "manual_params", "n_clusters"],
                "config_params": ["model_data", "edit"],
        }
    }

    # Aliases to support shorthand keys from mapping tables
    MODELS["KNN"] = MODELS.get("KNN Classifier")
    MODELS["LR"] = MODELS.get("Logistic Regression")

    model_info = MODELS[model_name]
    
    # Pick function and expected params
    if action not in model_info:
        raise ValueError(f"Action '{action}' not found for model '{model_name}'")
    func = model_info[action]
    if action=="script":
        expected_params = model_info["script_params"]
    elif action =="config":
        expected_params = model_info['config_params']
        # st.write("config")
    elif action =="prepare_report":
        expected_params = model_info['prepare_report_params']
    elif action =="validate":
        expected_params = ["params"]
        # model_info['validate_params']
        # st.write("validate")
    elif action == "model_description":
        fun = None
        return  model_info['model_description']
    elif action == "model_reference":
        fun = None
        return model_info['model_reference']

    else : expected_params ={}
    # st.write(param_dict)
    # Extract only the needed params for this model
    args = {k: param_dict[k] for k in expected_params if k in param_dict}

    # st.write(args)
    # Call the model function dynamically
    return func(**args)

def get_model_data(model_name,action):
    MODELS = {
#----------------------------------------------------------Classification----------------------------------------------------------
        
        "Decision Tree Classifier": {
            "model_description": model_description,
            "model_reference": model_reference_code
        },
        
        "Gradient Boosting Classifier": {
            "model_description": model_description,
            "model_reference": model_reference_code
        },
        "KNN Classifier": {
            "model_description":model_description,
            "model_reference":model_reference_code
        },

        "Logistic Regression": {
            "model_description":model_description,
            "model_reference":model_reference_code
        },

        "Naive Bayes Classifier": {
            "model_description": model_description,
            "model_reference": model_reference_code
        },

        "Random Forest Classifier": {
            "model_description": model_description,
            "model_reference": model_reference_code
        },
        "Support Vector Machine Classifier": {
            "model_description": model_description,
            "model_reference": model_reference_code
        },
        "XGBoost Classifier": {
            "model_description": model_description,
            "model_reference": model_reference_code
        },
    
#----------------------------------------------------------Regression----------------------------------------------------------
        
        "Gradient Boosting Regressor": {
            "model_description": model_description,
            "model_reference": model_reference_code
        },
        
        
        "KNN_Regressor": {
            "model_description": model_description,
            "model_reference": model_reference_code
        },


        "Random Forest Regression":{
            "model_description":model_description,
            "model_reference":model_reference_code
        },

             
        "Linear Regression": {
            "model_description":model_description,
            "model_reference":model_reference_code
        },
      

        "Support Vector Regression": {
            "model_description":model_description,
            "model_reference":model_reference_code
        },
    
        "XGBoost Regressor": {
            "model_description": model_description,
            "model_reference": model_reference_code
        },


#----------------------------------------------------------Unsupervised----------------------------------------------------------
        
        "DBSCAN": {
            "model_description": model_description,
            "model_reference": model_reference_code
        },
        
        
        "Hierarchical Clustering": {
            "model_description":model_description,
            "model_reference":  model_reference_code
        },

        
    
        
        "KMeans Clustering": {
            "model_description":model_description,
            "model_reference":model_reference_code
        },
        
        
        "Unsupervised KNN Clustering": {
            "model_description": model_description,
            "model_reference": model_reference_code
        },
    
    }
    
    # Aliases to support shorthand keys from mapping tables
    MODELS["KNN"] = MODELS.get("KNN Classifier")
    MODELS["LR"] = MODELS.get("Logistic Regression")
    
     # Pick function and expected params
    if model_name not in MODELS:
        return "No description available for this model."
    model_info = MODELS[model_name]
    
    if action == "model_description":
        return  model_info['model_description']
    elif action == "model_reference":
        return model_info['model_reference']