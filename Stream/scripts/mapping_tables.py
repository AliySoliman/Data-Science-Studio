# mapping_tables.py
# This file maps AI recommendation terms to actual system function calls and categories.

# Maps pipeline_decisions["transform"] strategies -> transformation_execution types/categories
TRANSFORM_STRATEGY_MAP = {
    # Exact matches
    "StandardScaler (Z-score)": {"type": "standardization", "category": "Z-score Standardization"},
    "Z-score Standardization": {"type": "standardization", "category": "Z-score Standardization"},
    "Z-Score": {"type": "standardization", "category": "Z-score Standardization"},
    "Z-score": {"type": "standardization", "category": "Z-score Standardization"},
    "Log / Power Transform": {"type": "transformation", "category": "computation"},
    "Log Transform": {"type": "transformation", "category": "computation"},
    "Log-Transform": {"type": "transformation", "category": "computation"},
    "RobustScaler (IQR-based)": {"type": "standardization", "category": "Robust Scaler"},
    "RobustScaler": {"type": "standardization", "category": "Robust Scaler"},
    "Robust Scaler": {"type": "standardization", "category": "Robust Scaler"},
    "Log-Transform + RobustScaler": {"type": "standardization", "category": "Robust Scaler"},
    "MinMax Standardization": {"type": "standardization", "category": "MinMax Standardization"},
    "MinMaxScaler": {"type": "standardization", "category": "MinMax Standardization"},
    "Min-Max Normalization": {"type": "standardization", "category": "MinMax Standardization"},
    "Normalization": {"type": "standardization", "category": "MinMax Standardization"},
    "Standard Normalization": {"type": "standardization", "category": "Z-score Standardization"},
}

# Maps pipeline_decisions["encode"] strategies -> transformation_execution types/categories
ENCODE_STRATEGY_MAP = {
    # Exact matches
    "Binary Mapping (0/1)": {"type": "encoding", "category": "Binary Encoding"},
    "Binary Encoding": {"type": "encoding", "category": "Binary Encoding"},
    "One-Hot Encoding": {"type": "encoding", "category": "one-hot encoding"},
    "One Hot Encoding": {"type": "encoding", "category": "one-hot encoding"},
    "OneHot": {"type": "encoding", "category": "one-hot encoding"},
    "Target or Frequency Encoding": {"type": "encoding", "category": "Target Encoding"},
    "Target Encoding": {"type": "encoding", "category": "Target Encoding"},
    "Frequency Encoding": {"type": "encoding", "category": "Label Encoding"},
    "Label Encoding": {"type": "encoding", "category": "Label Encoding"},
    "Ordinal Encoding": {"type": "encoding", "category": "Ordinal Encoding"},
    # Fuzzy AI variants — map to closest available encoding
    "Hash Encoding": {"type": "encoding", "category": "Binary Encoding"},
    "Hash Encoding (with rare levels grouped)": {"type": "encoding", "category": "Binary Encoding"},
    "PCA-reduced Binary": {"type": "encoding", "category": "Binary Encoding"},
    "PCA-reduced Binary (with rare levels grouped)": {"type": "encoding", "category": "Binary Encoding"},
    "Hash Encoding or PCA-reduced Binary (with rare levels grouped)": {"type": "encoding", "category": "Binary Encoding"},
    "Target or Frequency Encoding (with rare levels grouped)": {"type": "encoding", "category": "Target Encoding"},
    "Target Encoding (with rare levels grouped)": {"type": "encoding", "category": "Target Encoding"},
    "Frequency Encoding (with rare levels grouped)": {"type": "encoding", "category": "Label Encoding"},
    "Ordinal Encoding (custom order)": {"type": "encoding", "category": "Ordinal Encoding"},
    "Ordinal Encoding (natural order)": {"type": "encoding", "category": "Ordinal Encoding"},
    "Binary / Label Encoding": {"type": "encoding", "category": "Label Encoding"},
    "Label / Ordinal Encoding": {"type": "encoding", "category": "Ordinal Encoding"},
}

# Maps model names from recommendations -> exact keys in models_execution.py MODELS dict
MODEL_NAME_MAP = {
    # Random Forest
    "Random Forest Classifier": "Random Forest Classifier",
    "Random Forest Regressor": "Random Forest Regression",
    "Random Forest": "Random Forest Classifier",          # ambiguous → default classifier
    # XGBoost
    "XGBoost Classifier": "XGBoost Classifier",
    "XGBoost Regressor": "XGBoost Regressor",
    "XGBoost": "XGBoost Classifier",
    "XGB Classifier": "XGBoost Classifier",
    "XGB Regressor": "XGBoost Regressor",
    # Logistic Regression
    "Logistic Regression": "LR",
    "Logistic": "LR",
    # KNN
    "KNN": "KNN",
    "KNN Classifier": "KNN",
    "K-Nearest Neighbors": "KNN",
    "K-Nearest Neighbour": "KNN",
    "KNN Regressor": "KNN_Regressor",
    "KNN Regression": "KNN_Regressor",
    # Decision Tree
    "Decision Tree": "Decision Tree Classifier",
    "Decision Tree Classifier": "Decision Tree Classifier",
    "Decision Tree Regressor": "Decision Tree Regression",
    "Decision Tree Regression": "Decision Tree Regression",
    "SVR": "Support Vector Regression",
    "Support Vector Regression": "Support Vector Regression",
    # Gradient Boosting
    "Gradient Boosting Classifier": "Gradient Boosting Classifier",
    "Gradient Boosting Regressor": "Gradient Boosting Regressor",
    "Gradient Boosting": "Gradient Boosting Classifier",
    "GBM": "Gradient Boosting Classifier",
    "GBM Regressor": "Gradient Boosting Regressor",
    # SVM / SVC
    "SVC": "Support Vector Machine Classifier",
    "SVM": "Support Vector Machine Classifier",
    "Support Vector Machine": "Support Vector Machine Classifier",
    "Support Vector Classifier": "Support Vector Machine Classifier",
    "Support Vector Machine Classifier": "Support Vector Machine Classifier",
    # Naive Bayes
    "Naive Bayes": "Naive Bayes Classifier",
    "Naive Bayes Classifier": "Naive Bayes Classifier",
    "GaussianNB": "Naive Bayes Classifier",
    "Gaussian Naive Bayes": "Naive Bayes Classifier",
    # Linear Regression
    "Linear Regression": "Linear Regression",
    "Ridge Regression": "Linear Regression",
    "Lasso Regression": "Linear Regression",
    # Clustering
    "KMeans": "KMeans Clustering",
    "K-Means": "KMeans Clustering",
    "K-Means Clustering": "KMeans Clustering",
    "KMeans Clustering": "KMeans Clustering",
    "DBSCAN": "DBSCAN",
    "Hierarchical Clustering": "Hierarchical Clustering",
    "Agglomerative Clustering": "Hierarchical Clustering",
}

# Default parameters for each model (used when not grid searching)
MODEL_DEFAULT_PARAMS = {
    "Random Forest Classifier": {"use_grid_search": False, "manual_params": {"n_estimators": 100, "max_depth": 5}, "cv_folds": 5},
    "Random Forest Regression": {"use_grid_search": False, "manual_params": {"n_estimators": 100, "max_depth": 5}, "cv_folds": 5},
    "XGBoost Classifier": {"use_grid_search": False, "manual_params": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3}, "cv_folds": 5},
    "XGBoost Regressor": {"use_grid_search": False, "manual_params": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3}, "cv_folds": 5},
    "LR": {"use_grid_search": False, "manual_params": {"C": 1.0, "penalty": "l2", "solver": "lbfgs"}, "cv_folds": 5, "max_iter": 100},
    "KNN": {"use_grid_search": False, "manual_params": {"n_neighbors": 5, "weights": "uniform", "metric": "minkowski"}, "cv_folds": 5},
    "KNN_Regressor": {"use_grid_search": False, "manual_params": {"n_neighbors": 5, "weights": "uniform"}, "cv_folds": 5},
    "Decision Tree Classifier": {"use_grid_search": False, "max_depth": 5, "min_samples_leaf": 1, "random_state": 42},
    "Gradient Boosting Classifier": {"use_grid_search": False, "manual_params": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3}, "cv_folds": 5},
    "Gradient Boosting Regressor": {"use_grid_search": False, "manual_params": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3}, "cv_folds": 5},
    "Support Vector Machine Classifier": {"use_grid_search": False, "manual_params": {"C": 1.0, "kernel": "rbf", "gamma": "scale"}, "cv_folds": 5},
    "Naive Bayes Classifier": {"use_grid_search": False, "manual_params": {"var_smoothing": 1e-9}, "cv_folds": 5, "model_type": "gaussian"},
    "KMeans Clustering": {"k_method": "auto", "auto_k": True, "manual_k": 3, "max_k": 10, "edit": False},
    "DBSCAN": {"use_grid_search": False, "manual_params": {"eps": 0.5, "min_samples": 5}, "cv_folds": 5},
    "Hierarchical Clustering": {"n_clusters": 3, "linkage": "ward", "metric": "euclidean", "compute_full_tree": "auto", "distance_threshold": None, "edit": False},
    "Linear Regression": {"model_type": "linear", "alpha": 1.0, "edit": False},
}

# Maps model keys -> list of reporting choices (function_name and user_title)
MODEL_REPORTING_MAP = {
    "KNN": [
        {"function_name": "create_ml_summary_text", "user_title": "Summary Text"},
        {"function_name": "create_performance_metrics_plot", "user_title": "Performance Metrics"},
        {"function_name": "create_confusion_matrix_plot", "user_title": "Confusion Matrix"},
        {"function_name": "create_classification_report_table", "user_title": "Classification Report"}
    ],
    "KNN_Regressor": [
        {"function_name": "create_ml_summary_text", "user_title": "Summary Text"},
        {"function_name": "create_performance_metrics_plot", "user_title": "Performance Metrics"},
        {"function_name": "create_knn_prediction_analysis_plot", "user_title": "Prediction Analysis"},
        {"function_name": "create_knn_residual_analysis_plot", "user_title": "Residual Analysis"},
        {"function_name": "create_classification_report_table", "user_title": "Regression Report"}
    ],
    "LR": [
        {"function_name": "create_ml_summary_text", "user_title": "Summary Text"},
        {"function_name": "create_performance_metrics_plot", "user_title": "Performance Metrics"},
        {"function_name": "create_confusion_matrix_plot", "user_title": "Confusion Matrix"},
        {"function_name": "create_classification_report_table", "user_title": "Classification Report"},
        {"function_name": "create_roc_curve_plot_wrapper", "user_title": "ROC Curve"},
        {"function_name": "create_hyperparameters_table", "user_title": "Hyperparameters"},
        {"function_name": "create_dataset_info_table", "user_title": "Dataset Info"},
        {"function_name": "create_performance_interpretation", "user_title": "Performance Interpretation"}
    ],
    "Decision Tree Regression": [
        {"function_name": "create_ml_summary_text", "user_title": "Summary Text"},
        {"function_name": "create_performance_metrics_plot", "user_title": "Performance Metrics"},
        {"function_name": "create_prediction_analysis_plot", "user_title": "Prediction Analysis"},
        {"function_name": "create_feature_importance_plot", "user_title": "Feature Importance"}
    ],
    "Support Vector Regression": [
        {"function_name": "create_ml_summary_text", "user_title": "Summary Text"},
        {"function_name": "create_performance_metrics_plot", "user_title": "Performance Metrics"},
        {"function_name": "create_prediction_analysis_plot", "user_title": "Prediction Analysis"}
    ],
    "Gradient Boosting Regressor": [
        {"function_name": "create_ml_summary_text", "user_title": "Summary Text"},
        {"function_name": "create_performance_metrics_plot", "user_title": "Performance Metrics"},
        {"function_name": "create_error_metrics_plot", "user_title": "Error Analysis"},
        {"function_name": "create_prediction_analysis_plot", "user_title": "Prediction Analysis"},
        {"function_name": "create_residual_analysis_plot", "user_title": "Residual Analysis"},
        {"function_name": "create_classification_report_table", "user_title": "Regression Report"}
    ],
    "DBSCAN": [
        {"function_name": "create_ml_summary_text", "user_title": "Summary Text"},
        {"function_name": "create_cluster_distribution_plot", "user_title": "Cluster Distribution"},
        {"function_name": "create_quality_metrics_plot", "user_title": "Quality Metrics"},
        {"function_name": "create_parameter_importance_plot", "user_title": "Parameter Importance"},
        {"function_name": "create_classification_report_table", "user_title": "Cluster Report"}
    ],
    "Naive Bayes Classifier": [
        {"function_name": "create_summary_text", "user_title": "Summary Text"},
        {"function_name": "create_performance_metrics_plot", "user_title": "Performance Metrics"},
        {"function_name": "create_confusion_matrix_plot", "user_title": "Confusion Matrix"},
        {"function_name": "create_class_distribution_plot", "user_title": "Class Distribution"},
        {"function_name": "create_classification_report_table", "user_title": "Classification Report"}
    ],
    "Support Vector Machine Classifier": [
        {"function_name": "create_summary_text", "user_title": "Summary Text"},
        {"function_name": "create_performance_metrics_plot", "user_title": "Performance Metrics"},
        {"function_name": "create_confusion_matrix_plot", "user_title": "Confusion Matrix"},
        {"function_name": "create_per_class_plot", "user_title": "Per-Class Performance"},
        {"function_name": "create_model_config_block", "user_title": "Model Configuration"},
        {"function_name": "create_performance_interpretation", "user_title": "Performance Interpretation"},
        {"function_name": "create_classification_report_table", "user_title": "Classification Report"}
    ],
    "Random Forest Classifier": [
        {"function_name": "create_rf_summary_text", "user_title": "Summary Text"},
        {"function_name": "create_rf_performance_plot", "user_title": "Performance Metrics"},
        {"function_name": "create_rf_confusion_matrix_plot", "user_title": "Confusion Matrix"},
        {"function_name": "create_rf_feature_importance_plot", "user_title": "Feature Importance"},
        {"function_name": "create_rf_classification_report_table", "user_title": "Classification Report"}
    ],
    "Gradient Boosting Classifier": [
        {"function_name": "create_gbc_summary_text", "user_title": "Summary Text"},
        {"function_name": "create_gbc_performance_plot", "user_title": "Performance Metrics"},
        {"function_name": "create_gbc_confusion_matrix_plot", "user_title": "Confusion Matrix"},
        {"function_name": "create_gbc_feature_importance_plot", "user_title": "Feature Importance"},
        {"function_name": "create_gbc_classification_report_table", "user_title": "Classification Report"}
    ],
    "KMeans Clustering": [
        {"function_name": "create_clustering_summary_text", "user_title": "Summary Text"},
        {"function_name": "create_cluster_distribution_plot", "user_title": "Cluster Distribution"},
        {"function_name": "create_metrics_radar_chart", "user_title": "Metrics Radar Chart"},
        {"function_name": "create_performance_metrics_table", "user_title": "Performance Metrics"},
        {"function_name": "create_cluster_analysis_text", "user_title": "Cluster Analysis"},
        {"function_name": "create_configuration_details_text", "user_title": "Configuration Details"}
    ],
    "Linear Regression": [
        {"function_name": "create_regression_summary_text", "user_title": "Summary Text"},
        {"function_name": "create_performance_comparison", "user_title": "Performance Comparison"},
        {"function_name": "create_residuals_plot", "user_title": "Residuals Plot"},
        {"function_name": "create_coefficients_table", "user_title": "Model Coefficients"},
        {"function_name": "create_feature_importance_plot", "user_title": "Feature Importance"}
    ],
    "Hierarchical Clustering": [
        {"function_name": "get_common_metrics", "user_title": "Summary Metrics"},
        {"function_name": "create_dendrogram_plot", "user_title": "Dendrogram"},
        {"function_name": "create_cluster_ditribution_plot", "user_title": "Cluster Distribution"},
        {"function_name": "create_PCA_Projection_plot", "user_title": "PCA Projection"},
        {"function_name": "create_silhouette_plot", "user_title": "Silhouette Plot"}
    ],
    "Random Forest Regression": [
        {"function_name": "create_ml_summary_text", "user_title": "Summary Text"},
        {"function_name": "create_feature_importance_plot", "user_title": "Feature Importance"},
        {"function_name": "create_prediction_analysis_plot", "user_title": "Prediction Analysis"},
        {"function_name": "create_residual_analysis_plot", "user_title": "Residual Analysis"},
        {"function_name": "model_configuration_insights", "user_title": "Model Insights"},
        {"function_name": "create_model_recommendations", "user_title": "Model Recommendations"}
    ],
    "XGBoost Classifier": [
        {"function_name": "create_xgb_summary_text", "user_title": "Summary Text"},
        {"function_name": "create_xgb_performance_plot", "user_title": "Performance Metrics"},
        {"function_name": "create_xgb_confusion_matrix_plot", "user_title": "Confusion Matrix"},
        {"function_name": "create_xgb_feature_importance_plot", "user_title": "Feature Importance"},
        {"function_name": "create_xgb_classification_report_table", "user_title": "Classification Report"}
    ],
    "XGBoost Regressor": [
        {"function_name": "create_xgb_summary_text", "user_title": "Summary Text"},
        {"function_name": "create_xgb_performance_metrics_plot", "user_title": "Performance Metrics"},
        {"function_name": "create_xgb_error_metrics_plot", "user_title": "Error Analysis"},
        {"function_name": "create_xgb_prediction_analysis_plot", "user_title": "Prediction Analysis"},
        {"function_name": "create_xgb_residual_analysis_plot", "user_title": "Residual Analysis"},
        {"function_name": "create_xgb_regression_report_table", "user_title": "Regression Report"}
    ],
    "Decision Tree Classifier": [
        {"function_name": "create_dt_summary_text", "user_title": "Summary Text"},
        {"function_name": "create_dt_performance_plot", "user_title": "Performance Metrics"},
        {"function_name": "create_dt_confusion_matrix_plot", "user_title": "Confusion Matrix"},
        {"function_name": "create_dt_tree_structure_plot", "user_title": "Tree Structure"},
        {"function_name": "create_dt_feature_importance_plot", "user_title": "Feature Importance"}
    ],
    "Visualization": [
        {"function_name": "create_visualization_plot", "user_title": "Visualization Plot"},
        {"function_name": "create_visualization_comments", "user_title": "Visualization Comments"}
    ]
}


